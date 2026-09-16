use anyhow::{Context, Result, anyhow};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

const MODEL_REPO: &str = "Supertone/supertonic-3";

/// Default pinned HuggingFace revision (immutable commit SHA). Overridable
/// via `MODEL_REVISION`; never defaults to the mutable `main` branch.
pub const DEFAULT_MODEL_REVISION: &str = "3cadd1ee6394adea1bd021217a0e650ede09a323";

const MODEL_FILES: &[&str] = &[
    "onnx/duration_predictor.onnx",
    "onnx/text_encoder.onnx",
    "onnx/vector_estimator.onnx",
    "onnx/vocoder.onnx",
    "onnx/unicode_indexer.json",
    "onnx/tts.json",
    "config.json",
    "voice_styles/M1.json",
    "voice_styles/M2.json",
    "voice_styles/M3.json",
    "voice_styles/M4.json",
    "voice_styles/M5.json",
    "voice_styles/F1.json",
    "voice_styles/F2.json",
    "voice_styles/F3.json",
    "voice_styles/F4.json",
    "voice_styles/F5.json",
];

pub struct ModelPaths {
    pub duration_predictor: std::path::PathBuf,
    pub text_encoder: std::path::PathBuf,
    pub vector_estimator: std::path::PathBuf,
    pub vocoder: std::path::PathBuf,
    pub unicode_indexer: std::path::PathBuf,
    pub tts_config: std::path::PathBuf,
    pub voice_files: Vec<(String, std::path::PathBuf)>,
}

/// Expected SHA-256 digests keyed by repository-relative filename.
pub type ExpectedHashes = HashMap<String, String>;

/// Load an expected-hashes manifest: JSON object mapping
/// `"<repo-relative path>" -> "<sha256 hex>"`.
pub fn load_expected_hashes(path: &str) -> Result<ExpectedHashes> {
    let data = std::fs::read_to_string(path)
        .with_context(|| format!("cannot read model hashes file '{path}'"))?;
    let hashes: ExpectedHashes =
        serde_json::from_str(&data).with_context(|| "model hashes file is not valid JSON")?;
    for (name, digest) in &hashes {
        if digest.len() != 64 || !digest.chars().all(|c| c.is_ascii_hexdigit()) {
            anyhow::bail!("invalid sha256 for '{name}': must be 64 hex chars");
        }
    }
    Ok(hashes)
}

pub fn sha256_of_file(path: &Path) -> Result<String> {
    let mut file = std::fs::File::open(path)?;
    let mut hasher = Sha256::new();
    std::io::copy(&mut file, &mut hasher)?;
    Ok(hex::encode(hasher.finalize()))
}

fn sidecar_path(dest: &Path) -> PathBuf {
    let mut name = dest.file_name().map(|n| n.to_owned()).unwrap_or_default();
    name.push(".sha256");
    dest.with_file_name(name)
}

/// Verify a cached file: against the expected manifest when available,
/// otherwise against the sidecar digest written at download time
/// (trust-on-first-use tamper evidence for the local cache).
fn verify_cached_file(
    filename: &str,
    local_path: &Path,
    expected: Option<&ExpectedHashes>,
) -> Result<bool> {
    if !local_path.is_file()
        || !std::fs::metadata(local_path)
            .map(|m| m.len() > 0)
            .unwrap_or(false)
    {
        return Ok(false);
    }
    let actual = sha256_of_file(local_path)?;
    if let Some(hashes) = expected
        && let Some(want) = hashes.get(filename)
    {
        if constant_time_eq_str(&actual, want) {
            return Ok(true);
        }
        tracing::warn!("hash mismatch for cached {filename}: expected manifest digest");
        return Ok(false);
    }
    // No manifest: compare against the sidecar recorded at download time.
    let sidecar = sidecar_path(local_path);
    match std::fs::read_to_string(&sidecar) {
        Ok(recorded) if constant_time_eq_str(actual.trim(), recorded.trim()) => Ok(true),
        Ok(_) => {
            tracing::warn!("hash mismatch for cached {filename}: differs from recorded digest");
            Ok(false)
        }
        Err(_) => {
            // Legacy cache without a sidecar: record current digest
            // (trust-on-first-use) so future tampering is detected.
            tracing::warn!(
                "no recorded digest for cached {filename}; recording current hash (trust-on-first-use)"
            );
            if std::fs::write(&sidecar, &actual).is_err() {
                tracing::warn!("could not write sidecar digest for {filename}");
            }
            Ok(true)
        }
    }
}

fn constant_time_eq_str(a: &str, b: &str) -> bool {
    a.len() == b.len() && {
        let mut diff = 0u8;
        for (x, y) in a.bytes().zip(b.bytes()) {
            diff |= x ^ y;
        }
        diff == 0
    }
}

async fn download_file(
    client: &reqwest::Client,
    url: &str,
    dest: &std::path::Path,
    hf_token: Option<&str>,
) -> Result<()> {
    let mut req = client.get(url);
    if let Some(token) = hf_token {
        req = req.bearer_auth(token);
    }

    let resp = req.send().await?;
    let status = resp.status();
    if !status.is_success() {
        anyhow::bail!("HTTP {status} for {url}");
    }

    if let Some(parent) = dest.parent() {
        tokio::fs::create_dir_all(parent).await?;
    }

    let temporary = dest.with_extension("download");
    let result = async {
        let mut file = tokio::fs::File::create(&temporary).await?;
        let mut stream = resp.bytes_stream();
        use futures_util::StreamExt;
        use tokio::io::AsyncWriteExt;
        while let Some(chunk) = stream.next().await {
            file.write_all(&chunk?).await?;
        }
        file.flush().await?;
        file.sync_all().await?;
        tokio::fs::rename(&temporary, dest).await?;
        Ok::<(), anyhow::Error>(())
    }
    .await;
    if result.is_err() {
        let _ = tokio::fs::remove_file(&temporary).await;
    }
    result
}

/// Download (or reuse cached) model files pinned to `revision`.
///
/// Every reused or downloaded file is integrity-checked: against
/// `expected_hashes` when provided, otherwise against the sidecar digest
/// recorded at download time. Mismatches trigger deletion + redownload; a
/// persistent mismatch fails safely instead of loading an unverified model.
pub async fn download_models_with_options<F>(
    cache_dir: &Path,
    hf_token: Option<&str>,
    revision: &str,
    expected_hashes: Option<&ExpectedHashes>,
    on_progress: F,
) -> Result<ModelPaths>
where
    F: Fn(f32) + Send + Sync + 'static,
{
    if revision.trim().is_empty() {
        anyhow::bail!("model revision must not be empty");
    }
    std::fs::create_dir_all(cache_dir)?;
    let cache_path = cache_dir
        .canonicalize()
        .unwrap_or_else(|_| std::env::current_dir().unwrap().join(cache_dir));

    let client = reqwest::Client::builder()
        .user_agent("SonicBoom/0.1")
        .build()?;

    let total = MODEL_FILES.len();
    let mut downloaded = 0usize;
    let mut paths: std::collections::HashMap<String, std::path::PathBuf> = Default::default();

    for &filename in MODEL_FILES {
        // Preserve subdirectory structure from filename
        let local_path = cache_path.join(filename.replace('/', std::path::MAIN_SEPARATOR_STR));

        // Reuse only verified non-empty regular files.
        match verify_cached_file(filename, &local_path, expected_hashes) {
            Ok(true) => {
                tracing::info!("Already cached (verified): {filename}");
                paths.insert(filename.to_string(), local_path);
                downloaded += 1;
                on_progress(downloaded as f32 / total as f32);
                continue;
            }
            Ok(false) => {}
            Err(e) => {
                tracing::warn!("could not verify cached {filename}: {e}");
            }
        }
        let _ = tokio::fs::remove_file(&local_path).await;

        let url = format!("https://huggingface.co/{MODEL_REPO}/resolve/{revision}/{filename}");
        tracing::info!("Downloading {filename} (revision {revision})...");

        const MAX_RETRIES: u32 = 5;
        let mut last_err = None;
        let mut success = false;
        for attempt in 0..MAX_RETRIES {
            match download_file(&client, &url, &local_path, hf_token).await {
                Ok(()) => {
                    // Verify what we just wrote before accepting it.
                    let digest = sha256_of_file(&local_path)?;
                    if let Some(hashes) = expected_hashes
                        && let Some(want) = hashes.get(filename)
                        && !constant_time_eq_str(&digest, want)
                    {
                        tracing::warn!(
                            "Downloaded {filename} failed hash verification (attempt {})",
                            attempt + 1
                        );
                        let _ = tokio::fs::remove_file(&local_path).await;
                        last_err = Some(anyhow!("hash mismatch for downloaded {filename}"));
                    } else {
                        // Record the digest for future cache verification.
                        let sidecar = sidecar_path(&local_path);
                        if let Err(e) = std::fs::write(&sidecar, &digest) {
                            tracing::warn!("could not write digest sidecar: {e}");
                        }
                        success = true;
                        break;
                    }
                }
                Err(e) => {
                    tracing::warn!(
                        "Download attempt {}/{MAX_RETRIES} failed for {filename}: {e}",
                        attempt + 1
                    );
                    // Remove incompletely created files
                    let _ = tokio::fs::remove_file(&local_path).await;
                    last_err = Some(e);
                    tokio::time::sleep(std::time::Duration::from_secs(2u64.pow(attempt))).await;
                }
            }
        }

        if !success {
            return Err(last_err.unwrap());
        }

        paths.insert(filename.to_string(), local_path);
        downloaded += 1;
        on_progress(downloaded as f32 / total as f32);
    }

    let voice_files: Vec<(String, std::path::PathBuf)> =
        ["M1", "M2", "M3", "M4", "M5", "F1", "F2", "F3", "F4", "F5"]
            .iter()
            .filter_map(|&name| {
                let key = format!("voice_styles/{name}.json");
                paths.get(&key).map(|p| (name.to_string(), p.clone()))
            })
            .collect();

    Ok(ModelPaths {
        duration_predictor: paths["onnx/duration_predictor.onnx"].clone(),
        text_encoder: paths["onnx/text_encoder.onnx"].clone(),
        vector_estimator: paths["onnx/vector_estimator.onnx"].clone(),
        vocoder: paths["onnx/vocoder.onnx"].clone(),
        unicode_indexer: paths["onnx/unicode_indexer.json"].clone(),
        tts_config: paths["onnx/tts.json"].clone(),
        voice_files,
    })
}

/// Backwards-compatible entry point using the pinned default revision and
/// sidecar-based cache verification.
pub async fn download_models<F>(
    cache_dir: &Path,
    hf_token: Option<&str>,
    on_progress: F,
) -> Result<ModelPaths>
where
    F: Fn(f32) + Send + Sync + 'static,
{
    download_models_with_options(
        cache_dir,
        hf_token,
        DEFAULT_MODEL_REVISION,
        None,
        on_progress,
    )
    .await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sha256_of_file_matches_known_digest() {
        let path = std::env::temp_dir().join(format!("sonicboom-hash-{}.bin", std::process::id()));
        std::fs::write(&path, b"abc").unwrap();
        let digest = sha256_of_file(&path).unwrap();
        assert_eq!(
            digest,
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    fn manifest_mismatch_rejects_cached_file() {
        let dir = std::env::temp_dir().join(format!("sonicboom-verify-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let file = dir.join("model.onnx");
        std::fs::write(&file, b"tampered-bytes").unwrap();
        let mut hashes = ExpectedHashes::new();
        hashes.insert(
            "model.onnx".to_string(),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad".to_string(),
        );
        assert!(!verify_cached_file("model.onnx", &file, Some(&hashes)).unwrap());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn manifest_match_accepts_cached_file() {
        let dir = std::env::temp_dir().join(format!("sonicboom-verify-ok-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let file = dir.join("model.onnx");
        std::fs::write(&file, b"abc").unwrap();
        let mut hashes = ExpectedHashes::new();
        hashes.insert(
            "model.onnx".to_string(),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad".to_string(),
        );
        assert!(verify_cached_file("model.onnx", &file, Some(&hashes)).unwrap());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn invalid_manifest_is_rejected() {
        let path =
            std::env::temp_dir().join(format!("sonicboom-manifest-{}.json", std::process::id()));
        std::fs::write(&path, r#"{"a.onnx": "not-a-hash"}"#).unwrap();
        assert!(load_expected_hashes(path.to_str().unwrap()).is_err());
        std::fs::remove_file(&path).unwrap();
    }
}
