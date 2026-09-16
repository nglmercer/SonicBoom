use anyhow::{Context, Result, anyhow};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::path::Path;

const MODEL_REPO: &str = "Supertone/supertonic-3";

/// Default pinned HuggingFace revision (immutable commit SHA). Overridable
/// via `MODEL_REVISION`, which must itself be a 40-hex commit SHA; a custom
/// revision additionally requires an explicit trusted manifest.
pub const DEFAULT_MODEL_REVISION: &str = "3cadd1ee6394adea1bd021217a0e650ede09a323";

/// Built-in trust root: expected SHA-256 digests for every model file at
/// [`DEFAULT_MODEL_REVISION`], compiled into the binary so the default
/// installation is verified out of the box.
const TRUSTED_MODEL_HASHES_JSON: &str = include_str!("../../models.sha256.json");

/// Authoritative list of model files. Every entry must have a trusted digest
/// before any file is accepted (see [`validate_manifest`]).
pub const MODEL_FILES: &[&str] = &[
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

/// Immutable-revision check: exactly 40 hexadecimal characters (a full Git
/// commit SHA). Rejects `main`, tags, short SHAs, and anything else mutable.
pub fn is_valid_commit_sha(value: &str) -> bool {
    value.len() == 40 && value.bytes().all(|b| b.is_ascii_hexdigit())
}

fn validate_sha256(name: &str, digest: &str) -> Result<()> {
    if digest.len() != 64 || !digest.chars().all(|c| c.is_ascii_hexdigit()) {
        anyhow::bail!("invalid sha256 for '{name}': must be 64 hex chars");
    }
    Ok(())
}

/// Validate that a manifest is complete and well-formed: every
/// [`MODEL_FILES`] entry present with a valid digest, and no unexpected keys
/// (source/manifest drift fails loudly).
pub fn validate_manifest(hashes: &ExpectedHashes) -> Result<()> {
    for filename in MODEL_FILES {
        let expected = hashes
            .get(*filename)
            .ok_or_else(|| anyhow!("missing trusted digest for {filename}"))?;
        validate_sha256(filename, expected)?;
    }
    for key in hashes.keys() {
        if !MODEL_FILES.contains(&key.as_str()) {
            anyhow::bail!("unexpected model digest entry: {key}");
        }
    }
    Ok(())
}

/// Parse and validate the compiled-in trusted manifest.
pub fn builtin_trusted_hashes() -> Result<ExpectedHashes> {
    let hashes: ExpectedHashes = serde_json::from_str(TRUSTED_MODEL_HASHES_JSON)
        .context("built-in model hash manifest is not valid JSON")?;
    validate_manifest(&hashes).context("built-in model hash manifest is invalid")?;
    Ok(hashes)
}

/// Load an operator-provided expected-hashes manifest: JSON object mapping
/// `"<repo-relative path>" -> "<sha256 hex>"`. The manifest must be complete
/// (see [`validate_manifest`]); partial manifests fail closed.
pub fn load_expected_hashes(path: &str) -> Result<ExpectedHashes> {
    let metadata = std::fs::metadata(path)
        .with_context(|| format!("cannot read model hashes file '{path}'"))?;
    if !metadata.is_file() {
        anyhow::bail!("model hashes file '{path}' is not a regular file");
    }
    let data = std::fs::read_to_string(path)
        .with_context(|| format!("cannot read model hashes file '{path}'"))?;
    let hashes: ExpectedHashes =
        serde_json::from_str(&data).with_context(|| "model hashes file is not valid JSON")?;
    validate_manifest(&hashes).with_context(|| "model hashes file is incomplete or invalid")?;
    Ok(hashes)
}

/// Resolve the (revision, trusted hashes) pair under one policy shared by the
/// HTTP server and the reusable engine:
///
/// - default revision + no custom manifest → built-in trusted manifest;
/// - any revision + custom manifest → the validated custom manifest;
/// - custom revision + no custom manifest → hard error (never pair a custom
///   revision with another revision's hashes).
pub fn resolve_trust(
    revision: &str,
    custom_manifest_path: Option<&str>,
) -> Result<(String, ExpectedHashes)> {
    if !is_valid_commit_sha(revision) {
        anyhow::bail!("MODEL_REVISION must be a 40-character commit SHA, got '{revision}'");
    }
    if let Some(path) = custom_manifest_path {
        let hashes = load_expected_hashes(path)?;
        return Ok((revision.to_string(), hashes));
    }
    if revision != DEFAULT_MODEL_REVISION {
        anyhow::bail!(
            "custom MODEL_REVISION requires MODEL_SHA256_JSON_PATH with trusted hashes for that exact revision"
        );
    }
    Ok((revision.to_string(), builtin_trusted_hashes()?))
}

pub fn sha256_of_file(path: &Path) -> Result<String> {
    let mut file = std::fs::File::open(path)?;
    let mut hasher = Sha256::new();
    std::io::copy(&mut file, &mut hasher)?;
    Ok(hex::encode(hasher.finalize()))
}

/// Verify a cached file against its previously trusted digest. There is no
/// trust-on-first-use: without a matching trusted digest the file is
/// rejected (the caller deletes and redownloads it).
fn verify_cached_file(
    filename: &str,
    local_path: &Path,
    expected: &ExpectedHashes,
) -> Result<bool> {
    if !local_path.is_file()
        || !std::fs::metadata(local_path)
            .map(|m| m.len() > 0)
            .unwrap_or(false)
    {
        return Ok(false);
    }
    let want = expected
        .get(filename)
        .ok_or_else(|| anyhow!("missing trusted digest for {filename}"))?;
    let actual = sha256_of_file(local_path)?;
    if constant_time_eq_str(&actual, want) {
        return Ok(true);
    }
    tracing::warn!("hash mismatch for cached {filename}: expected trusted digest");
    Ok(false)
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
/// Every file — cached or freshly downloaded — is accepted only after its
/// SHA-256 matches the previously trusted `expected_hashes`. Mismatches
/// trigger deletion + redownload; a persistent mismatch fails instead of
/// loading an unverified model.
pub async fn download_models_with_options<F>(
    cache_dir: &Path,
    hf_token: Option<&str>,
    revision: &str,
    expected_hashes: &ExpectedHashes,
    on_progress: F,
) -> Result<ModelPaths>
where
    F: Fn(f32) + Send + Sync + 'static,
{
    if !is_valid_commit_sha(revision) {
        anyhow::bail!("model revision must be a 40-character commit SHA");
    }
    validate_manifest(expected_hashes)?;
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

        // Reuse only files verified against the trusted manifest.
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
                    let want = expected_hashes
                        .get(filename)
                        .ok_or_else(|| anyhow!("missing trusted digest for {filename}"))?;
                    if !constant_time_eq_str(&digest, want) {
                        tracing::warn!(
                            "Downloaded {filename} failed hash verification (attempt {})",
                            attempt + 1
                        );
                        let _ = tokio::fs::remove_file(&local_path).await;
                        last_err = Some(anyhow!("hash mismatch for downloaded {filename}"));
                    } else {
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

/// Download model files using the default pinned revision and the built-in
/// trusted manifest. Verification is mandatory, never trust-on-first-use.
pub async fn download_models<F>(
    cache_dir: &Path,
    hf_token: Option<&str>,
    on_progress: F,
) -> Result<ModelPaths>
where
    F: Fn(f32) + Send + Sync + 'static,
{
    let hashes = builtin_trusted_hashes()?;
    download_models_with_options(
        cache_dir,
        hf_token,
        DEFAULT_MODEL_REVISION,
        &hashes,
        on_progress,
    )
    .await
}

#[cfg(test)]
mod tests {
    use super::*;

    fn builtin() -> ExpectedHashes {
        builtin_trusted_hashes().expect("built-in manifest must load")
    }

    #[test]
    fn builtin_manifest_is_complete_and_valid() {
        let hashes = builtin();
        assert_eq!(hashes.len(), MODEL_FILES.len());
        assert!(validate_manifest(&hashes).is_ok());
        // Spot-check one pinned digest (config.json at the pinned revision).
        assert_eq!(
            hashes["config.json"],
            "4099082b107a9d4029849ac76b89eca65e03732660969c2babe5bf308c7357f2"
        );
    }

    #[test]
    fn partial_manifest_is_rejected() {
        let mut hashes = builtin();
        hashes.remove("config.json");
        assert!(validate_manifest(&hashes).is_err());
        assert!(validate_manifest(&ExpectedHashes::new()).is_err());
    }

    #[test]
    fn unexpected_manifest_key_is_rejected() {
        let mut hashes = builtin();
        hashes.insert("evil.bin".to_string(), "0".repeat(64));
        assert!(validate_manifest(&hashes).is_err());
    }

    #[test]
    fn malformed_hash_is_rejected() {
        for bad in [
            "not-a-hash",
            &"0".repeat(63),
            &"0".repeat(65),
            &"z".repeat(64),
        ] {
            let mut hashes = builtin();
            hashes.insert("config.json".to_string(), bad.to_string());
            assert!(validate_manifest(&hashes).is_err(), "accepted {bad:?}");
        }
    }

    #[test]
    fn commit_sha_validation() {
        assert!(is_valid_commit_sha(DEFAULT_MODEL_REVISION));
        assert!(is_valid_commit_sha(
            "3CADD1EE6394ADEA1BD021217A0E650EDE09A323"
        ));
        for bad in [
            "",
            "main",
            "master",
            "latest",
            "refs/heads/main",
            "v3",
            "abc123",
            "3cadd1ee6394adea1bd021217a0e650ede09a32", // 39 chars
            "3cadd1ee6394adea1bd021217a0e650ede09a323f", // 41 chars
            "zcadd1ee6394adea1bd021217a0e650ede09a323", // non-hex
        ] {
            assert!(!is_valid_commit_sha(bad), "accepted {bad:?}");
        }
    }

    #[test]
    fn resolve_trust_couples_custom_revisions_to_manifests() {
        // Default revision works out of the box.
        let (rev, hashes) = resolve_trust(DEFAULT_MODEL_REVISION, None).unwrap();
        assert_eq!(rev, DEFAULT_MODEL_REVISION);
        assert_eq!(hashes.len(), MODEL_FILES.len());
        // Custom revision without a custom manifest fails.
        let custom = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
        assert!(resolve_trust(custom, None).is_err());
        // Mutable revision names fail even before manifest checks.
        assert!(resolve_trust("main", None).is_err());
    }

    #[test]
    fn resolve_trust_accepts_complete_custom_manifest() {
        let dir = std::env::temp_dir().join(format!("sonicboom-manifest-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("custom.json");
        let json = serde_json::to_string(&builtin()).unwrap();
        std::fs::write(&path, json).unwrap();
        let custom = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
        let (rev, hashes) = resolve_trust(custom, Some(path.to_str().unwrap())).unwrap();
        assert_eq!(rev, custom);
        assert_eq!(hashes.len(), MODEL_FILES.len());
        // Partial custom manifest fails.
        std::fs::write(&path, r#"{"config.json": "0"}"#).unwrap();
        assert!(resolve_trust(custom, Some(path.to_str().unwrap())).is_err());
        // Directory instead of a file fails.
        assert!(resolve_trust(custom, Some(dir.to_str().unwrap())).is_err());
        std::fs::remove_dir_all(&dir).unwrap();
    }

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
    fn tampered_cached_file_is_rejected_without_tofu_fallback() {
        let dir = std::env::temp_dir().join(format!("sonicboom-verify-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let file = dir.join("model.onnx");
        std::fs::write(&file, b"tampered-bytes").unwrap();
        let mut hashes = ExpectedHashes::new();
        for name in MODEL_FILES {
            hashes.insert(name.to_string(), "0".repeat(64));
        }
        hashes.insert(
            "onnx/vocoder.onnx".to_string(),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad".to_string(),
        );
        // Wrong content for a trusted name is rejected, not recorded.
        assert!(!verify_cached_file("onnx/vocoder.onnx", &file, &hashes).unwrap());
        assert!(
            !dir.join("model.onnx.sha256").exists(),
            "no sidecar trust may be created"
        );
        // Unknown names are rejected even if the file exists.
        assert!(verify_cached_file("unknown.bin", &file, &hashes).is_err());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn known_matching_file_is_accepted() {
        let dir = std::env::temp_dir().join(format!("sonicboom-verify-ok-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let file = dir.join("model.onnx");
        std::fs::write(&file, b"abc").unwrap();
        let mut hashes = ExpectedHashes::new();
        for name in MODEL_FILES {
            hashes.insert(name.to_string(), "0".repeat(64));
        }
        hashes.insert(
            "onnx/vocoder.onnx".to_string(),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad".to_string(),
        );
        assert!(verify_cached_file("onnx/vocoder.onnx", &file, &hashes).unwrap());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn download_rejects_mutable_revision_without_network() {
        // Validation happens before any network/cache work: an invalid
        // revision fails even with an unusable cache dir.
        let hashes = builtin();
        let rt = tokio::runtime::Runtime::new().unwrap();
        let result = rt.block_on(download_models_with_options(
            Path::new("/nonexistent-sonicboom-cache-dir"),
            None,
            "main",
            &hashes,
            |_| {},
        ));
        assert!(result.is_err());
    }
}
