use anyhow::{Context, Result, anyhow};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::path::Path;
use std::time::Duration;

const MODEL_REPO: &str = "Supertone/supertonic-3";

/// Default pinned HuggingFace revision (immutable commit SHA). Overridable
/// via `MODEL_REVISION`, which must itself be a 40-hex commit SHA; a custom
/// revision additionally requires an explicit trusted manifest.
pub const DEFAULT_MODEL_REVISION: &str = "3cadd1ee6394adea1bd021217a0e650ede09a323";

/// Built-in trust root: expected SHA-256 digests (and exact byte sizes) for
/// every model file at [`DEFAULT_MODEL_REVISION`], compiled into the binary
/// so the default installation is verified out of the box.
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

/// Expected exact byte sizes keyed by repository-relative filename. Sizes
/// bound downloads *before* authenticity verification completes; SHA-256
/// remains authoritative for authenticity.
pub type ExpectedSizes = HashMap<String, u64>;

/// Trusted download policy for one revision: hashes plus optional exact
/// sizes. The built-in manifest always carries sizes; operator manifests
/// should too (legacy hash-only manifests are accepted but cannot bound
/// downloads before hashing).
pub struct ExpectedTrust {
    pub hashes: ExpectedHashes,
    pub sizes: Option<ExpectedSizes>,
}

/// Network resource limits for model downloads.
#[derive(Debug, Clone)]
pub struct DownloadLimits {
    /// TCP/TLS connect timeout per file.
    pub connect_timeout: Duration,
    /// Total timeout per file, including the whole body transfer.
    pub timeout: Duration,
}

impl Default for DownloadLimits {
    fn default() -> Self {
        Self {
            connect_timeout: Duration::from_secs(10),
            timeout: Duration::from_secs(1800),
        }
    }
}

impl DownloadLimits {
    pub fn new(connect_timeout_secs: u64, timeout_secs: u64) -> Self {
        Self {
            connect_timeout: Duration::from_secs(connect_timeout_secs.max(1)),
            timeout: Duration::from_secs(timeout_secs.max(1)),
        }
    }
}

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

/// One manifest value: either a bare SHA-256 hex string (legacy) or an
/// object carrying both the digest and the exact expected byte size.
#[derive(Debug, serde::Deserialize)]
#[serde(untagged)]
enum ManifestEntry {
    HashOnly(String),
    WithSize { sha256: String, size: u64 },
}

/// Split a parsed manifest into hashes plus optional sizes. Mixing the two
/// value shapes in one file is rejected: manifests must be uniformly sized
/// or uniformly legacy so enforcement is never silently partial.
fn split_manifest(
    raw: &HashMap<String, ManifestEntry>,
) -> Result<(ExpectedHashes, Option<ExpectedSizes>)> {
    let mut hashes = ExpectedHashes::with_capacity(raw.len());
    let mut sizes = ExpectedSizes::new();
    let mut sized_count = 0usize;
    for (name, entry) in raw {
        match entry {
            ManifestEntry::HashOnly(digest) => {
                hashes.insert(name.clone(), digest.clone());
            }
            ManifestEntry::WithSize { sha256, size } => {
                hashes.insert(name.clone(), sha256.clone());
                sizes.insert(name.clone(), *size);
                sized_count += 1;
            }
        }
    }
    if sized_count > 0 && sized_count != raw.len() {
        anyhow::bail!(
            "model manifest mixes sized and hash-only entries; use one shape consistently"
        );
    }
    let sizes = if sized_count == 0 { None } else { Some(sizes) };
    Ok((hashes, sizes))
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

/// Validate an expected-size table: complete over [`MODEL_FILES`], every
/// size positive, no unexpected keys.
pub fn validate_sizes(sizes: &ExpectedSizes) -> Result<()> {
    for filename in MODEL_FILES {
        let size = sizes
            .get(*filename)
            .ok_or_else(|| anyhow!("missing trusted size for {filename}"))?;
        if *size == 0 {
            anyhow::bail!("trusted size for '{filename}' must be positive");
        }
    }
    for key in sizes.keys() {
        if !MODEL_FILES.contains(&key.as_str()) {
            anyhow::bail!("unexpected model size entry: {key}");
        }
    }
    Ok(())
}

fn parse_manifest_json(data: &str) -> Result<(ExpectedHashes, Option<ExpectedSizes>)> {
    let raw: HashMap<String, ManifestEntry> =
        serde_json::from_str(data).with_context(|| "model manifest is not valid JSON")?;
    let (hashes, sizes) = split_manifest(&raw)?;
    validate_manifest(&hashes).with_context(|| "model manifest is incomplete or invalid")?;
    if let Some(sizes) = &sizes {
        validate_sizes(sizes).with_context(|| "model size table is incomplete or invalid")?;
    }
    Ok((hashes, sizes))
}

/// Parse and validate the compiled-in trusted manifest.
pub fn builtin_trusted_hashes() -> Result<ExpectedHashes> {
    let hashes: ExpectedHashes = builtin_trust()?.hashes;
    Ok(hashes)
}

/// Parse and validate the compiled-in trusted size table.
pub fn builtin_trusted_sizes() -> Result<ExpectedSizes> {
    builtin_trust()?
        .sizes
        .ok_or_else(|| anyhow!("built-in model manifest is missing its size table"))
}

fn builtin_trust() -> Result<ExpectedTrust> {
    let (hashes, sizes) = parse_manifest_json(TRUSTED_MODEL_HASHES_JSON)
        .context("built-in model hash manifest is invalid")?;
    Ok(ExpectedTrust { hashes, sizes })
}

/// Load an operator-provided expected-hashes manifest: JSON object mapping
/// `"<repo-relative path>"` to either `"<sha256 hex>"` or
/// `{"sha256": "<hex>", "size": <bytes>}`. The manifest must be complete
/// (see [`validate_manifest`]); partial manifests fail closed.
pub fn load_expected_hashes(path: &str) -> Result<ExpectedHashes> {
    Ok(load_expected_trust(path)?.hashes)
}

/// Load an operator-provided size table, if the manifest carries one.
/// Returns `None` for legacy hash-only manifests.
pub fn load_expected_sizes(path: &str) -> Result<Option<ExpectedSizes>> {
    Ok(load_expected_trust(path)?.sizes)
}

fn load_expected_trust(path: &str) -> Result<ExpectedTrust> {
    let metadata = std::fs::metadata(path)
        .with_context(|| format!("cannot read model hashes file '{path}'"))?;
    if !metadata.is_file() {
        anyhow::bail!("model hashes file '{path}' is not a regular file");
    }
    let data = std::fs::read_to_string(path)
        .with_context(|| format!("cannot read model hashes file '{path}'"))?;
    let (hashes, sizes) =
        parse_manifest_json(&data).with_context(|| "model hashes file is incomplete or invalid")?;
    Ok(ExpectedTrust { hashes, sizes })
}

/// Resolve the (revision, trusted policy) pair under one policy shared by the
/// HTTP server and the reusable engine:
///
/// - default revision + no custom manifest → built-in trusted manifest
///   (hashes and sizes);
/// - any revision + custom manifest → the validated custom manifest;
/// - custom revision + no custom manifest → hard error (never pair a custom
///   revision with another revision's hashes).
pub fn resolve_trust(
    revision: &str,
    custom_manifest_path: Option<&str>,
) -> Result<(String, ExpectedTrust)> {
    if !is_valid_commit_sha(revision) {
        anyhow::bail!("MODEL_REVISION must be a 40-character commit SHA, got '{revision}'");
    }
    if let Some(path) = custom_manifest_path {
        let trust = load_expected_trust(path)?;
        return Ok((revision.to_string(), trust));
    }
    if revision != DEFAULT_MODEL_REVISION {
        anyhow::bail!(
            "custom MODEL_REVISION requires MODEL_SHA256_JSON_PATH with trusted hashes for that exact revision"
        );
    }
    Ok((revision.to_string(), builtin_trust()?))
}

pub fn sha256_of_file(path: &Path) -> Result<String> {
    let mut file = std::fs::File::open(path)?;
    let mut hasher = Sha256::new();
    std::io::copy(&mut file, &mut hasher)?;
    Ok(hex::encode(hasher.finalize()))
}

/// Verify a cached file against its previously trusted policy. There is no
/// trust-on-first-use: without a matching trusted digest the file is
/// rejected (the caller deletes and redownloads it). When a trusted size is
/// known it is checked first so obviously wrong files are rejected without
/// wasting CPU on a hash.
fn verify_cached_file(filename: &str, local_path: &Path, expected: &ExpectedTrust) -> Result<bool> {
    let metadata = match std::fs::metadata(local_path) {
        Ok(metadata) if metadata.is_file() && metadata.len() > 0 => metadata,
        _ => return Ok(false),
    };
    if let Some(sizes) = &expected.sizes
        && let Some(want_size) = sizes.get(filename)
        && metadata.len() != *want_size
    {
        tracing::warn!(
            "size mismatch for cached {filename}: got {} bytes, want {want_size}",
            metadata.len()
        );
        return Ok(false);
    }
    let want = expected
        .hashes
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

/// Download one file with independent resource controls, then verify size
/// and SHA-256 *before* renaming into the cache:
///
/// - `Content-Length`, when present, must equal the trusted size;
/// - the stream aborts past the trusted size and must match it exactly;
/// - the temp file's SHA-256 must match the trusted digest;
/// - only then is the temp file atomically renamed to `dest`.
///
/// Neither errors nor logs ever include the bearer token: failures report
/// the URL and sizes only.
async fn download_file(
    client: &reqwest::Client,
    url: &str,
    dest: &std::path::Path,
    hf_token: Option<&str>,
    expected_hash: &str,
    expected_size: Option<u64>,
) -> Result<()> {
    let mut req = client.get(url);
    if let Some(token) = hf_token {
        req = req.bearer_auth(token);
    }

    let resp = req.send().await.map_err(|e| {
        // reqwest errors echo the URL; the Authorization header is never
        // part of the message.
        anyhow!("request failed for {url}: {e}")
    })?;
    let status = resp.status();
    if !status.is_success() {
        anyhow::bail!("HTTP {status} for {url}");
    }
    if let (Some(want), Some(len)) = (expected_size, resp.content_length())
        && len != want
    {
        anyhow::bail!("content-length {len} does not match trusted size {want} for {url}");
    }

    if let Some(parent) = dest.parent() {
        tokio::fs::create_dir_all(parent).await?;
    }

    let temporary = dest.with_extension("download");
    let result = async {
        let mut file = tokio::fs::File::create(&temporary).await?;
        let mut stream = resp.bytes_stream();
        let mut received: u64 = 0;
        use futures_util::StreamExt;
        use tokio::io::AsyncWriteExt;
        while let Some(chunk) = stream.next().await {
            let chunk = chunk?;
            received += chunk.len() as u64;
            if let Some(want) = expected_size
                && received > want
            {
                anyhow::bail!(
                    "download exceeded trusted size {want} for {url} (aborted at {received} bytes)"
                );
            }
            file.write_all(&chunk).await?;
        }
        if let Some(want) = expected_size
            && received != want
        {
            anyhow::bail!("download size {received} does not match trusted size {want} for {url}");
        }
        file.flush().await?;
        file.sync_all().await?;
        drop(file);
        // Verify authenticity before the file may appear at its final path.
        let digest = tokio::task::spawn_blocking({
            let temporary = temporary.clone();
            move || sha256_of_file(&temporary)
        })
        .await??;
        if !constant_time_eq_str(&digest, expected_hash) {
            anyhow::bail!("hash mismatch for downloaded {url}");
        }
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
/// SHA-256 matches the previously trusted `expected.hashes`, within
/// `limits`. Mismatches trigger deletion + redownload; a persistent
/// mismatch fails instead of loading an unverified model.
pub async fn download_models_with_options<F>(
    cache_dir: &Path,
    hf_token: Option<&str>,
    revision: &str,
    expected: &ExpectedTrust,
    limits: &DownloadLimits,
    on_progress: F,
) -> Result<ModelPaths>
where
    F: Fn(f32) + Send + Sync + 'static,
{
    if !is_valid_commit_sha(revision) {
        anyhow::bail!("model revision must be a 40-character commit SHA");
    }
    validate_manifest(&expected.hashes)?;
    if let Some(sizes) = &expected.sizes {
        validate_sizes(sizes)?;
    }
    std::fs::create_dir_all(cache_dir)?;
    let cache_path = cache_dir
        .canonicalize()
        .unwrap_or_else(|_| std::env::current_dir().unwrap().join(cache_dir));

    // Explicit timeouts: startup must never hang indefinitely on a network.
    let client = reqwest::Client::builder()
        .user_agent("SonicBoom/0.1")
        .connect_timeout(limits.connect_timeout)
        .timeout(limits.timeout)
        .build()?;

    let total = MODEL_FILES.len();
    let mut downloaded = 0usize;
    let mut paths: std::collections::HashMap<String, std::path::PathBuf> = Default::default();

    for &filename in MODEL_FILES {
        // Preserve subdirectory structure from filename
        let local_path = cache_path.join(filename.replace('/', std::path::MAIN_SEPARATOR_STR));

        // Reuse only files verified against the trusted manifest.
        match verify_cached_file(filename, &local_path, expected) {
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
        let want_hash = expected
            .hashes
            .get(filename)
            .ok_or_else(|| anyhow!("missing trusted digest for {filename}"))?
            .clone();
        let want_size = expected
            .sizes
            .as_ref()
            .and_then(|s| s.get(filename).copied());

        const MAX_RETRIES: u32 = 5;
        let mut last_err = None;
        let mut success = false;
        for attempt in 0..MAX_RETRIES {
            // `download_file` verifies size and hash before rename, so any
            // success here is an authenticated file at its final path.
            match download_file(&client, &url, &local_path, hf_token, &want_hash, want_size).await {
                Ok(()) => {
                    success = true;
                    break;
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
/// trusted manifest (hashes, sizes, and default timeouts). Verification is
/// mandatory, never trust-on-first-use.
pub async fn download_models<F>(
    cache_dir: &Path,
    hf_token: Option<&str>,
    on_progress: F,
) -> Result<ModelPaths>
where
    F: Fn(f32) + Send + Sync + 'static,
{
    let trust = builtin_trust()?;
    download_models_with_options(
        cache_dir,
        hf_token,
        DEFAULT_MODEL_REVISION,
        &trust,
        &DownloadLimits::default(),
        on_progress,
    )
    .await
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::SocketAddr;

    fn builtin() -> ExpectedHashes {
        builtin_trusted_hashes().expect("built-in manifest must load")
    }

    fn builtin_trust_for_test() -> ExpectedTrust {
        builtin_trust().expect("built-in trust must load")
    }

    fn trust_with(hashes: ExpectedHashes, sizes: Option<ExpectedSizes>) -> ExpectedTrust {
        ExpectedTrust { hashes, sizes }
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
    fn builtin_manifest_carries_exact_sizes() {
        let sizes = builtin_trusted_sizes().expect("built-in sizes must load");
        assert_eq!(sizes.len(), MODEL_FILES.len());
        assert!(validate_sizes(&sizes).is_ok());
        // Spot-checks against the pinned revision (verified 2026-09-16
        // against hash-matching local copies of the pinned files).
        assert_eq!(sizes["config.json"], 174);
        assert_eq!(sizes["onnx/vector_estimator.onnx"], 256_534_781);
        assert_eq!(sizes["onnx/vocoder.onnx"], 101_424_195);
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
    fn manifest_shapes_must_not_mix() {
        let mut entries: HashMap<String, ManifestEntry> = HashMap::new();
        for name in MODEL_FILES {
            entries.insert(name.to_string(), ManifestEntry::HashOnly("0".repeat(64)));
        }
        entries.insert(
            "config.json".to_string(),
            ManifestEntry::WithSize {
                sha256: "0".repeat(64),
                size: 174,
            },
        );
        assert!(split_manifest(&entries).is_err());
    }

    #[test]
    fn legacy_hash_only_manifest_loads_without_sizes() {
        let dir = std::env::temp_dir().join(format!("sonicboom-legacy-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("legacy.json");
        let json = serde_json::to_string(&builtin()).unwrap();
        std::fs::write(&path, json).unwrap();
        let trust = load_expected_trust(path.to_str().unwrap()).unwrap();
        assert_eq!(trust.hashes.len(), MODEL_FILES.len());
        assert!(trust.sizes.is_none());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn zero_size_is_rejected() {
        let mut sizes = builtin_trusted_sizes().unwrap();
        sizes.insert("config.json".to_string(), 0);
        assert!(validate_sizes(&sizes).is_err());
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
        let (rev, trust) = resolve_trust(DEFAULT_MODEL_REVISION, None).unwrap();
        assert_eq!(rev, DEFAULT_MODEL_REVISION);
        assert_eq!(trust.hashes.len(), MODEL_FILES.len());
        assert!(trust.sizes.is_some());
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
        let (rev, trust) = resolve_trust(custom, Some(path.to_str().unwrap())).unwrap();
        assert_eq!(rev, custom);
        assert_eq!(trust.hashes.len(), MODEL_FILES.len());
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
        let trust = trust_with(hashes, None);
        // Wrong content for a trusted name is rejected, not recorded.
        assert!(!verify_cached_file("onnx/vocoder.onnx", &file, &trust).unwrap());
        assert!(
            !dir.join("model.onnx.sha256").exists(),
            "no sidecar trust may be created"
        );
        // Unknown names are rejected even if the file exists.
        assert!(verify_cached_file("unknown.bin", &file, &trust).is_err());
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
        let trust = trust_with(hashes, None);
        assert!(verify_cached_file("onnx/vocoder.onnx", &file, &trust).unwrap());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn cached_file_with_wrong_size_is_rejected_before_hashing() {
        let dir =
            std::env::temp_dir().join(format!("sonicboom-verify-size-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let file = dir.join("model.onnx");
        std::fs::write(&file, b"abc").unwrap(); // 3 bytes
        let mut hashes = ExpectedHashes::new();
        let mut sizes = ExpectedSizes::new();
        for name in MODEL_FILES {
            hashes.insert(name.to_string(), "0".repeat(64));
            sizes.insert(name.to_string(), 1);
        }
        // Correct digest for "abc", wrong size: must be rejected without
        // needing the digest comparison to fail.
        hashes.insert(
            "onnx/vocoder.onnx".to_string(),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad".to_string(),
        );
        sizes.insert("onnx/vocoder.onnx".to_string(), 999);
        let trust = trust_with(hashes, Some(sizes));
        assert!(!verify_cached_file("onnx/vocoder.onnx", &file, &trust).unwrap());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn download_rejects_mutable_revision_without_network() {
        // Validation happens before any network/cache work: an invalid
        // revision fails even with an unusable cache dir.
        let trust = builtin_trust_for_test();
        let rt = tokio::runtime::Runtime::new().unwrap();
        let result = rt.block_on(download_models_with_options(
            Path::new("/nonexistent-sonicboom-cache-dir"),
            None,
            "main",
            &trust,
            &DownloadLimits::default(),
            |_| {},
        ));
        assert!(result.is_err());
    }

    // --- Local-server download tests (no external network) ---

    const ABC_SHA256: &str = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";

    /// Serve one canned HTTP response on loopback, then stop.
    async fn serve_once(response: Vec<u8>) -> (SocketAddr, tokio::task::JoinHandle<()>) {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let handle = tokio::spawn(async move {
            use tokio::io::{AsyncReadExt, AsyncWriteExt};
            if let Ok((mut sock, _)) = listener.accept().await {
                let mut buf = [0u8; 8192];
                let _ = sock.read(&mut buf).await;
                let _ = sock.write_all(&response).await;
            }
        });
        (addr, handle)
    }

    fn plain_response(body: &[u8], content_length: Option<usize>) -> Vec<u8> {
        let mut head = b"HTTP/1.1 200 OK\r\nContent-Type: application/octet-stream\r\n".to_vec();
        if let Some(len) = content_length {
            head.extend_from_slice(format!("Content-Length: {len}\r\n").as_bytes());
        }
        head.extend_from_slice(b"Connection: close\r\n\r\n");
        head.extend_from_slice(body);
        head
    }

    fn test_client(limits: &DownloadLimits) -> reqwest::Client {
        reqwest::Client::builder()
            .connect_timeout(limits.connect_timeout)
            .timeout(limits.timeout)
            .build()
            .unwrap()
    }

    fn download_scratch(test: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "sonicboom-download-{test}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[tokio::test]
    async fn content_length_mismatch_is_rejected() {
        let dir = download_scratch("content-length");
        let dest = dir.join("file.bin");
        // Server claims 100 bytes; trusted size is 3.
        let (addr, server) = serve_once(plain_response(b"abc", Some(100))).await;
        let client = test_client(&DownloadLimits::default());
        let err = download_file(
            &client,
            &format!("http://{addr}/file.bin"),
            &dest,
            Some("SECRET-HF-TOKEN"),
            ABC_SHA256,
            Some(3),
        )
        .await
        .expect_err("content-length mismatch must fail");
        assert!(err.to_string().contains("content-length"), "{err}");
        assert!(!err.to_string().contains("SECRET-HF-TOKEN"), "{err}");
        assert!(!dest.exists());
        assert!(!dir.join("file.download").exists());
        server.abort();
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[tokio::test]
    async fn oversized_stream_is_aborted_and_cleaned() {
        let dir = download_scratch("oversize");
        let dest = dir.join("file.bin");
        // No content-length; 10 streaming bytes against a trusted size of 3.
        let (addr, server) = serve_once(plain_response(b"0123456789", None)).await;
        let client = test_client(&DownloadLimits::default());
        let err = download_file(
            &client,
            &format!("http://{addr}/file.bin"),
            &dest,
            None,
            ABC_SHA256,
            Some(3),
        )
        .await
        .expect_err("oversized stream must abort");
        assert!(err.to_string().contains("exceeded trusted size"), "{err}");
        assert!(!dest.exists());
        assert!(!dir.join("file.download").exists());
        server.abort();
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[tokio::test]
    async fn undersized_stream_fails() {
        let dir = download_scratch("undersize");
        let dest = dir.join("file.bin");
        let (addr, server) = serve_once(plain_response(b"ab", Some(2))).await;
        let client = test_client(&DownloadLimits::default());
        let err = download_file(
            &client,
            &format!("http://{addr}/file.bin"),
            &dest,
            None,
            ABC_SHA256,
            Some(3),
        )
        .await
        .expect_err("undersized stream must fail");
        // Content-length (2) already disagrees with the trusted size (3).
        assert!(err.to_string().contains("trusted size"), "{err}");
        assert!(!dest.exists());
        server.abort();
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[tokio::test]
    async fn hash_mismatch_never_reaches_final_path() {
        let dir = download_scratch("hash-mismatch");
        let dest = dir.join("file.bin");
        // Right size (3), wrong content.
        let (addr, server) = serve_once(plain_response(b"xyz", Some(3))).await;
        let client = test_client(&DownloadLimits::default());
        let err = download_file(
            &client,
            &format!("http://{addr}/file.bin"),
            &dest,
            None,
            ABC_SHA256,
            Some(3),
        )
        .await
        .expect_err("hash mismatch must fail");
        assert!(err.to_string().contains("hash mismatch"), "{err}");
        assert!(!dest.exists(), "unverified bytes at final path");
        assert!(!dir.join("file.download").exists());
        server.abort();
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[tokio::test]
    async fn verified_download_lands_at_final_path() {
        let dir = download_scratch("verified");
        let dest = dir.join("file.bin");
        let (addr, server) = serve_once(plain_response(b"abc", Some(3))).await;
        let client = test_client(&DownloadLimits::default());
        download_file(
            &client,
            &format!("http://{addr}/file.bin"),
            &dest,
            None,
            ABC_SHA256,
            Some(3),
        )
        .await
        .expect("verified download must succeed");
        assert_eq!(std::fs::read(&dest).unwrap(), b"abc");
        assert!(!dir.join("file.download").exists());
        server.abort();
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[tokio::test]
    async fn download_timeout_cleans_temporary_file() {
        let dir = download_scratch("timeout");
        let dest = dir.join("file.bin");
        // Server accepts but never responds.
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            use tokio::io::AsyncReadExt;
            if let Ok((mut sock, _)) = listener.accept().await {
                let mut buf = [0u8; 1024];
                let _ = sock.read(&mut buf).await;
                tokio::time::sleep(Duration::from_secs(30)).await;
            }
        });
        let limits = DownloadLimits::new(5, 1);
        let client = test_client(&limits);
        let err = download_file(
            &client,
            &format!("http://{addr}/file.bin"),
            &dest,
            Some("SECRET-HF-TOKEN"),
            ABC_SHA256,
            Some(3),
        )
        .await
        .expect_err("stalled download must time out");
        assert!(!err.to_string().contains("SECRET-HF-TOKEN"), "{err}");
        assert!(!dest.exists());
        assert!(!dir.join("file.download").exists());
        server.abort();
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
