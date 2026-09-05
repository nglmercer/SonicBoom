use anyhow::Result;
use std::path::Path;
use tokio::io::AsyncWriteExt;

const MODEL_REPO: &str = "Supertone/supertonic-3";

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

pub async fn download_models<F>(
    cache_dir: &Path,
    hf_token: Option<&str>,
    on_progress: F,
) -> Result<ModelPaths>
where
    F: Fn(f32) + Send + Sync + 'static,
{
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

        // Skip only non-empty regular files. A zero-byte or partial file is
        // removed and downloaded atomically below.
        if local_path.is_file()
            && std::fs::metadata(&local_path)
                .map(|metadata| metadata.len() > 0)
                .unwrap_or(false)
        {
            tracing::info!("Already cached: {filename}");
            paths.insert(filename.to_string(), local_path);
            downloaded += 1;
            on_progress(downloaded as f32 / total as f32);
            continue;
        }
        let _ = tokio::fs::remove_file(&local_path).await;

        let url = format!("https://huggingface.co/{MODEL_REPO}/resolve/main/{filename}");
        tracing::info!("Downloading {filename}...");

        const MAX_RETRIES: u32 = 5;
        let mut last_err = None;
        let mut success = false;
        for attempt in 0..MAX_RETRIES {
            match download_file(&client, &url, &local_path, hf_token).await {
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
