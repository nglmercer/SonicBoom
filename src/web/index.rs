use axum::{
    extract::State,
    http::{header, StatusCode},
    response::{IntoResponse, Response},
};

use crate::{tts::ModelStatus, AppState};

pub async fn get_index(State(state): State<AppState>) -> Response {
    // 음성 목록 수집
    let voices: Vec<String> = {
        let status = state.model_status.read().await;
        match &*status {
            ModelStatus::Ready(handle) => {
                let mut names: Vec<String> = handle.voice_styles.keys().cloned().collect();
                names.sort();
                names
            }
            _ => vec![],
        }
    };

    let status_msg = {
        let status = state.model_status.read().await;
        match &*status {
            ModelStatus::Idle => "모델 준비 중...".to_string(),
            ModelStatus::Downloading { progress } => {
                format!("모델 다운로드 중... ({:.0}%)", progress * 100.0)
            }
            ModelStatus::Loading => "모델 로딩 중...".to_string(),
            ModelStatus::Ready(_) => String::new(),
            ModelStatus::Failed(e) => format!("모델 로드 실패: {e}"),
        }
    };

    let model_ready = !voices.is_empty();

    let voice_options: String = voices
        .iter()
        .map(|v| format!(r#"<option value="{v}">{v}</option>"#))
        .collect::<Vec<_>>()
        .join("\n");

    let html = format!(
        r#"<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SonicBoom TTS</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    background: #f5f5f7;
    min-height: 100vh;
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 1rem;
  }}
  .card {{
    background: #fff;
    border-radius: 16px;
    box-shadow: 0 4px 24px rgba(0,0,0,0.08);
    padding: 2rem;
    width: 100%;
    max-width: 560px;
  }}
  h1 {{ font-size: 1.5rem; margin-bottom: 1.5rem; color: #1d1d1f; }}
  label {{ display: block; font-size: 0.85rem; color: #6e6e73; margin-bottom: 0.3rem; }}
  textarea, select {{
    width: 100%;
    padding: 0.65rem 0.8rem;
    border: 1px solid #d2d2d7;
    border-radius: 8px;
    font-size: 1rem;
    background: #fafafa;
    outline: none;
    transition: border-color 0.15s;
  }}
  textarea:focus, select:focus {{ border-color: #0071e3; background: #fff; }}
  textarea {{ resize: vertical; min-height: 120px; font-family: inherit; }}
  .row {{ display: flex; gap: 1rem; margin-top: 1rem; }}
  .row .field {{ flex: 1; }}
  button {{
    margin-top: 1.25rem;
    width: 100%;
    padding: 0.75rem;
    background: #0071e3;
    color: #fff;
    border: none;
    border-radius: 8px;
    font-size: 1rem;
    cursor: pointer;
    transition: background 0.15s;
  }}
  button:hover:not(:disabled) {{ background: #0077ed; }}
  button:disabled {{ background: #b0b8c1; cursor: not-allowed; }}
  #status {{
    margin-top: 1rem;
    font-size: 0.875rem;
    color: #6e6e73;
    min-height: 1.2em;
    text-align: center;
  }}
  #status.error {{ color: #d70015; }}
  audio {{ margin-top: 1rem; width: 100%; display: none; }}
  .model-status {{
    margin-bottom: 1rem;
    padding: 0.6rem 0.8rem;
    border-radius: 8px;
    background: #fff3cd;
    color: #856404;
    font-size: 0.875rem;
    display: {model_status_display};
  }}
</style>
</head>
<body>
<div class="card">
  <h1>SonicBoom TTS</h1>
  <div class="model-status" id="modelStatus">{status_msg}</div>

  <label for="text">텍스트</label>
  <textarea id="text" placeholder="음성으로 변환할 텍스트를 입력하세요..."></textarea>

  <div class="row">
    <div class="field">
      <label for="voice">음성 모델</label>
      <select id="voice" {voice_disabled}>
        {voice_options_html}
      </select>
    </div>
    <div class="field">
      <label for="lang">언어</label>
      <select id="lang" {voice_disabled}>
        <option value="en">English</option>
        <option value="ko">한국어</option>
        <option value="es">Español</option>
        <option value="pt">Português</option>
        <option value="fr">Français</option>
      </select>
    </div>
  </div>

  <button id="btn" onclick="synthesize()" {btn_disabled}>음성 생성</button>
  <div id="status"></div>
  <audio id="player" controls></audio>
  <p style="margin-top:1.5rem;font-size:0.75rem;color:#aeaeb2;text-align:center;">
    <a href="https://huggingface.co/Supertone/supertonic-2" target="_blank" rel="noopener"
       style="color:inherit;">Supertonic 2</a> 모델을 활용해 음성을 생성하고 있습니다
  </p>
</div>

<script>
const MODEL_READY = {model_ready_js};

async function synthesize() {{
  const text = document.getElementById('text').value.trim();
  if (!text) {{
    setStatus('텍스트를 입력해 주세요.', true);
    return;
  }}

  const voice = document.getElementById('voice').value;
  const lang = document.getElementById('lang').value;
  const btn = document.getElementById('btn');
  btn.disabled = true;
  setStatus('생성 중...');

  try {{
    const params = new URLSearchParams({{ voice, lang }});
    const resp = await fetch('/api/tts?' + params.toString(), {{
      method: 'POST',
      headers: {{ 'Content-Type': 'text/plain' }},
      body: text,
    }});

    if (!resp.ok) {{
      const msg = await resp.text().catch(() => resp.statusText);
      throw new Error(`HTTP ${{resp.status}}: ${{msg}}`);
    }}

    const blob = await resp.blob();
    const url = URL.createObjectURL(blob);
    const player = document.getElementById('player');
    player.src = url;
    player.style.display = 'block';
    player.play();
    setStatus('');
  }} catch (e) {{
    setStatus(e.message, true);
  }} finally {{
    btn.disabled = false;
  }}
}}

function setStatus(msg, isError) {{
  const el = document.getElementById('status');
  el.textContent = msg;
  el.className = isError ? 'error' : '';
}}

// 모델이 아직 준비 안 됐으면 주기적으로 새로고침
if (!MODEL_READY) {{
  setTimeout(() => location.reload(), 5000);
}}
</script>
</body>
</html>"#,
        model_status_display = if status_msg.is_empty() { "none" } else { "block" },
        voice_options_html = if voice_options.is_empty() {
            r#"<option value="">사용 가능한 음성 없음</option>"#.to_string()
        } else {
            voice_options
        },
        voice_disabled = if !model_ready { "disabled" } else { "" },
        btn_disabled = if !model_ready { "disabled" } else { "" },
        model_ready_js = if model_ready { "true" } else { "false" },
    );

    (StatusCode::OK, [(header::CONTENT_TYPE, "text/html; charset=utf-8")], html).into_response()
}
