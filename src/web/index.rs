use axum::{
    extract::State,
    http::{StatusCode, header},
    response::{IntoResponse, Response},
};

use crate::{AppState, tts::ModelStatus};

// Load the HTML template at compile time
const TEMPLATE: &str = include_str!("../../templates/index.html");

/// Liveness: the process is alive. Always 200 when the server can respond.
/// Used by container `HEALTHCHECK` / load-balancer liveness probes.
pub async fn get_health() -> impl IntoResponse {
    (StatusCode::OK, "OK")
}

/// Readiness: the model is loaded and the server can synthesize.
/// Used by load-balancer readiness probes and orchestrators.
pub async fn get_ready(State(state): State<AppState>) -> impl IntoResponse {
    let status = state.model_status.read().await;
    match &*status {
        ModelStatus::Ready(_) => (StatusCode::OK, "ready"),
        ModelStatus::Downloading { .. } => (StatusCode::SERVICE_UNAVAILABLE, "model downloading"),
        ModelStatus::Loading => (StatusCode::SERVICE_UNAVAILABLE, "model loading"),
        ModelStatus::Idle => (StatusCode::SERVICE_UNAVAILABLE, "model idle"),
        ModelStatus::Failed(_) => (StatusCode::SERVICE_UNAVAILABLE, "model failed to load"),
    }
}

pub async fn get_index(State(state): State<AppState>) -> Response {
    // Collect available voices
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
            ModelStatus::Idle => "Preparing model...".to_string(),
            ModelStatus::Downloading { progress } => {
                format!("Downloading model... ({:.0}%)", progress * 100.0)
            }
            ModelStatus::Loading => "Loading model...".to_string(),
            ModelStatus::Ready(_) => String::new(),
            // Never render internal load errors into the page.
            ModelStatus::Failed(_) => "Model load failed. Check server logs.".to_string(),
        }
    };

    let model_ready = !voices.is_empty();

    let voice_options: String = voice_options_html(&voices);

    let html = TEMPLATE
        .replace("STATUS_MSG", &status_msg)
        .replace(
            "STATUS_HIDDEN",
            if status_msg.is_empty() { "hidden" } else { "" },
        )
        .replace(
            "VOICE_OPTIONS",
            if voice_options.is_empty() {
                "<option value=\"\">No voices available</option>"
            } else {
                &voice_options
            },
        )
        .replace("VOICE_DISABLED", if !model_ready { "disabled" } else { "" })
        .replace("BTN_DISABLED", if !model_ready { "disabled" } else { "" })
        .replace("MODEL_READY_JS", if model_ready { "true" } else { "false" });

    (
        StatusCode::OK,
        [(header::CONTENT_TYPE, "text/html; charset=utf-8")],
        html,
    )
        .into_response()
}

/// Render voice names as `<option>` elements. Model metadata is still
/// data: names are escaped for both the attribute and text contexts so a
/// hostile or surprising voice name can never break out of the markup.
fn voice_options_html(voices: &[String]) -> String {
    voices
        .iter()
        .map(|v| {
            let escaped = html_escape(v);
            format!(r#"<option value="{escaped}">{escaped}</option>"#)
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn html_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&#39;")
}

#[cfg(test)]
mod tests {
    use super::{TEMPLATE, voice_options_html};

    #[test]
    fn index_template_has_no_inline_script_or_style() {
        assert!(!TEMPLATE.contains("<script>"), "inline script found");
        assert!(!TEMPLATE.contains("<style>"), "inline style found");
        assert!(!TEMPLATE.contains("style="), "inline style attribute found");
        assert!(TEMPLATE.contains(r#"<script src="/static/index.js" defer></script>"#));
        assert!(TEMPLATE.contains(r#"<link rel="stylesheet" href="/static/index.css">"#));
        assert!(TEMPLATE.contains("data-model-ready="));
    }

    #[test]
    fn hostile_voice_names_are_escaped() {
        let voices = vec![
            "\"><script>alert(1)</script>".to_string(),
            "A&B".to_string(),
            "a<b>c".to_string(),
            "quote\"test".to_string(),
            "apostrophe'test".to_string(),
            "M1".to_string(),
        ];
        let html = voice_options_html(&voices);
        assert!(
            !html.contains("<script>"),
            "voice markup became executable: {html}"
        );
        assert!(!html.contains("\"><script>"), " breakout: {html}");
        assert!(
            html.contains("&quot;&gt;&lt;script&gt;"),
            "missing escape: {html}"
        );
        assert!(html.contains("A&amp;B"), "missing escape: {html}");
        assert!(html.contains("a&lt;b&gt;c"), "missing escape: {html}");
        assert!(html.contains("quote&quot;test"), "missing escape: {html}");
        assert!(
            html.contains("apostrophe&#39;test"),
            "missing escape: {html}"
        );
        assert!(html.contains(r#"<option value="M1">M1</option>"#));
    }
}
