use crate::auth::token::Token;

pub fn login_page(error: Option<&str>) -> String {
    let error_html = error
        .map(|e| format!(r#"<p class="error">{}</p>"#, html_escape(e)))
        .unwrap_or_default();

    format!(
        r#"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SonicBoom Admin - Login</title>
<link rel="stylesheet" href="/static/admin.css">
</head>
<body class="login">
<div class="box">
  <h1>Admin Login</h1>
  {}
  <form method="post" action="/admin/login">
    <input type="text" name="id" placeholder="ID" required autocomplete="username">
    <input type="password" name="pw" placeholder="Password" required autocomplete="current-password">
    <button type="submit">Login</button>
  </form>
</div>
</body>
</html>"#,
        error_html
    )
}

/// Render the token admin page.
///
/// `tokens` exposes only safe metadata (id, fingerprint, timestamps,
/// status) — never raw bearer values. `new_token` carries a freshly created
/// raw token to display exactly once, and `csrf_token` is embedded in every
/// mutation form.
pub fn admin_page(tokens: &[Token], csrf_token: &str, new_token: Option<&str>) -> String {
    let rows: String = tokens
        .iter()
        .map(|t| {
            let created = t.created_at.format("%Y-%m-%d %H:%M UTC").to_string();
            let expires = t
                .expires_at
                .map(|e| e.format("%Y-%m-%d %H:%M UTC").to_string())
                .unwrap_or_else(|| "Never".to_string());
            let status = if t.revoked {
                "Revoked"
            } else if t.is_valid() {
                "Active"
            } else {
                "Expired"
            };
            format!(
                r#"<tr>
  <td><code>{}</code></td>
  <td><code>{}</code></td>
  <td>{}</td>
  <td>{}</td>
  <td>{}</td>
  <td>
    {}
  </td>
</tr>"#,
                html_escape(&t.id),
                html_escape(&t.fingerprint()),
                html_escape(&created),
                html_escape(&expires),
                status,
                if !t.revoked {
                    format!(
                        r#"<form class="inline-form" method="post" action="/admin/tokens/{}/revoke"
      <input type="hidden" name="csrf_token" value="{}">
      <button type="submit">Revoke</button>
    </form>"#,
                        html_escape(&t.id),
                        html_escape(csrf_token)
                    )
                } else {
                    String::new()
                }
            )
        })
        .collect();

    let new_token_html = new_token
        .map(|raw| {
            format!(
                r#"<div class="new-token">
  <h2>New Token Created</h2>
  <p><strong>Copy this token now. It cannot be displayed again.</strong></p>
  <p><code>{}</code></p>
</div>"#,
                html_escape(raw)
            )
        })
        .unwrap_or_default();

    format!(
        r#"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SonicBoom Admin</title>
<link rel="stylesheet" href="/static/admin.css">
</head>
<body class="admin">
<h1>Token Management
  <form class="logout-form" method="post" action="/admin/logout">
    <input type="hidden" name="csrf_token" value="{}">
    <button type="submit">Logout</button>
  </form>
</h1>
{}
<table>
  <thead><tr><th>ID</th><th>Fingerprint</th><th>Created</th><th>Expires</th><th>Status</th><th>Action</th></tr></thead>
  <tbody>{}</tbody>
</table>
<div class="create-form">
  <h2>Create New Token</h2>
  <form method="post" action="/admin/tokens">
    <input type="hidden" name="csrf_token" value="{}">
    <label>Expires: <input type="datetime-local" name="expires_at"></label>
    <small>(leave blank for no expiry; interpreted as UTC)</small>
    <br><br>
    <button type="submit">Generate Token</button>
  </form>
</div>
</body>
</html>"#,
        html_escape(csrf_token),
        new_token_html,
        rows,
        html_escape(csrf_token)
    )
}

/// Small error page for rejected admin form submissions (e.g. a
/// malformed token expiry). Carries no inline script or style.
pub fn error_page(message: &str) -> String {
    format!(
        r#"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SonicBoom Admin - Error</title>
<link rel="stylesheet" href="/static/admin.css">
</head>
<body class="login">
<div class="box">
  <h1>Request Rejected</h1>
  <p class="error">{}</p>
  <p><a href="/admin">Back to admin panel</a></p>
</div>
</body>
</html>"#,
        html_escape(message)
    )
}

fn html_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::auth::token::{Token, hash_token_value};

    #[test]
    fn admin_page_never_renders_raw_tokens() {
        let raw = "super-secret-bearer-value";
        let token = Token::new(hash_token_value(raw), None);
        let html = admin_page(std::slice::from_ref(&token), "csrf", None);
        assert!(!html.contains(raw), "raw token leaked into admin page");
        assert!(html.contains(token.fingerprint().trim_end_matches('…')));
    }

    #[test]
    fn admin_page_embeds_csrf_in_mutation_forms() {
        let token = Token::new(hash_token_value("x"), None);
        let html = admin_page(std::slice::from_ref(&token), "csrf-abc", None);
        assert!(html.contains(r#"name="csrf_token" value="csrf-abc""#));
        assert!(html.contains(r#"method="post" action="/admin/logout""#));
        assert!(!html.contains(r#"href="/admin/logout""#));
    }

    #[test]
    fn new_token_banner_marks_one_time_display() {
        let html = admin_page(&[], "csrf", Some("raw-token-once"));
        assert!(html.contains("raw-token-once"));
        assert!(html.contains("Copy this token now. It cannot be displayed again."));
    }

    #[test]
    fn error_page_escapes_message_and_links_back() {
        let html = error_page("<script>alert(1)</script>");
        assert!(!html.contains("<script>"), "message became markup: {html}");
        assert!(html.contains("&lt;script&gt;"));
        assert!(html.contains(r#"href="/admin""#));
    }

    #[test]
    fn admin_pages_have_no_inline_script_or_style() {
        let token = Token::new(hash_token_value("x"), None);
        for html in [
            login_page(None),
            admin_page(std::slice::from_ref(&token), "csrf", None),
            error_page("bad input"),
        ] {
            assert!(!html.contains("<script"), "inline script found");
            assert!(!html.contains("<style"), "inline style found");
            assert!(!html.contains("style="), "inline style attribute found");
            assert!(html.contains(r#"href="/static/admin.css""#));
        }
    }
}
