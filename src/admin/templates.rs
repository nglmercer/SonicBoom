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
<style>
  body {{ font-family: sans-serif; display: flex; justify-content: center; align-items: center; min-height: 100vh; margin: 0; background: #f5f5f5; }}
  .box {{ background: white; padding: 2rem; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); width: 300px; }}
  h1 {{ margin-top: 0; font-size: 1.4rem; }}
  input {{ width: 100%; padding: 0.5rem; margin-bottom: 1rem; box-sizing: border-box; border: 1px solid #ccc; border-radius: 4px; }}
  button {{ width: 100%; padding: 0.6rem; background: #333; color: white; border: none; border-radius: 4px; cursor: pointer; }}
  .error {{ color: red; margin-bottom: 1rem; }}
</style>
</head>
<body>
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
                        r#"<form method="post" action="/admin/tokens/{}/revoke" style="display:inline">
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
<style>
  body {{ font-family: sans-serif; max-width: 900px; margin: 2rem auto; padding: 0 1rem; }}
  h1 {{ display: flex; justify-content: space-between; align-items: center; }}
  .logout-form {{ display: inline; }}
  .logout-form button {{ background: none; color: #666; font-size: 0.9rem; text-decoration: underline; padding: 0; cursor: pointer; border: none; }}
  table {{ width: 100%; border-collapse: collapse; margin-top: 1rem; }}
  th, td {{ text-align: left; padding: 0.5rem; border-bottom: 1px solid #ddd; }}
  th {{ background: #f0f0f0; }}
  code {{ font-size: 0.8rem; word-break: break-all; }}
  .create-form {{ margin-top: 2rem; background: #f9f9f9; padding: 1rem; border-radius: 4px; }}
  .create-form h2 {{ margin-top: 0; }}
  .new-token {{ margin-top: 1rem; background: #e8f5e9; padding: 1rem; border-radius: 4px; border: 1px solid #a5d6a7; }}
  input, select {{ padding: 0.4rem; margin-right: 0.5rem; border: 1px solid #ccc; border-radius: 4px; }}
  button {{ padding: 0.4rem 1rem; background: #333; color: white; border: none; border-radius: 4px; cursor: pointer; }}
  button[type=submit][name=action][value=revoke] {{ background: #c00; }}
</style>
</head>
<body>
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
    <small>(leave blank for no expiry)</small>
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
}
