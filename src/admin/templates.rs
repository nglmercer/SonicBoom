use crate::auth::token::Token;

pub fn login_page(error: Option<&str>) -> String {
    let error_html = error
        .map(|e| format!(r#"<p class="error">{}</p>"#, html_escape(e)))
        .unwrap_or_default();

    format!(
        r#"<!DOCTYPE html>
<html lang="ko">
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
    <input type="text" name="id" placeholder="ID" required>
    <input type="password" name="pw" placeholder="Password" required>
    <button type="submit">Login</button>
  </form>
</div>
</body>
</html>"#,
        error_html
    )
}

pub fn admin_page(tokens: &[Token]) -> String {
    let rows: String = tokens
        .iter()
        .map(|t| {
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
  <td>{}</td>
  <td>{}</td>
  <td>
    {}
  </td>
</tr>"#,
                html_escape(&t.value),
                html_escape(&expires),
                status,
                if !t.revoked {
                    format!(
                        r#"<form method="post" action="/admin/tokens/{}/revoke" style="display:inline">
      <button type="submit">Revoke</button>
    </form>"#,
                        html_escape(&t.id)
                    )
                } else {
                    String::new()
                }
            )
        })
        .collect();

    format!(
        r#"<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SonicBoom Admin</title>
<style>
  body {{ font-family: sans-serif; max-width: 900px; margin: 2rem auto; padding: 0 1rem; }}
  h1 {{ display: flex; justify-content: space-between; align-items: center; }}
  a.logout {{ font-size: 0.9rem; color: #666; }}
  table {{ width: 100%; border-collapse: collapse; margin-top: 1rem; }}
  th, td {{ text-align: left; padding: 0.5rem; border-bottom: 1px solid #ddd; }}
  th {{ background: #f0f0f0; }}
  code {{ font-size: 0.8rem; word-break: break-all; }}
  .create-form {{ margin-top: 2rem; background: #f9f9f9; padding: 1rem; border-radius: 4px; }}
  .create-form h2 {{ margin-top: 0; }}
  input, select {{ padding: 0.4rem; margin-right: 0.5rem; border: 1px solid #ccc; border-radius: 4px; }}
  button {{ padding: 0.4rem 1rem; background: #333; color: white; border: none; border-radius: 4px; cursor: pointer; }}
  button[type=submit][name=action][value=revoke] {{ background: #c00; }}
</style>
</head>
<body>
<h1>Token Management <a class="logout" href="/admin/logout">Logout</a></h1>
<table>
  <thead><tr><th>Token</th><th>Expires</th><th>Status</th><th>Action</th></tr></thead>
  <tbody>{}</tbody>
</table>
<div class="create-form">
  <h2>Create New Token</h2>
  <form method="post" action="/admin/tokens">
    <label>Expires: <input type="datetime-local" name="expires_at"></label>
    <small>(leave blank for no expiry)</small>
    <br><br>
    <button type="submit">Generate Token</button>
  </form>
</div>
</body>
</html>"#,
        rows
    )
}

fn html_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}
