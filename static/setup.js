
const steps = ["1", "1b", "2", "3", "4"];
let index = 0;
const form = document.getElementById("wizard");
const error = document.getElementById("error");

function show(i) {
  index = i;
  document.querySelectorAll(".step").forEach(el =>
    el.classList.toggle("active", el.dataset.step === steps[i]));
  const visibleSteps = usageMode() === "advanced" ? steps : steps.filter(step => step !== "1b");
  document.getElementById("step-no").textContent = visibleSteps.indexOf(steps[i]) + 1;
  document.getElementById("step-count").textContent = visibleSteps.length;
  document.getElementById("back").style.visibility =
    i === 0 ? "hidden" : "visible";
  const last = i === steps.length - 1;
  document.getElementById("next").hidden = last;
  document.getElementById("finish").hidden = !last;
  if (last) renderSummary();
}

function usageMode() {
  return form.querySelector('input[name="usage_mode"]:checked').value;
}

function renderSummary() {
  const mode = usageMode();
  let bind = "127.0.0.1", port = 17842, auth = "local";
  if (mode === "lan") { bind = "0.0.0.0"; auth = "token"; }
  if (mode === "advanced") {
    bind = form.bind.value || "127.0.0.1";
    port = form.port.value || 17842;
    auth = form.auth_mode.value;
  }
  document.getElementById("summary").textContent =
    "API: http://" + bind + ":" + port + "\n" +
    "Authentication: " + auth + "\n" +
    "Audio: " + (form.audio_device.value || "default") + "\n" +
    "Inference steps: " + (form.inference_steps.value || 5);
}

document.getElementById("next").onclick = () => {
  if (usageMode() === "advanced" && steps[index] === "1") { show(1); return; }
  if (steps[index] === "1b") { show(2); return; }
  show(steps[index] === "1" ? 2 : Math.min(index + 1, steps.length - 1));
};
document.getElementById("back").onclick = () => {
  if (steps[index] === "1b") { show(0); return; }
  show(steps[index] === "2" && usageMode() !== "advanced" ? 0 : Math.max(index - 1, 0));
};
form.querySelectorAll('input[name="usage_mode"]').forEach(r =>
  r.addEventListener("change", () => { if (index === 0) renderSummary(); }));
["bind", "port", "auth_mode", "audio_device", "inference_steps"].forEach(name => {
  const el = form.elements[name];
  if (el) el.addEventListener("change", renderSummary);
});

document.getElementById("test-audio").onclick = async () => {
  try {
    const device = form.audio_device.value;
    if (!device) throw new Error("Choose an audio output first.");
    const commit = await apiFetch("/api/config", { method: "PATCH", headers: { "content-type": "application/json" },
      body: JSON.stringify({ changes: { audio: { output_device: device } } }) });
    if (!commit.ok) throw new Error("Could not configure selected output.");
    for (let attempt = 0; attempt < 20; attempt++) {
      const output = await apiFetch("/api/audio/output");
      const active = await output.json();
      if (active.device === device && active.available) break;
      if (attempt === 19) throw new Error("Selected output is unavailable. Check configuration status.");
      await new Promise(resolve => setTimeout(resolve, 150));
    }
    const res = await apiFetch("/api/tts/play", { method: "POST",
      headers: { "content-type": "text/plain" }, body: "This is a test of the SonicBoom audio setup." });
    if (res.ok) { error.textContent = "Test audio played (check your output device)."; }
    else { error.textContent = "Test failed: HTTP " + res.status; }
  } catch (e) { error.textContent = "Test failed: " + e; }
};

async function apiFetch(path, options) { return fetch(path, options); }

form.onsubmit = async (event) => {
  event.preventDefault();
  error.textContent = "";
  const data = {
    usage_mode: usageMode(),
    audio_device: form.audio_device.value || "default",
    inference_steps: Number(form.inference_steps.value || 5),
    model_cache_dir: form.model_cache_dir.value || null,
    bind: form.bind.value || null,
    port: form.port.value ? Number(form.port.value) : null,
    auth_mode: form.auth_mode.value || null,
    admin_username: form.admin_username.value || "admin",
    admin_password: form.admin_password.value,
  };
  try {
    const res = await apiFetch("/setup/complete", { method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(data) });
    const body = await res.json();
    if (!res.ok) { error.textContent = body.message || "Setup failed"; return; }
    let msg = "SonicBoom is ready.\n\nAPI: " + body.api_url +
      "\nAudio: " + body.audio_device + "\nAuthentication: " + body.auth;
    if (body.api_token) {
      msg += "\n\nAPI token (shown once — store it safely):\n" + body.api_token;
    }
    alert(msg);
    const api = new URL(body.api_url);
    window.location.href = `${window.location.protocol}//${window.location.hostname}:${api.port}/`;
  } catch (e) { error.textContent = "Setup failed: " + e; }
};

(async () => {
  try {
    const statusResponse = await apiFetch("/setup/status");
    const status = await statusResponse.json();
    form.model_cache_dir.value = status.model.cache_dir;
    form.port.value = status.server.port;
    const res = await apiFetch("/setup/audio-devices");
    if (res.ok) {
      const list = await res.json();
      const select = document.getElementById("audio-device");
      const devices = list.devices || [];
      document.getElementById("test-audio").disabled = false;
      devices.forEach(d => {
        const option = document.createElement("option");
        option.value = d.id;
        option.textContent = d.name;
        select.appendChild(option);
      });
    }
  } catch (e) { /* device enumeration is optional */ }
})();

show(0);
