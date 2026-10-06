// Backend state is exposed via data attributes (CSP has no unsafe-inline).
const MODEL_READY = document.body.dataset.modelReady === 'true';

// DOM Cache
const tokenGroup = document.getElementById('tokenGroup');
const apiTokenInput = document.getElementById('apiToken');
const textInput = document.getElementById('text');
const voiceSelect = document.getElementById('voice');
const langSelect = document.getElementById('lang');
const generateBtn = document.getElementById('btn');
const statusDiv = document.getElementById('status');
const audioPlayer = document.getElementById('player');

// Server info discovered from GET /api/info. When the
// server runs in local (loopback) mode no API token is
// required, so the token field is hidden (spec §32).
let serverInfo = null;

// Store active audio object URL to prevent memory leaks
let currentAudioUrl = null;

// Centralized API client: attaches the bearer token only
// when the server actually requires one (spec §32).
async function apiFetch(path, options = {}) {
  const headers = new Headers(options.headers || {});
  const apiToken = apiTokenInput.value.trim();
  if (serverInfo && serverInfo.auth && serverInfo.auth.required && apiToken) {
    headers.set('Authorization', `Bearer ${apiToken}`);
  }
  return fetch(path, { ...options, headers });
}

async function loadServerInfo() {
  try {
    const resp = await apiFetch('/api/info');
    if (!resp.ok) {
      throw new Error(`HTTP ${resp.status}`);
    }
    serverInfo = await resp.json();
  } catch (e) {
    console.warn('Could not load server info:', e);
    serverInfo = null;
  }
  if (serverInfo && serverInfo.auth && !serverInfo.auth.required) {
    tokenGroup.hidden = true;
  } else {
    tokenGroup.hidden = false;
  }
}

// Initialize Event Listeners
loadServerInfo().then(refreshConfiguration);
generateBtn.addEventListener('click', synthesize);

async function synthesize() {
  const text = textInput.value.trim();
  if (!text) {
    setStatus('Please enter some text.', true);
    return;
  }

  if (serverInfo && serverInfo.auth && serverInfo.auth.required) {
    const apiToken = apiTokenInput.value.trim();
    if (!apiToken) {
      setStatus('Please enter your API token.', true);
      return;
    }
  }

  const voice = voiceSelect.value;
  const lang = langSelect.value;

  // Set UI Loading State
  generateBtn.disabled = true;
  generateBtn.classList.add('loading');
  generateBtn.textContent = 'Generating Speech...';
  setStatus('');

  try {
    const params = new URLSearchParams({ voice, lang });
    const resp = await apiFetch(`/api/tts?${params.toString()}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'text/plain',
      },
      body: text,
    });

    if (!resp.ok) {
      const msg = await resp.text().catch(() => resp.statusText);
      throw new Error(`HTTP ${resp.status}: ${msg}`);
    }

    const blob = await resp.blob();

    // Revoke previous audio session URL to free up browser memory
    if (currentAudioUrl) {
      URL.revokeObjectURL(currentAudioUrl);
    }

    currentAudioUrl = URL.createObjectURL(blob);
    audioPlayer.src = currentAudioUrl;
    audioPlayer.style.display = 'block';

    // Play naturally, catching browser auto-play prevention errors
    audioPlayer.play().catch(err => {
      console.log("Audio waiting for user interaction to play: ", err);
    });

    setStatus('');
  } catch (e) {
    setStatus(e.message, true);
  } finally {
    // Reset UI State
    generateBtn.disabled = false;
    generateBtn.classList.remove('loading');
    generateBtn.textContent = 'Generate Speech';
  }
}

function setStatus(msg, isError) {
  statusDiv.textContent = msg;
  statusDiv.className = isError ? 'error' : '';
}

// Refresh periodically if model is not ready yet
if (!MODEL_READY) {
  setTimeout(() => location.reload(), 5000);
}


let configSnapshot = null;
let configSchema = {};
const configSetting = document.getElementById('configSetting');
const configValue = document.getElementById('configValue');
const configError = document.getElementById('configError');

async function configRequest(path, options) {
  const response = await apiFetch(path, options);
  const body = await response.json();
  if (!response.ok) throw new Error(body.message || body.error || `HTTP ${response.status}`);
  return body;
}

function selectedSetting() {
  const path = configSetting.value;
  const metadata = configSchema[path];
  if (!metadata || !configSnapshot) return;
  const value = path.split('.').reduce((object, key) => object?.[key], configSnapshot.config);
  configValue.value = Array.isArray(value) ? JSON.stringify(value) : value ?? '';
  configValue.type = ['integer', 'number'].includes(metadata.type) ? 'number' : 'text';
  configValue.step = metadata.type === 'integer' ? '1' : 'any';
  for (const limit of ['min', 'max']) {
    if (metadata[limit] !== undefined) configValue.setAttribute(limit, metadata[limit]);
    else configValue.removeAttribute(limit);
  }
  document.getElementById('configDescription').textContent = metadata.description;
  document.getElementById('configApply').textContent = `Apply: ${metadata.apply}`;
  document.getElementById('configOptions')?.remove();
  const options = document.createElement('datalist');
  options.id = 'configOptions';
  for (const value of metadata.allowed_values || (metadata.type === 'boolean' ? ['true', 'false'] : [])) {
    const option = document.createElement('option'); option.value = value; options.append(option);
  }
  configValue.after(options); configValue.setAttribute('list', options.id);
  if (metadata.allowed_values_source === 'audio_devices') {
    configRequest('/api/audio/devices').then(list => {
      for (const device of list.devices || []) {
        const option = document.createElement('option'); option.value = device.id; options.append(option);
      }
    }).catch(error => { configError.textContent = error.message; });
  }
}

async function refreshConfiguration() {
  try {
    const status = await configRequest('/api/config/status');
    document.getElementById('configPath').textContent = status.config_path;
    document.getElementById('configOpen').hidden = serverInfo?.mode !== 'desktop';
    document.getElementById('configStatus').textContent =
      `Revision ${status.revision} · Watcher ${status.watcher} · Last reload ${status.last_reload || '—'} · Pending: ${(status.pending_applies || []).join(', ') || 'none'}`;
    configError.textContent = [status.last_error, ...status.apply_failures.map(f => `${f.path}: ${f.error} (active: ${JSON.stringify(f.active)})`)].filter(Boolean).join('\n');
    if (!configSnapshot || configSnapshot.revision !== status.revision) {
      const previous = configSnapshot;
      const snapshot = await configRequest('/api/config');
      configSnapshot = snapshot;
      if (Object.keys(configSchema).length === 0) {
        configSchema = await configRequest('/api/config/schema');
        for (const [path, setting] of Object.entries(configSchema)) {
          if (setting.secret) continue;
          const option = document.createElement('option'); option.value = path; option.textContent = path; configSetting.append(option);
        }
      }
      if (previous) document.getElementById('configStatus').textContent += ' · Configuration changed';
      if (document.activeElement !== configValue) selectedSetting();
    }
  } catch (error) { configError.textContent = error.message; }
}
configSetting.addEventListener('change', selectedSetting);
apiTokenInput.addEventListener('change', refreshConfiguration);
document.getElementById('configRefresh').addEventListener('click', refreshConfiguration);
for (const [id, action] of [['configReload', 'reload'], ['configValidate', 'validate']]) {
  document.getElementById(id).addEventListener('click', async () => {
    try {
      await configRequest(`/api/config/${action}`, { method: 'POST' });
      if (action === 'validate') configError.textContent = 'Configuration file is valid.';
      else await refreshConfiguration();
    } catch (error) { configError.textContent = error.message; }
  });
}
document.getElementById('configForm').addEventListener('submit', async event => {
  event.preventDefault();
  if (!configSnapshot) return;
  try {
    const path = configSetting.value;
    const type = configSchema[path].type;
    let value = configValue.value;
    if (configSchema[path].nullable && (value === '' || value === 'null')) value = null;
    else if (['integer', 'number'].includes(type)) { value = Number(value); if (!Number.isFinite(value)) throw new Error('Expected a number'); }
    else if (['array', 'boolean'].includes(type)) value = JSON.parse(value);
    const changes = {};
    const parts = path.split('.');
    let target = changes;
    for (const part of parts.slice(0, -1)) target = target[part] = {};
    target[parts.at(-1)] = value;
    await configRequest('/api/config', { method: 'PATCH', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ expected_revision: configSnapshot.revision, changes }) });
    await refreshConfiguration();
  } catch (error) { configError.textContent = error.message; }
});
setInterval(async () => {
  await loadServerInfo();
  if (document.getElementById('configuration').open) await refreshConfiguration();
}, 3000);

document.getElementById('configOpen').addEventListener('click', async () => {
  try { await configRequest('/api/config/open', { method: 'POST' }); }
  catch (error) { configError.textContent = error.message; }
});
