// Backend state is exposed via data attributes (CSP has no unsafe-inline).
const MODEL_READY = document.body.dataset.modelReady === 'true';

// DOM Cache
const apiTokenInput = document.getElementById('apiToken');
const textInput = document.getElementById('text');
const voiceSelect = document.getElementById('voice');
const langSelect = document.getElementById('lang');
const generateBtn = document.getElementById('btn');
const statusDiv = document.getElementById('status');
const audioPlayer = document.getElementById('player');

// Store active audio object URL to prevent memory leaks
let currentAudioUrl = null;

// Initialize Event Listeners
generateBtn.addEventListener('click', synthesize);

async function synthesize() {
  const text = textInput.value.trim();
  if (!text) {
    setStatus('Please enter some text.', true);
    return;
  }

  const apiToken = apiTokenInput.value.trim();
  if (!apiToken) {
    setStatus('Please enter your API token.', true);
    return;
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
    const resp = await fetch(`/api/tts?${params.toString()}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'text/plain',
        'Authorization': `Bearer ${apiToken}`,
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
