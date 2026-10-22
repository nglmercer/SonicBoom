<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref, shallowRef, watch } from 'vue';
import WaveSurfer from 'wavesurfer.js';
import { t } from '../../i18n';
import { cspNonce } from '../../app/csp';

const props = withDefaults(defineProps<{ blob: Blob | null; label: string; compact?: boolean; autoplay?: boolean }>(), { compact: false, autoplay: false });
const emit = defineEmits<{ play: []; pause: []; ended: []; ready: [duration: number]; error: [message: string] }>();

const host = ref<HTMLElement | null>(null);
const instance = shallowRef<WaveSurfer | null>(null);
const playing = ref(false);
const position = ref(0);
const total = ref(0);
const problem = ref('');

/** The waveform needs a real browser: a canvas to draw on and Web Audio to
 * decode the recording. Test DOMs have neither, so the player stays inert. */
function supported(): boolean {
  return typeof window !== 'undefined' && typeof window.AudioContext === 'function';
}

function theme(name: string, fallback: string): string {
  if (typeof window === 'undefined') return fallback;
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim() || fallback;
}

function format(seconds: number): string {
  const value = Math.max(0, Math.floor(seconds));
  return `${Math.floor(value / 60)}:${String(value % 60).padStart(2, '0')}`;
}

function destroy() {
  instance.value?.destroy();
  instance.value = null;
  playing.value = false;
  position.value = 0;
  total.value = 0;
  problem.value = '';
}

function load() {
  destroy();
  const container = host.value;
  const blob = props.blob;
  if (!container || !blob || !supported()) return;
  const wavesurfer = WaveSurfer.create({
    container,
    height: props.compact ? 40 : 76,
    ...colors(),
    cursorWidth: 2,
    barWidth: 2,
    barGap: 1,
    barRadius: 2,
    normalize: true,
    hideScrollbar: true,
    dragToSeek: true,
    autoScroll: false,
    autoCenter: false,
    // The shadow root ships one inline stylesheet: the page nonce unlocks it.
    cspNonce: cspNonce() || undefined,
  });
  instance.value = wavesurfer;
  wavesurfer.on('ready', duration => { total.value = duration; emit('ready', duration); if (props.autoplay) void wavesurfer.play().catch(() => {}); });
  wavesurfer.on('play', () => { playing.value = true; emit('play'); });
  wavesurfer.on('pause', () => { playing.value = false; emit('pause'); });
  wavesurfer.on('finish', () => { playing.value = false; emit('ended'); });
  wavesurfer.on('timeupdate', time => { position.value = time; });
  wavesurfer.on('error', error => { problem.value = error.message; emit('error', error.message); });
  void wavesurfer.loadBlob(blob).catch((error: Error) => { problem.value = error.message; emit('error', error.message); });
}

function colors() {
  const accent = theme('--accent', '#076c77');
  return { waveColor: theme('--border', '#8193a0'), progressColor: accent, cursorColor: accent };
}

function toggle() { if (playing.value) instance.value?.pause(); else void instance.value?.play().catch(() => {}); }

function seek(seconds: number) {
  const player = instance.value;
  if (!player) return;
  player.setTime(Math.min(Math.max(0, seconds), total.value || seconds));
  position.value = player.getCurrentTime();
}

function onKeydown(event: KeyboardEvent) {
  const step = event.shiftKey ? 10 : 5;
  if (event.key === 'ArrowRight' || event.key === 'ArrowUp') seek(position.value + step);
  else if (event.key === 'ArrowLeft' || event.key === 'ArrowDown') seek(position.value - step);
  else if (event.key === 'Home') seek(0);
  else if (event.key === 'End') seek(total.value);
  else if (event.key === ' ' || event.key === 'Enter') toggle();
  else return;
  event.preventDefault();
}

const sliderLabel = computed(() => `${format(position.value)} / ${format(total.value)}`);

// The waveform follows the light/dark theme of the page.
let themeWatcher: MutationObserver | null = null;
onMounted(() => {
  load();
  if (typeof document === 'undefined' || typeof MutationObserver === 'undefined') return;
  themeWatcher = new MutationObserver(() => instance.value?.setOptions(colors()));
  themeWatcher.observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });
});
watch(() => props.blob, load);
watch(() => props.compact, load);
onBeforeUnmount(() => { themeWatcher?.disconnect(); themeWatcher = null; destroy(); });
defineExpose({ play: () => instance.value?.play(), pause: () => instance.value?.pause(), toggle, seekTo: seek, getDuration: () => total.value });
</script>
<template>
  <div class="waveform-player" :class="{ compact }">
    <div ref="host" class="waveform-canvas" role="slider" tabindex="0" :aria-label="label" :aria-valuemin="0" :aria-valuemax="Math.round(total)" :aria-valuenow="Math.round(position)" :aria-valuetext="sliderLabel" @keydown="onKeydown"/>
    <div class="player-bar">
      <button type="button" class="player-toggle" :aria-label="playing ? t('Pause') : t('Play')" :aria-pressed="playing" @click="toggle">
        <svg v-if="playing" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><rect x="7" y="5" width="3.5" height="14" rx="1"/><rect x="13.5" y="5" width="3.5" height="14" rx="1"/></svg>
        <svg v-else viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><path d="M8 5.5v13a1 1 0 0 0 1.5.87l11-6.5a1 1 0 0 0 0-1.74l-11-6.5A1 1 0 0 0 8 5.5Z"/></svg>
      </button>
      <p class="player-time">{{ sliderLabel }}</p>
    </div>
    <p v-if="problem" class="field-error" role="alert">{{ problem }}</p>
  </div>
</template>
