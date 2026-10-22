<script setup lang="ts">
import { computed, nextTick, onMounted, onBeforeUnmount, provide, ref, watch, defineAsyncComponent } from 'vue';
import { t } from '../i18n';
import type { HomeData } from '../types';
import { createStudio, studioKey } from '../composables/useStudio';
import StudioLayout, { type StudioTab } from '../components/StudioLayout.vue';
import SpeechGenerator from '../components/studio/SpeechGenerator.vue';
const VoiceLibrary = defineAsyncComponent(() => import('../components/studio/VoiceLibrary.vue'));
const EnginesPanel = defineAsyncComponent(() => import('../components/studio/EnginesPanel.vue'));
const SettingsPanel = defineAsyncComponent(() => import('../components/studio/SettingsPanel.vue'));
import FeedbackNotice from '../components/FeedbackNotice.vue';
import { createApi } from '../services/api';
import { createHttpClient } from '../services/http';
import { errorMessage, renderMessage } from '../i18n';
const tokenStorageKey = 'sonicboom.apiToken';
const tokenDraft = ref('');
const rememberToken = ref(false);
const connecting = ref(false);
const connectionError = ref('');
const connected = ref(false);
let connectionAttempt = 0;
let connectionController: AbortController | null = null;
async function connect() {
  const candidate = tokenDraft.value.trim().replace(/^Bearer\s+/i, '').trim();
  if (!candidate) { connectionError.value = t('Please enter your API token.'); return; }
  const attempt = ++connectionAttempt;
  connectionController?.abort();
  connectionController = new AbortController();
  connecting.value = true; connectionError.value = '';
  try {
    await createApi(createHttpClient(() => candidate)).voices(connectionController.signal);
    if (attempt !== connectionAttempt) return;
    token.value = candidate; connected.value = true; tokenDraft.value = '';
    try {
      if (rememberToken.value) sessionStorage.setItem(tokenStorageKey, candidate);
      else sessionStorage.removeItem(tokenStorageKey);
    } catch { /* Storage may be disabled; the active connection still works. */ }
  } catch (error) {
    if (attempt === connectionAttempt) { connectionError.value = renderMessage(errorMessage(error)); }
  } finally { if (attempt === connectionAttempt) connecting.value = false; }
}
function disconnect() {
  connectionAttempt++; connectionController?.abort(); connecting.value = false;
  token.value = ''; tokenDraft.value = ''; connected.value = false; connectionError.value = '';
  try { sessionStorage.removeItem(tokenStorageKey); } catch { /* Storage unavailable. */ }
}

const props = defineProps<{ data: HomeData }>();
const studio = createStudio(props.data); provide(studioKey, studio);
const { token, info, loading, errors } = studio;
const tabs: StudioTab[] = ['create', 'voices', 'engines', 'settings'];
function tabFromHash(): StudioTab | null {
  const hash = window.location.hash.replace('#/', '').replace('#', '') as StudioTab;
  return tabs.includes(hash) ? hash : null;
}
const active = ref<StudioTab>(tabFromHash() ?? 'create');
onMounted(() => {
  try {
    const saved = sessionStorage.getItem(tokenStorageKey);
    if (saved) { tokenDraft.value = saved; rememberToken.value = true; void connect(); }
  } catch { /* Storage unavailable. */ }
  const sync = () => { const tab = tabFromHash(); if (tab) active.value = tab; };
  window.addEventListener('hashchange', sync);
});
watch(active, value => {
  const hash = `#/${value}`;
  if (window.location.hash !== hash) history.replaceState(null, '', hash);
});
onBeforeUnmount(() => { connectionAttempt++; connectionController?.abort(); });
const modelStatus = computed(() => info.value?.model.status || (props.data.modelReady ? 'ready' : 'idle'));
async function compose() { active.value = 'create'; await nextTick(); document.getElementById('text')?.focus(); }
</script>
<template>
  <StudioLayout v-model="active">
    <template #actions><span v-if="modelStatus !== 'ready'" id="modelStatus" class="model-status">{{ t(modelStatus === 'failed' ? 'Model load failed. Check server logs.' : modelStatus === 'downloading' ? 'Downloading model…' : modelStatus === 'loading' ? 'Loading model...' : 'Preparing model...') }}</span><button :disabled="loading" @click="studio.refresh()">{{ t('Refresh') }}</button></template>
    <template #notice><div v-if="info?.auth.required" id="tokenGroup" class="access-notice"><label for="apiToken">{{ t('API token') }}</label><p class="muted"><a href="/admin">{{ t('create one in the admin panel') }}</a></p><p v-if="connected" role="status">{{ t('Token connected') }}</p><form v-else @submit.prevent="connect"><input id="apiToken" v-model="tokenDraft" type="password" :placeholder="t('Paste your API token here')" autocomplete="off" spellcheck="false" :disabled="connecting"><label><input v-model="rememberToken" type="checkbox" :disabled="connecting"> {{ t('Remember token in this tab') }}</label><button type="submit" :disabled="connecting || !tokenDraft.trim()">{{ t(connecting ? 'Connecting…' : 'Connect token') }}</button></form><button v-if="connected || connecting" type="button" @click="disconnect">{{ t('Disconnect token') }}</button><p v-if="connectionError" class="field-error" role="alert">{{ connectionError }}</p></div><FeedbackNotice :feedback="errors.info ? { kind: 'error', message: errors.info } : null"/></template>
    <template #create><KeepAlive><SpeechGenerator v-if="active === 'create'"/></KeepAlive></template>
    <template #voices><KeepAlive><VoiceLibrary v-if="active === 'voices'" @compose="compose" @engines="active = 'engines'"/></KeepAlive></template>
    <template #engines><KeepAlive><EnginesPanel v-if="active === 'engines'"/></KeepAlive></template>
    <template #settings><KeepAlive><SettingsPanel v-if="active === 'settings'"/></KeepAlive></template>
  </StudioLayout>
</template>
