import { it, expect, vi, afterEach } from 'vitest';
import { mount } from '@vue/test-utils';
import { nextTick } from 'vue';
import WaveformPlayer from '../../src/components/audio/WaveformPlayer.vue';
const mock = vi.hoisted(() => {
  const handlers: Record<string, (...args: any[]) => void> = {};
  let current = 0;
  const player = {
    on: vi.fn((name: string, callback: (...args: any[]) => void) => { handlers[name] = callback; }),
    loadBlob: vi.fn(async () => {}), destroy: vi.fn(), setOptions: vi.fn(),
    setTime: vi.fn((time: number) => { current = time; handlers.timeupdate?.(time); }),
    getCurrentTime: () => current, pause: vi.fn(), play: vi.fn(async () => {}),
  };
  return { handlers, player };
});
vi.mock('wavesurfer.js', () => ({ default: { create: () => mock.player } }));
afterEach(() => vi.unstubAllGlobals());
it('shows decoded duration, tracks playback and clamps skips at audio boundaries', async () => {
  vi.stubGlobal('AudioContext', function () {});
  const wrapper = mount(WaveformPlayer, { props: { blob: new Blob(['audio']), label: 'Speech' } });
  mock.handlers.ready(23.75); await nextTick();
  expect(wrapper.text()).toContain('23.75');
  const click = async (label: string) => wrapper.findAll('button').find(button => button.text() === label)!.trigger('click');
  await click('Back 10 seconds'); expect(mock.player.setTime).toHaveBeenLastCalledWith(0);
  mock.handlers.timeupdate(20); await nextTick();
  await click('Forward 10 seconds'); expect(mock.player.setTime).toHaveBeenLastCalledWith(23.75);
  expect(wrapper.emitted('progress')?.at(-1)).toEqual([23.75, 23.75]);
  await click('Stop audio'); expect(mock.player.pause).toHaveBeenCalled();
  expect(mock.player.setTime).toHaveBeenLastCalledWith(0);
  wrapper.unmount();
});
