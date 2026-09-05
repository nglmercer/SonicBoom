# SonicBoom TikTools process plugin

This plugin runs SonicBoom as a TikTools process plugin. It prepares the
Supertonic model in the process plugin data directory, synthesizes WAV files,
and returns `HostIntent::AudioPlay` to TikTools. TikTools owns playback; this
binary does not start Axum, Rodio, a tray, or an HTTP listener.

The model is downloaded on first preparation and is never bundled in the
`.plugin` archive. Runtime data is stored below:

```text
<TIKTOOLS_PLUGIN_DATA_DIR>/
├── models/
└── generated/
```

SonicBoom engine code is MIT licensed. Supertonic model weights are distributed
under their own OpenRAIL license and are separate from this plugin's code.

## Local build

The plugin uses the TikTools SDK from the public `remake` branch, so this
repository does not depend on a sibling local checkout:

```bash
cargo build --release --no-default-features --features tiktools-plugin --bin sonicboom-tiktools-plugin
```

To install the reusable Rust packager from the TikTools GitHub branch and create
the installer archive:

```bash
cargo install --git https://github.com/nglmercer/TikTools-app \
  --branch remake --package tiktools-plugin-sdk \
  --features packager --bin tiktools-plugin-pack --locked

tiktools-plugin-pack \
  --manifest plugins/sonicboom.tts/plugin.json \
  --entry target/release/sonicboom-tiktools-plugin \
  --output dist/plugins/sonicboom.tts.plugin
```

On Windows, pass the `.exe` entry path. The CLI writes
`dist/plugins/sonicboom.tts.plugin` with the manifest and required SHA-256
checksums. Model files are downloaded at runtime and are not bundled.

The standalone SonicBoom server remains the default build and continues to
expose `/api/tts` and `/v1/audio/speech`.
