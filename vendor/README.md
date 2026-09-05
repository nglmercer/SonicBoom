# Vendored TikTools plugin SDK

These three crates are vendored from the TikTools-app `remake` checkout so
SonicBoom can build its TikTools process plugin without requiring a sibling
repository at build time.

The source was synchronized from TikTools commit `85efb5d`. Keep the vendored
SDK aligned with the TikTools host contracts when either side changes the
plugin protocol or ABI.
