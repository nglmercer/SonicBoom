# SonicBoom Docker images.
#
# Targets:
#   runtime          - headless network TTS server (CPU, no local playback) [default]
#   runtime-playback - server with local ALSA speaker playback enabled
#   runtime-cuda     - headless network TTS server with CUDA acceleration
#
# All base images are pinned by digest for reproducibility. When updating,
# refresh both the tag and the digest together.
#
# Build examples:
#   docker build -t sonicboom:test .
#   docker build --target runtime -t sonicboom:cpu .
#   docker build --target runtime-playback -t sonicboom:playback .
#   docker build --target runtime-cuda -t sonicboom:cuda .

# ============================================================================
# Builder: headless CPU server (no ALSA / no local playback)
# edition = "2024" requires rustc >= 1.85 (1.89+ for current locked deps)
# ============================================================================
FROM rust:1.89-trixie@sha256:57407b378b2b6e07b48a6135a20c87cc22ea6e249c0acf6cb1833ead3cf116e9 AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    libopus-dev \
    pkg-config \
    libssl-dev \
    cmake \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# ort crate downloads a pinned prebuilt ONNX Runtime (see Cargo.lock: ort 2.0.0-rc.11)
ENV ORT_STRATEGY=download

COPY Cargo.toml Cargo.lock ./

# Warm the registry/git caches (does not hide build failures).
RUN cargo fetch --locked

COPY src ./src
COPY templates ./templates

RUN cargo build --release --locked --no-default-features --features server

# Stage the ONNX Runtime shared library (if dynamically linked) at a known
# path for the runtime image. Fail the build if it is needed but missing.
RUN set -e; \
    mkdir -p /build/ort-dist; \
    if ldd /build/target/release/SonicBoom | grep -q libonnxruntime; then \
      ORT_LIB="$(find /build/target/release/build /root/.cargo /usr/local/cargo -name 'libonnxruntime.so*' 2>/dev/null | sort | head -1)"; \
      if [ -z "$ORT_LIB" ]; then echo "ERROR: libonnxruntime required by binary but not found in builder" >&2; exit 1; fi; \
      echo "Staging $ORT_LIB"; \
      cp "$ORT_LIB" /build/ort-dist/; \
    else \
      echo "Binary does not dynamically link libonnxruntime; nothing to stage"; \
    fi

# ============================================================================
# Builder: server with local ALSA playback
# ============================================================================
FROM rust:1.89-trixie@sha256:57407b378b2b6e07b48a6135a20c87cc22ea6e249c0acf6cb1833ead3cf116e9 AS builder-playback

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    libopus-dev \
    libasound2-dev \
    pkg-config \
    libssl-dev \
    cmake \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

ENV ORT_STRATEGY=download

COPY Cargo.toml Cargo.lock ./

RUN cargo fetch --locked

COPY src ./src
COPY templates ./templates

RUN cargo build --release --locked

RUN set -e; \
    mkdir -p /build/ort-dist; \
    if ldd /build/target/release/SonicBoom | grep -q libonnxruntime; then \
      ORT_LIB="$(find /build/target/release/build /root/.cargo /usr/local/cargo -name 'libonnxruntime.so*' 2>/dev/null | sort | head -1)"; \
      if [ -z "$ORT_LIB" ]; then echo "ERROR: libonnxruntime required by binary but not found in builder" >&2; exit 1; fi; \
      echo "Staging $ORT_LIB"; \
      cp "$ORT_LIB" /build/ort-dist/; \
    else \
      echo "Binary does not dynamically link libonnxruntime; nothing to stage"; \
    fi

# ============================================================================
# Builder: CUDA-enabled headless server
# ============================================================================
FROM nvidia/cuda:12.9.2-devel-ubuntu24.04@sha256:16656a1ef115bca9e1f820c6349876f1486d2b3c9a0e615773799fe402960dc5 AS builder-cuda

WORKDIR /build

# Rust 1.89 toolchain + native build deps on top of the CUDA devel image.
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    libopus-dev \
    pkg-config \
    libssl-dev \
    cmake \
    && rm -rf /var/lib/apt/lists/* \
    && curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
        | sh -s -- -y --profile minimal --default-toolchain 1.89.0
ENV PATH="/root/.cargo/bin:${PATH}"

ENV ORT_STRATEGY=download

COPY Cargo.toml Cargo.lock ./

RUN cargo fetch --locked

COPY src ./src
COPY templates ./templates

# The `cuda` feature enables the ONNX Runtime CUDA execution provider.
RUN cargo build --release --locked --no-default-features --features server,cuda

RUN set -e; \
    mkdir -p /build/ort-dist; \
    if ldd /build/target/release/SonicBoom | grep -q libonnxruntime; then \
      ORT_LIBS="$(find /build/target/release/build /root/.cargo /usr/local/cargo -name 'libonnxruntime.so*' 2>/dev/null | sort)"; \
      if [ -z "$ORT_LIBS" ]; then echo "ERROR: libonnxruntime required by binary but not found in builder" >&2; exit 1; fi; \
      echo "Staging: $ORT_LIBS"; \
      cp $ORT_LIBS /build/ort-dist/; \
    else \
      echo "Binary does not dynamically link libonnxruntime; nothing to stage"; \
    fi
# The CUDA execution provider additionally needs its shared provider
# libraries at runtime. The ort build script places them next to the binary
# in the deterministic cargo output dir; stage them and fail if absent.
RUN set -e; \
    for lib in libonnxruntime_providers_shared.so libonnxruntime_providers_cuda.so; do \
      if [ ! -f "/build/target/release/deps/$lib" ]; then echo "ERROR: $lib not found in builder output" >&2; exit 1; fi; \
      echo "Staging $lib"; \
      cp "/build/target/release/deps/$lib" /build/ort-dist/; \
    done; \
    ls /build/ort-dist/

# ============================================================================
# Runtime: CUDA-enabled headless server
# docker build --target runtime-cuda -t sonicboom:cuda .
# ============================================================================
FROM nvidia/cuda:12.9.2-cudnn-runtime-ubuntu24.04@sha256:070f8f2672df1b05b84c0409a5fd1d54ddfd646e5b9d8dee7878131271b563fc AS runtime-cuda

RUN apt-get update && apt-get install -y --no-install-recommends \
    libopus0 \
    libssl3 \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --system --uid 10001 --create-home --home-dir /app sonicboom

WORKDIR /app

COPY --from=builder-cuda /build/target/release/SonicBoom /app/sonicboom
COPY --from=builder-cuda /build/ort-dist/ /app/ort-lib/

# The CUDA execution provider cannot work without its shared provider
# libraries; fail the build unless they were staged.
RUN set -e; \
    if ldd /app/sonicboom | grep -q libonnxruntime; then \
      if [ -z "$(ls -A /app/ort-lib/)" ]; then echo "ERROR: libonnxruntime required but not staged" >&2; exit 1; fi; \
    fi; \
    if [ ! -f /app/ort-lib/libonnxruntime_providers_shared.so ]; then echo "ERROR: libonnxruntime_providers_shared.so missing from CUDA image" >&2; exit 1; fi; \
    echo "Using staged ONNX Runtime: $(ls /app/ort-lib/)"

RUN mkdir -p /app/models /app/logs /app/data /app/temp_audio \
    && chown -R sonicboom:sonicboom /app

ENV LD_LIBRARY_PATH=/app/ort-lib:/app
ENV RUST_LOG=info
ENV PORT=3000
ENV MODEL_CACHE_DIR=/app/models
ENV LOG_DIR=/app/logs
ENV TOKEN_STORE_PATH=/app/data/tokens.json
ENV TEMP_AUDIO_DIR=/app/temp_audio

USER sonicboom

EXPOSE 3000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -fsS http://127.0.0.1:3000/health || exit 1

CMD ["/app/sonicboom"]
# ============================================================================
# Runtime: server with local ALSA playback
# ============================================================================
FROM ubuntu:24.04@sha256:69cecf4bbf72d2d44a9eef1b71fb98c7fb973d78af11399deccef19beb008ad9 AS runtime-playback

RUN apt-get update && apt-get install -y --no-install-recommends \
    libopus0 \
    libasound2t64 \
    libssl3 \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --system --uid 10001 --create-home --home-dir /app sonicboom

WORKDIR /app

COPY --from=builder-playback /build/target/release/SonicBoom /app/sonicboom
COPY --from=builder-playback /build/ort-dist/ /app/ort-lib/

RUN set -e; \
    if ldd /app/sonicboom | grep -q libonnxruntime; then \
      if [ -z "$(ls -A /app/ort-lib/)" ]; then echo "ERROR: libonnxruntime required but not staged" >&2; exit 1; fi; \
      echo "Using staged ONNX Runtime: $(ls /app/ort-lib/)"; \
    fi

RUN mkdir -p /app/models /app/logs /app/data /app/temp_audio \
    && chown -R sonicboom:sonicboom /app

ENV LD_LIBRARY_PATH=/app/ort-lib:/app
ENV RUST_LOG=info
ENV PORT=3000
ENV MODEL_CACHE_DIR=/app/models
ENV LOG_DIR=/app/logs
ENV TOKEN_STORE_PATH=/app/data/tokens.json
ENV TEMP_AUDIO_DIR=/app/temp_audio

USER sonicboom

EXPOSE 3000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -fsS http://127.0.0.1:3000/health || exit 1

CMD ["/app/sonicboom"]

# ============================================================================
# Runtime: headless CPU server (default)
# ============================================================================
FROM ubuntu:24.04@sha256:69cecf4bbf72d2d44a9eef1b71fb98c7fb973d78af11399deccef19beb008ad9 AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    libopus0 \
    libssl3 \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --system --uid 10001 --create-home --home-dir /app sonicboom

WORKDIR /app

COPY --from=builder /build/target/release/SonicBoom /app/sonicboom
COPY --from=builder /build/ort-dist/ /app/ort-lib/

# Fail the build if the binary needs libonnxruntime but none was staged.
RUN set -e; \
    if ldd /app/sonicboom | grep -q libonnxruntime; then \
      if [ -z "$(ls -A /app/ort-lib/)" ]; then echo "ERROR: libonnxruntime required but not staged" >&2; exit 1; fi; \
      echo "Using staged ONNX Runtime: $(ls /app/ort-lib/)"; \
    fi

RUN mkdir -p /app/models /app/logs /app/data /app/temp_audio \
    && chown -R sonicboom:sonicboom /app

ENV LD_LIBRARY_PATH=/app/ort-lib:/app
ENV RUST_LOG=info
ENV PORT=3000
ENV MODEL_CACHE_DIR=/app/models
ENV LOG_DIR=/app/logs
ENV TOKEN_STORE_PATH=/app/data/tokens.json
ENV TEMP_AUDIO_DIR=/app/temp_audio

USER sonicboom

EXPOSE 3000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -fsS http://127.0.0.1:3000/health || exit 1

CMD ["/app/sonicboom"]

