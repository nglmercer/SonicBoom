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
#
# STAGE ORDER IS LOAD-BEARING: `runtime` must remain the LAST stage so a
# plain `docker build .` produces the headless CPU server. The CI
# dockerfile-lint job fails if the final stage is anything else.

# ============================================================================
# Builder: headless CPU server (no ALSA / no local playback)
# edition = "2024" requires rustc >= 1.85 (1.89+ for current locked deps)
#
# The builder is Ubuntu 24.04 — the same distro as the runtime stages — so
# binaries can never link against a newer glibc than the runtime provides.
# ============================================================================
FROM ubuntu:24.04@sha256:69cecf4bbf72d2d44a9eef1b71fb98c7fb973d78af11399deccef19beb008ad9 AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
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

# ort crate downloads a pinned prebuilt ONNX Runtime (see Cargo.lock)
ENV ORT_STRATEGY=download

COPY Cargo.toml Cargo.lock ./

# Warm the registry/git caches (does not hide build failures).
RUN cargo fetch --locked

COPY src ./src
COPY templates ./templates
COPY static ./static
COPY models.sha256.json ./models.sha256.json

RUN cargo build --release --locked --no-default-features --features server

# Stage the ONNX Runtime shared library (if dynamically linked) at a known
# path for the runtime image. Fail the build if it is needed but missing.
RUN set -eu; \
    mkdir -p /build/ort-dist; \
    if ldd /build/target/release/SonicBoom | grep -q libonnxruntime; then \
      matches="$(find /build/target/release/build /root/.cargo /usr/local/cargo -type f -name 'libonnxruntime.so*' 2>/dev/null | sort)"; \
      count="$(printf '%s\n' "$matches" | sed '/^$/d' | wc -l)"; \
      if [ "$count" -ne 1 ]; then echo "ERROR: expected exactly one ONNX Runtime library, found $count" >&2; printf '%s\n' "$matches" >&2; exit 1; fi; \
      echo "Staging $matches"; \
      cp "$matches" /build/ort-dist/; \
    else \
      echo "Binary does not dynamically link libonnxruntime; nothing to stage"; \
    fi

# ============================================================================
# Builder: server with local ALSA playback
#
# Ubuntu 24.04, matching the runtime: no newer-glibc-than-runtime risk.
# ============================================================================
FROM ubuntu:24.04@sha256:69cecf4bbf72d2d44a9eef1b71fb98c7fb973d78af11399deccef19beb008ad9 AS builder-playback

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    ca-certificates \
    libopus-dev \
    libasound2-dev \
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
COPY static ./static
COPY models.sha256.json ./models.sha256.json

RUN cargo build --release --locked

RUN set -eu; \
    mkdir -p /build/ort-dist; \
    if ldd /build/target/release/SonicBoom | grep -q libonnxruntime; then \
      matches="$(find /build/target/release/build /root/.cargo /usr/local/cargo -type f -name 'libonnxruntime.so*' 2>/dev/null | sort)"; \
      count="$(printf '%s\n' "$matches" | sed '/^$/d' | wc -l)"; \
      if [ "$count" -ne 1 ]; then echo "ERROR: expected exactly one ONNX Runtime library, found $count" >&2; printf '%s\n' "$matches" >&2; exit 1; fi; \
      echo "Staging $matches"; \
      cp "$matches" /build/ort-dist/; \
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
    build-essential \
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
COPY static ./static
COPY models.sha256.json ./models.sha256.json

# The `cuda` feature enables the ONNX Runtime CUDA execution provider.
RUN cargo build --release --locked --no-default-features --features server,cuda

RUN set -eu; \
    mkdir -p /build/ort-dist; \
    if ldd /build/target/release/SonicBoom | grep -q libonnxruntime; then \
      matches="$(find /build/target/release/build /root/.cargo /usr/local/cargo -type f -name 'libonnxruntime.so*' 2>/dev/null | sort)"; \
      count="$(printf '%s\n' "$matches" | sed '/^$/d' | wc -l)"; \
      if [ "$count" -ne 1 ]; then echo "ERROR: expected exactly one ONNX Runtime library, found $count" >&2; printf '%s\n' "$matches" >&2; exit 1; fi; \
      echo "Staging $matches"; \
      cp "$matches" /build/ort-dist/; \
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
    for prov in libonnxruntime_providers_shared.so libonnxruntime_providers_cuda.so; do if [ ! -f "/app/ort-lib/$prov" ]; then echo "ERROR: $prov missing from CUDA image" >&2; exit 1; fi; done; \
    echo "Using staged ONNX Runtime: $(ls /app/ort-lib/)"; \
    if ldd /app/sonicboom | grep -q 'not found'; then echo "ERROR: unresolved shared-library dependency" >&2; ldd /app/sonicboom >&2; exit 1; fi; \
    for prov in /app/ort-lib/*.so; do if ldd "$prov" | grep -q 'not found'; then echo "ERROR: unresolved provider dependency in $prov" >&2; ldd "$prov" >&2; exit 1; fi; done

RUN mkdir -p /app/models /app/logs /app/data /app/temp_audio \
    && chown -R sonicboom:sonicboom /app

ENV LD_LIBRARY_PATH=/app/ort-lib:/app
ENV RUST_LOG=info
ENV PORT=17842
ENV MODEL_CACHE_DIR=/app/models
ENV LOG_DIR=/app/logs
ENV TOKEN_STORE_PATH=/app/data/tokens.json
ENV TEMP_AUDIO_DIR=/app/temp_audio

USER sonicboom

EXPOSE 17842

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -fsS http://127.0.0.1:17842/health || exit 1

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
    fi; \
    if ldd /app/sonicboom | grep -q 'not found'; then echo "ERROR: unresolved shared-library dependency" >&2; ldd /app/sonicboom >&2; exit 1; fi

RUN mkdir -p /app/models /app/logs /app/data /app/temp_audio \
    && chown -R sonicboom:sonicboom /app

ENV LD_LIBRARY_PATH=/app/ort-lib:/app
ENV RUST_LOG=info
ENV PORT=17842
ENV MODEL_CACHE_DIR=/app/models
ENV LOG_DIR=/app/logs
ENV TOKEN_STORE_PATH=/app/data/tokens.json
ENV TEMP_AUDIO_DIR=/app/temp_audio

USER sonicboom

EXPOSE 17842

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -fsS http://127.0.0.1:17842/health || exit 1

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
    fi; \
    if ldd /app/sonicboom | grep -q 'not found'; then echo "ERROR: unresolved shared-library dependency" >&2; ldd /app/sonicboom >&2; exit 1; fi

RUN mkdir -p /app/models /app/logs /app/data /app/temp_audio \
    && chown -R sonicboom:sonicboom /app

ENV LD_LIBRARY_PATH=/app/ort-lib:/app
ENV RUST_LOG=info
ENV PORT=17842
ENV MODEL_CACHE_DIR=/app/models
ENV LOG_DIR=/app/logs
ENV TOKEN_STORE_PATH=/app/data/tokens.json
ENV TEMP_AUDIO_DIR=/app/temp_audio

USER sonicboom

EXPOSE 17842

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -fsS http://127.0.0.1:17842/health || exit 1

CMD ["/app/sonicboom"]

