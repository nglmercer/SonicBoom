# Stage 1: Build
FROM rust:1.83-bookworm AS builder

WORKDIR /build

# 빌드 의존성 설치
RUN apt-get update && apt-get install -y \
    libopus-dev \
    pkg-config \
    libssl-dev \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# ort 크레이트가 ONNX Runtime 라이브러리를 자동 다운로드하도록 설정
ENV ORT_STRATEGY=download

COPY Cargo.toml Cargo.lock ./

# 의존성 캐싱을 위한 더미 빌드
RUN mkdir src && echo 'fn main(){}' > src/main.rs \
    && cargo build --release 2>/dev/null || true \
    && rm -rf src

COPY src ./src

RUN cargo build --release

# Stage 2: Runtime (CPU only)
FROM ubuntu:24.04 AS runtime

RUN apt-get update && apt-get install -y \
    libopus0 \
    libssl3 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /build/target/release/SonicBoom /app/sonicboom

# ORT가 다운로드한 공유 라이브러리 복사 (있는 경우)
RUN find / -name "libonnxruntime*.so*" 2>/dev/null | head -1 | xargs -I{} cp {} /app/ || true

ENV LD_LIBRARY_PATH=/app
ENV RUST_LOG=info
ENV PORT=3000

EXPOSE 3000

CMD ["/app/sonicboom"]

# Stage 2 (CUDA): nvidia/cuda 기반 런타임
# docker build --target runtime-cuda -t sonicboom:cuda .
FROM nvidia/cuda:12.4.0-runtime-ubuntu24.04 AS runtime-cuda

RUN apt-get update && apt-get install -y \
    libopus0 \
    libssl3 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /build/target/release/SonicBoom /app/sonicboom

ENV RUST_LOG=info
ENV PORT=3000

EXPOSE 3000

CMD ["/app/sonicboom"]
