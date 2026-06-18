# syntax=docker/dockerfile:1

FROM rust:slim-trixie AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    cmake \
    libssl-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY Cargo.toml Cargo.lock* ./

# RUN if [ -f Cargo.lock ]; then \
#         cargo fetch --locked; \
#     else \
#         cargo fetch; \
#     fi

COPY src ./src

RUN cargo build --release --bin semantic_gateway

RUN mkdir -p /app/ort-libs \
    && find /app/target -type f \( \
        -name 'libonnxruntime*.so*' \
        -o -name 'onnxruntime*.dll' \
        -o -name 'libonnxruntime*.dylib' \
    \) -exec cp -v {} /app/ort-libs/ \; || true

FROM debian:trixie-slim AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    libgcc-s1 \
    libgomp1 \
    libssl3t64 \
    libstdc++6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /app/target/release/semantic_gateway /usr/local/bin/semantic_gateway
COPY --from=builder /app/ort-libs/ /app/
COPY model ./model

RUN printf '%s\n' \
    '#!/bin/sh' \
    'set -eu' \
    'echo "Semantic Cache Gateway container starting..."' \
    'echo "QDRANT_URL=${QDRANT_URL:-http://qdrant:6334}"' \
    'if [ "${WAIT_FOR_QDRANT:-true}" = "true" ]; then' \
    '  echo "Waiting for Qdrant REST health endpoint..."' \
    '  until curl -fsS "http://qdrant:6333/healthz" >/dev/null; do' \
    '    sleep 1' \
    '  done' \
    'fi' \
    'echo "Starting semantic_gateway..."' \
    'exec /usr/local/bin/semantic_gateway' \
    > /usr/local/bin/gateway-entrypoint.sh \
    && chmod +x /usr/local/bin/gateway-entrypoint.sh

ENV APP_HOST=0.0.0.0
ENV APP_PORT=3000
ENV QDRANT_URL=http://qdrant:6334
ENV CACHE_SCORE_THRESHOLD=0.90
ENV LD_LIBRARY_PATH=/app
ENV WAIT_FOR_QDRANT=true

EXPOSE 3000

ENTRYPOINT ["/usr/local/bin/gateway-entrypoint.sh"]
