# ---------------------------------------------------------------------------
# Stage 1: build the PrismML fork of llama.cpp with CUDA.
#
# The -devel CUDA base ships only the CUDA toolkit, not a host toolchain, so we
# install gcc/g++ ourselves (mirrors the fork's own .devops/cuda.Dockerfile).
# libssl-dev is required because -hf HTTPS downloads use llama.cpp's built-in
# OpenSSL HTTP client (LLAMA_CURL is deprecated in this fork; LLAMA_OPENSSL
# defaults ON).
# ---------------------------------------------------------------------------
FROM nvidia/cuda:12.8.1-devel-ubuntu24.04 AS llama-build

# CUDA arch is overridable at build time without editing this file. Default 86
# targets the deploy host's RTX 3090 (SM 8.6).
ARG CUDA_DOCKER_ARCH=86

RUN apt-get update && apt-get install -y --no-install-recommends \
      gcc-14 g++-14 build-essential cmake git ca-certificates libssl-dev \
    && rm -rf /var/lib/apt/lists/*

ENV CC=gcc-14 CXX=g++-14 CUDAHOSTCXX=g++-14

# Pin the fork to an exact commit for reproducibility (release tag
# prism-b9594-38c66ad, branch prism).
RUN git clone https://github.com/PrismML-Eng/llama.cpp.git /src \
    && git -C /src checkout 38c66ad0241da4f9fcce541cda8edc219086cec5

# Build flags copied from the fork's .devops/cuda.Dockerfile, with the arch
# fixed and tests off.
RUN cmake -S /src -B /src/build \
      -DGGML_NATIVE=OFF -DGGML_CUDA=ON -DGGML_BACKEND_DL=ON -DGGML_CPU_ALL_VARIANTS=ON \
      -DLLAMA_BUILD_TESTS=OFF -DCMAKE_CUDA_ARCHITECTURES=${CUDA_DOCKER_ARCH} \
      -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXE_LINKER_FLAGS=-Wl,--allow-shlib-undefined \
    && cmake --build /src/build --config Release -j"$(nproc)"

# Collect artifacts into one directory so the backend-DL loader (searches the
# executable dir) and ld.so both resolve co-located libs.
RUN mkdir -p /out && cp /src/build/bin/llama-server /out/ \
    && find /src/build -name "*.so*" -exec cp -P {} /out/ \;

# ---------------------------------------------------------------------------
# Stage 2: runtime image. The CUDA -runtime base is minimal, so we install the
# Python toolchain and runtime libs ourselves. Ubuntu 24.04 ships Python 3.12,
# matching requires-python = ">=3.12,<3.13".
#
# Per-package rationale:
#   python3.12* + build-essential + python3.12-dev -- preserve the
#     venv-build-at-startup path (native wheels may compile; the old python:3.12
#     base provided a toolchain via buildpack-deps).
#   libgomp1  -- ggml OpenMP runtime.
#   libssl3 + ca-certificates -- the -hf HTTPS download.
#   curl      -- the entrypoint health poll.
#   xz-utils  -- the ffmpeg `tar --xz` extraction (Debian's base bundled xz;
#                Ubuntu minimal does not).
# ---------------------------------------------------------------------------
FROM nvidia/cuda:12.8.1-runtime-ubuntu24.04

RUN apt-get update && apt-get install -y --no-install-recommends \
      python3.12 python3.12-venv python3.12-dev build-essential \
      libgomp1 libssl3 ca-certificates curl xz-utils \
    && ln -sf /usr/bin/python3.12 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

# uv reads .python-version (3.12) and uses the system 3.12 interpreter.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# The llama.cpp artifacts and expensive apt layers sit before `ADD . /app` so
# app-source changes don't invalidate them.
COPY --from=llama-build /out /opt/llama
# Co-located libs resolve without patchelf/rpath.
ENV LD_LIBRARY_PATH=/opt/llama
# llama.cpp reads LLAMA_CACHE first for its HF cache dir, so -hf caches into the
# writable /models bind mount.
ENV LLAMA_CACHE=/models

# Install ffmpeg from BtbN's static build rather than Ubuntu's apt (whose mirrors
# are painfully slow). This is a statically-linked GPL binary for x86_64 glibc,
# so it needs no runtime libs -- we just extract ffmpeg/ffprobe onto the PATH.
ADD https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-linux64-gpl.tar.xz /tmp/ffmpeg.tar.xz
RUN tar --extract --xz --file /tmp/ffmpeg.tar.xz --directory /tmp \
    && mv /tmp/ffmpeg-master-latest-linux64-gpl/bin/ffmpeg /tmp/ffmpeg-master-latest-linux64-gpl/bin/ffprobe /usr/local/bin/ \
    && rm -rf /tmp/ffmpeg.tar.xz /tmp/ffmpeg-master-latest-linux64-gpl

COPY --chmod=0755 docker/entrypoint.sh /usr/local/bin/sleeper-entrypoint

# Change the working directory to the `app` directory
WORKDIR /app

# Copy the project into the image
ADD . /app

# Build the environment from the destination's mounted uv cache at startup
# rather than baking .venv into the image. Baking it produces a ~7GB image
# with no caching opportunity: it gets rebuilt and shipped dozens of times a
# day, and every push transfers the whole thing. Instead the deploy target
# keeps a bind-mounted uv cache, so `uv run` here only materializes the venv
# from already-cached wheels -- new/changed deps download, everything else is
# already local. UV_LINK_MODE=copy avoids hardlink warnings across the
# cache/venv filesystem boundary.
ENV UV_LINK_MODE=copy

# The entrypoint supervises both llama-server and sleeper. CMD is the
# experimental extra-args slot: whatever is in CMD / compose `command:` is
# appended to the baked llama-server flags by the entrypoint.
ENTRYPOINT ["/usr/local/bin/sleeper-entrypoint"]
CMD []
