# Dev image only — the deploy image is /Dockerfile. Deps are NOT baked in;
# postCreateCommand runs `uv sync` into the bind-mounted workspace so the
# lockfile stays the single source of truth and lock changes never force an
# image rebuild.
FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
COPY --from=oven/bun:latest /usr/local/bin/bun /bin/

# git/ssh: VS Code source control inside the container.
# git-lfs: model weights (*.onnx, *.npz) are LFS-tracked; without it the
#   working tree holds pointer stubs and onnxruntime dies with
#   "INVALID_PROTOBUF" on load. `git lfs install` registers the smudge filter
#   so checkouts inside the container materialize the real files.
# build-essential: C/C++ toolchain for building wheels that ship no prebuilt binary.
# libportaudio2: sounddevice, imported by the client/say entrypoints.
# libsndfile1: soundfile (pulled by nemo/librosa) dlopens it.
# ffmpeg: audio decode/resample for formats libsndfile doesn't cover.
# zsh: interactive shell, set as root's login shell for VS Code terminals.
#
# The slim image ships a docker-clean hook that deletes downloaded .debs after
# install; drop it and keep packages so the BuildKit cache mounts below actually
# accumulate. /var/cache/apt holds the .debs, /var/lib/apt the package lists —
# both persist on the daemon across rebuilds, so only new/changed packages hit
# the (painfully slow) deb mirrors. sharing=locked serializes concurrent builds
# instead of corrupting the cache.
RUN rm -f /etc/apt/apt.conf.d/docker-clean \
    && echo 'Binary::apt::APT::Keep-Downloaded-Packages "true";' > /etc/apt/apt.conf.d/keep-cache
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install --yes --no-install-recommends \
    git git-lfs build-essential openssh-client zsh libportaudio2 libsndfile1 ffmpeg \
    && git lfs install --system \
    && chsh --shell /bin/zsh root

# omp (oh-my-pi): baked into the image so it survives container rebuilds.
# Its bin is TypeScript executed by bun, hence the bun runtime above.
ENV PATH="/root/.bun/bin:$PATH"
# `pm trust --all` runs the postinstalls bun blocks by default —
# onnxruntime-node fetches its native binary there; without it any omp
# feature touching onnx breaks at runtime.
RUN bun install --global @oh-my-pi/pi-coding-agent \
    && cd /root/.bun/install/global && bun pm trust --all

# uv cache lives on a named volume (different fs than the workspace), so
# hardlinking wheels out of the cache fails; copy instead of warning.
ENV UV_LINK_MODE=copy
