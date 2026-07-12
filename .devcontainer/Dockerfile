# Dev image only — the deploy image is /Dockerfile. Deps are NOT baked in;
# postCreateCommand runs `uv sync` into the bind-mounted workspace so the
# lockfile stays the single source of truth and lock changes never force an
# image rebuild.
FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
COPY --from=oven/bun:latest /usr/local/bin/bun /bin/

# git/ssh: VS Code source control inside the container.
# libportaudio2: sounddevice is imported at module top in serve_chat.py.
# libsndfile1: soundfile (pulled by nemo/librosa) dlopens it.
# zsh: interactive shell, set as root's login shell for VS Code terminals.
RUN apt-get update && apt-get install --yes --no-install-recommends \
    git openssh-client zsh libportaudio2 libsndfile1 \
    && rm -rf /var/lib/apt/lists/* \
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
