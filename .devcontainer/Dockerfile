# Dev image only — the deploy image is /Dockerfile. Deps are NOT baked in;
# postCreateCommand runs `uv sync` into the bind-mounted workspace so the
# lockfile stays the single source of truth and lock changes never force an
# image rebuild.
FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# git/ssh: VS Code source control inside the container.
# libportaudio2: sounddevice is imported at module top in serve_chat.py.
# libsndfile1: soundfile (pulled by nemo/librosa) dlopens it.
RUN apt-get update && apt-get install --yes --no-install-recommends \
    git openssh-client libportaudio2 libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# uv cache lives on a named volume (different fs than the workspace), so
# hardlinking wheels out of the cache fails; copy instead of warning.
ENV UV_LINK_MODE=copy
