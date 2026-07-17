#!/usr/bin/env bash
# PID-one supervisor for the Sleeper container. It runs two long-lived
# processes -- llama-server (the LLM runtime) and sleeper (the voice service) --
# and ties their lifecycles together so a crash in either brings the container
# down with a non-zero status (matching the "exit non-zero so the container
# reports the crash" intent in __main__.py). `wait -n` requires bash.
set -euo pipefail

# HACK: The model is deliberately NOT baked into the image. Ternary-Bonsai-27B is
# large and versioned independently of the app; baking it would balloon the image
# and rebuild/reship it on every code change. Instead llama-server downloads the
# GGUF via -hf into the writable /models bind mount (LLAMA_CACHE=/models) on first
# run and reuses it thereafter. We health-gate on it: sleeper is only started once
# llama-server answers /health, so the download-once + model-load completes before
# any request can reach the LLM. Only these five flags are baked; every other
# llama-server knob (context size, KV-cache quant, speculation, sampling, ...) is
# experimental and supplied at deploy time via "$@" (the container command:).
/opt/llama/llama-server \
  --hf-repo prism-ml/Ternary-Bonsai-27B-gguf \
  --hf-file Ternary-Bonsai-27B-Q2_0.gguf \
  --no-mmproj \
  --host 127.0.0.1 --port 8080 \
  "$@" &
llama_pid=$!

sleeper_pid=""

terminate() {
  kill -TERM "$llama_pid" 2>/dev/null || true
  [ -n "$sleeper_pid" ] && kill -TERM "$sleeper_pid" 2>/dev/null || true
}
trap terminate SIGTERM SIGINT

# Wait for the LLM to load (and download on first run) before starting sleeper.
# On first run llama-server downloads the multi-GB GGUF into /models; its progress
# bar is carriage-return-based and does not surface in Docker logs, so we emit our
# own heartbeat to prove the wait is alive rather than hung. If llama-server dies
# during load, fail fast rather than poll forever.
echo "entrypoint: waiting for llama-server /health (first run downloads the GGUF into /models -- can take a while)" >&2
waited=0
until curl -sf http://127.0.0.1:8080/health >/dev/null 2>&1; do
  if ! kill -0 "$llama_pid" 2>/dev/null; then
    echo "entrypoint: llama-server exited before becoming healthy" >&2
    exit 1
  fi
  sleep 1
  waited=$((waited + 1))
  if [ $((waited % 10)) -eq 0 ]; then
    echo "entrypoint: still waiting for llama-server /health (${waited}s elapsed)" >&2
  fi
done
echo "entrypoint: llama-server healthy; starting sleeper" >&2

uv run --locked sleeper &
sleeper_pid=$!

# Either child exiting is fatal. Reap the first to exit, kill the survivor, and
# propagate the exit status so the container reports the crash.
wait -n "$llama_pid" "$sleeper_pid"
status=$?
terminate
exit "$status"
