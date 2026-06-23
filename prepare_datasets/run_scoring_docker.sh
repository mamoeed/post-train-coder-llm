#!/usr/bin/env bash
# run_scoring_docker.sh
# Build the scoring image and run score_dolci.py inside an isolated container.
#
# Isolation applied:
#   --network none        : no network access for model-written code
#   --memory / --pids      : cap memory and process count (stops fork bombs / OOM)
#   --read-only + tmpfs    : container FS read-only except a small writable /tmp
#   non-root user (image)  : code runs as uid 1000, not root
#
# Environment:
#   Standard Library ONLY (LiveCodeBench alignment). External imports will fail.

set -euo pipefail

IMAGE="dolci-scoring:latest"
WORKDIR="$(pwd)"

# 1. Build (cheap after first time; layers cached).
docker build -f Dockerfile.scoring -t "$IMAGE" .

# 2. Run scoring.
docker run --rm \
  --network none \
  --memory=4g \
  --memory-swap=4g \
  --pids-limit=512 \
  --cpus=4 \
  --read-only \
  --tmpfs /tmp:rw,size=512m,exec \
  --shm-size=1g \
  -v "$WORKDIR/score_dolci.py:/work/score_dolci.py:ro" \
  -v "$WORKDIR/out:/work/out:rw" \
  "$IMAGE" \
  --clean out/dolci_clean.jsonl \
  --out out/dolci_scored.jsonl \
  --violations out/dolci_sanity_violations.jsonl \
  "$@"
