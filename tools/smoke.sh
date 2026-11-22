#!/bin/bash
# Minimal end-to-end smoke of a running Lapilli server.
# Usage: tools/smoke.sh [base_url] [model_alias]
set -euo pipefail

BASE="${1:-http://127.0.0.1:8080}"
MODEL="${2:-qwen3.8-flash-next-iq3_s}"

echo "== /v1/models =="
curl -fsS "$BASE/v1/models" | python3 -m json.tool | head -40

echo "== completion =="
curl -fsS "$BASE/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  --data-binary "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"What is 2+2? Answer with one word.\"}],\"max_tokens\":32,\"temperature\":0,\"reasoning_effort\":\"none\"}" \
  | python3 -m json.tool
