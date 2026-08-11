#!/bin/bash
# Pre-warm higgs radix prefix cache on startup.
# Sends one dummy request with a long system prompt to populate the radix.
# The user's first real request then hits the warm radix (~1s TTFT).
# Run AFTER higgs starts: ./prewarm.sh &

HIGGS="http://127.0.0.1:8000"
MODEL="bonsai-27b"

echo "Waiting for higgs..."
while ! curl -sf "$HIGGS/health" >/dev/null 2>&1; do sleep 2; done
echo "higgs ready. Pre-warming radix (this takes ~60s, runs in background)..."

# A long stable system prompt that matches the token volume of nanobot's
# system+tools (~3000 tokens). Replace with the ACTUAL nanobot system prompt
# for a perfect radix hit. Even an approximate match helps — radix caches
# at the block level (~64 tokens).
SYSTEM=$(python3 -c "
parts = [
    'You are nanobot, a tool-using assistant.',
    ' You help with coding, research, and tasks.',
] * 50
print(' '.join(parts))
")

curl -sf --max-time 180 "$HIGGS/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d "{
    \"model\": \"$MODEL\",
    \"messages\": [{\"role\":\"system\",\"content\":\"$SYSTEM\"},{\"role\":\"user\",\"content\":\"ok\"}],
    \"max_tokens\": 1,
    \"temperature\": 0
  }" >/dev/null 2>&1 &

echo "Pre-warm running in background. First real request will be fast."
