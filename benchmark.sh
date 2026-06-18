#!/usr/bin/env bash
set -uo pipefail

BASE_URL="${BASE_URL:-http://127.0.0.1:3000}"
ENDPOINT="$BASE_URL/v1/chat/completions"
MODEL="${MODEL:-llama-3.1-8b-instant}"

echo "=== Semantic Cache Gateway Benchmark ==="
echo "Gateway: $BASE_URL"
echo ""

if ! curl -fsS "$BASE_URL/health" >/dev/null; then
  echo "Gateway is not reachable at $BASE_URL."
  echo "Start it first with: cargo run --release"
  exit 1
fi

pretty_print_response() {
  local body_file="$1"

  if [[ ! -s "$body_file" ]]; then
    echo "(empty response body)"
    return 1
  fi

  if ! python3 -m json.tool "$body_file"; then
    echo "Raw non-JSON response:"
    cat "$body_file"
    echo ""
    return 1
  fi
}

request_json() {
  local method="$1"
  local url="$2"
  local data="${3:-}"
  local body_file
  local status

  body_file="$(mktemp)"

  if [[ "$method" == "POST" ]]; then
    if ! status=$(curl -sS -o "$body_file" -w "%{http_code}" \
      -X POST "$url" \
      -H "Content-Type: application/json" \
      --data "$data"); then
      echo "curl failed while calling $url"
      rm -f "$body_file"
      return 1
    fi
  else
    if ! status=$(curl -sS -o "$body_file" -w "%{http_code}" "$url"); then
      echo "curl failed while calling $url"
      rm -f "$body_file"
      return 1
    fi
  fi

  echo "HTTP $status"
  pretty_print_response "$body_file"
  local rc=$?
  rm -f "$body_file"
  return $rc
}

chat_payload() {
  local prompt="$1"
  python3 - "$MODEL" "$prompt" <<'PY'
import json
import sys

model = sys.argv[1]
prompt = sys.argv[2]
print(json.dumps({
    "model": model,
    "messages": [
        {"role": "user", "content": prompt}
    ]
}))
PY
}

run_case() {
  local title="$1"
  local prompt="$2"

  echo "$title"
  time request_json "POST" "$ENDPOINT" "$(chat_payload "$prompt")"
  echo ""
}

run_case "1. Cache MISS (first request, real LLM call):" \
  "What is arch linux?"

run_case "2. Cache HIT (semantically identical query):" \
  "What is arch linux?"

run_case "3. Cache HIT (semantically SIMILAR but different wording):" \
  "what's arch linux?"

echo "4. Live Metrics:"
request_json "GET" "$BASE_URL/metrics"
