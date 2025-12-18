#!/bin/bash
# AgentBeats controller launch script
# Expects AGENT_TYPE, HOST, AGENT_PORT

set -e

cd "$(dirname "$0")"

HOST=${HOST:-0.0.0.0}

case "${AGENT_TYPE}" in
  green)
    PORT=${AGENT_PORT:-9001}
    exec python start_green_agent.py \
      --host "$HOST" \
      --port "$PORT" \
      --mode white_agent
    ;;
  white)
    PORT=${AGENT_PORT:-9002}
    exec python start_white_agent.py \
      --host "$HOST" \
      --port "$PORT"
    ;;
  *)
    echo "Error: unknown AGENT_TYPE='${AGENT_TYPE}'"
    echo "Expected: green | white"
    exit 1
    ;;
esac