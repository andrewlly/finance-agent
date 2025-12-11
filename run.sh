#!/bin/bash
# AgentBeats controller launch script for WHITE agent
# This script is called by the AgentBeats controller to start the white agent
# The controller sets HOST and AGENT_PORT environment variables

# Change to script directory (project root)
cd "$(dirname "$0")"

python start_white_agent.py --host ${HOST:-0.0.0.0} --port ${AGENT_PORT:-9002}

