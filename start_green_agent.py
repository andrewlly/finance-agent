"""
Startup script for the Finance Agent (green_agent).
This starts the A2A-compatible server.
"""

import sys
import os

# Add the current directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from green_agent import start_finance_agent


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Start the Finance Agent A2A Server")
    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="Host to bind to (default: localhost)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("PORT", 9001)),
        help="Port to bind to (default: 9001 or PORT env var)"
    )
    parser.add_argument(
        "--agent-name",
        type=str,
        default="finance_agent",
        help="Agent name for loading TOML config (default: finance_agent)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="direct",
        choices=["direct", "white_agent"],
        help="Execution mode: 'direct' (run locally) or 'white_agent' (test external agents) (default: direct)"
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=30,
        help="Maximum number of steps/iterations (default: 30)"
    )

    args = parser.parse_args()

    mode_desc = "Run finance agent locally" if args.mode == "direct" else "Test external white agents"

    print(f"""
╔══════════════════════════════════════════════════════════╗
║          Finance Agent A2A Server                        ║
╚══════════════════════════════════════════════════════════╝

Starting server at: http://{args.host}:{args.port}
Agent config: {args.agent_name}.toml
Mode: {args.mode} ({mode_desc})
Max steps: {args.max_steps}

Press Ctrl+C to stop the server.
""")

    start_finance_agent(
        agent_name=args.agent_name,
        host=args.host,
        port=args.port,
        mode=args.mode,
        max_num_steps=args.max_steps,
    )
