"""
Startup script for the Finance WHITE Agent.
This is the agent that gets TESTED by the green agent.
"""

import sys
import os

# Add the current directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from white_agent import start_white_agent


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Start the Finance WHITE Agent A2A Server")
    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="Host to bind to (default: localhost)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=9002,
        help="Port to bind to (default: 9002)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="anthropic/claude-3-7-sonnet-20250219",
        help="Model to use (default: anthropic/claude-3-7-sonnet-20250219)"
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=30,
        help="Maximum number of turns (default: 30)"
    )
    parser.add_argument(
        "--tools",
        type=str,
        nargs="+",
        default=None,
        help="Tools to enable (default: all)"
    )

    args = parser.parse_args()

    start_white_agent(
        host=args.host,
        port=args.port,
        model_name=args.model,
        max_turns=args.max_turns,
        tools=args.tools,
    )
