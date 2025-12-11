"""CLI entry point for agentify-example-tau-bench."""

import typer
import asyncio

from green_agent import start_finance_agent
from white_agent import start_white_agent
from launcher import launch_evaluation

app = typer.Typer(help="Agentified Finance-Bench - Standardized agent assessment framework")


@app.command()
def green():
    """Start the green agent (assessment manager)."""
    start_finance_agent()


@app.command()
def white():
    """Start the white agent (target being tested)."""
    start_white_agent()


@app.command()
def launch():
    """Launch the complete evaluation workflow."""
    asyncio.run(launch_evaluation())

if __name__ == "__main__":
    app()