"""White agent implementation - the target agent being tested."""

# Load environment variables FIRST before any other imports
import dotenv
dotenv.load_dotenv()

import os
import uvicorn
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentSkill, AgentCard, AgentCapabilities
from a2a.utils import new_agent_text_message
from get_agent import get_agent


def prepare_white_agent_card(url):
    skill = AgentSkill(
        id="finance_tasks",
        name="Finance Task Completion",
        description="Answers finance-related questions using web search, EDGAR filings, and analysis tools",
        tags=["finance", "research", "analysis"],
        examples=[],
    )
    card = AgentCard(
        name="finance_white_agent",
        description="Finance agent for answering financial questions - WHITE AGENT (being tested)",
        url=url,
        version="1.0.0",
        default_input_modes=["text"],
        default_output_modes=["text"],
        capabilities=AgentCapabilities(),
        skills=[skill],
    )
    return card


class FinanceWhiteAgentExecutor(AgentExecutor):
    def __init__(
        self,
        model_name="anthropic/claude-3-7-sonnet-20250219",
        max_turns=30,
        tools=None,
    ):
        """
        Initialize the Finance White Agent Executor.

        Args:
            model_name (str): LLM model to use
            max_turns (int): Maximum agent turns
            tools (list): List of tools to enable
        """
        self.model_name = model_name
        self.max_turns = max_turns
        self.tools = tools or [
            "google_web_search",
            "retrieve_information",
            "parse_html_page",
            "edgar_search",
        ]
        self.agent = None

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Execute the finance agent for the given task."""
        try:
            # Get user question
            user_input = context.get_user_input()
            print(f"White agent: Received question: {user_input}")

            # Create agent if not already initialized
            if self.agent is None:
                print(f"White agent: Initializing with model {self.model_name}")
                self.agent = await get_agent(
                    model_name=self.model_name,
                    parameters={
                        "max_turns": self.max_turns,
                        "tools": self.tools,
                        "max_output_tokens": 16384,
                        "temperature": 0.0,
                    },
                )

            # Run the agent
            print("White agent: Processing query...")
            answer, metadata = await self.agent.run(user_input)

            # Send response
            print(f"White agent: Sending answer ({len(answer)} chars)")
            await event_queue.enqueue_event(
                new_agent_text_message(answer, context_id=context.context_id)
            )

        except Exception as e:
            error_msg = f"White agent error: {str(e)}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            await event_queue.enqueue_event(
                new_agent_text_message(error_msg, context_id=context.context_id)
            )

    async def cancel(self, context, event_queue) -> None:
        """Cancel the current execution."""
        print("White agent: Task cancelled")
        await event_queue.enqueue_event(
            new_agent_text_message("Task cancelled", context_id=context.context_id)
        )


def start_white_agent(
    host="localhost",
    port=9002,
    model_name="anthropic/claude-3-7-sonnet-20250219",
    max_turns=30,
    tools=None,
):
    """
    Start the White Agent A2A server.

    Args:
        host (str): Host to bind to
        port (int): Port to bind to
        model_name (str): LLM model to use
        max_turns (int): Maximum agent turns
        tools (list): List of tools to enable
    """
    print(f"""
╔══════════════════════════════════════════════════════════╗
║          Finance WHITE Agent (Being Tested)              ║
╚══════════════════════════════════════════════════════════╝

Starting white agent at: http://{host}:{port}
Model: {model_name}
Max turns: {max_turns}
Tools: {tools or "all"}

This is the agent that will be TESTED by the green agent.
Press Ctrl+C to stop the server.
""")

    # Following AgentBeats pattern: controller sets AGENT_URL environment variable
    # The controller reads CLOUDRUN_HOST/RAILWAY_PUBLIC_DOMAIN to know its public URL, then sets AGENT_URL for the agent
    # # without controller (for local development):
    # url = f"http://{host}:{port}"
    
    agent_url = os.environ.get("AGENT_URL")
    if agent_url:
        url = agent_url
        print(f"Using AGENT_URL from controller: {agent_url}")
    else:
        # Fallback: construct from host/port (for local development without controller)
        url = f"http://{host}:{port}"
        print(f"WARNING: AGENT_URL not set, using fallback: {url}")
        print(f"  This should be set by the controller when running with agentbeats run_ctrl")
    
    card = prepare_white_agent_card(url)
    
    print(f"URL: {url}")

    request_handler = DefaultRequestHandler(
        agent_executor=FinanceWhiteAgentExecutor(
            model_name=model_name, max_turns=max_turns, tools=tools
        ),
        task_store=InMemoryTaskStore(),
    )

    app = A2AStarletteApplication(
        agent_card=card,
        http_handler=request_handler,
    )

    uvicorn.run(app.build(), host=host, port=port)