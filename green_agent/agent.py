"""Finance agent implementation - manages assessment and evaluation."""

import csv
import json
import os
import random
import time
import tomllib
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import dotenv
import uvicorn
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.apps import A2AStarletteApplication
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCard, AgentCapabilities, AgentSkill, Message, SendMessageSuccessResponse
from a2a.utils import get_text_parts, new_agent_text_message
from get_agent import get_agent
from my_util import my_a2a, parse_tags

dotenv.load_dotenv()


# Cache for questions loaded from CSV
_questions_cache: List[str] = None


def load_questions_from_csv(csv_path: str = None) -> List[str]:
    """
    Load questions from public.csv file.
    
    Args:
        csv_path: Path to the CSV file. If None, uses data/public.csv relative to project root.
    
    Returns:
        List of questions from the CSV file.
    """
    global _questions_cache
    
    # Return cached questions if available
    if _questions_cache is not None:
        return _questions_cache
    
    # Determine CSV path
    if csv_path is None:
        # Get project root (parent of green_agent directory)
        project_root = Path(__file__).parent.parent
        csv_path = project_root / "data" / "public.csv"
    else:
        csv_path = Path(csv_path)
    
    questions = []
    
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                question = row.get('Question', '').strip()
                if question:  # Only add non-empty questions
                    questions.append(question)
        
        print(f"@@@ Loaded {len(questions)} questions from {csv_path}")
        _questions_cache = questions
        return questions
    except FileNotFoundError:
        print(f"@@@ WARNING: CSV file not found at {csv_path}")
        return []
    except Exception as e:
        print(f"@@@ ERROR loading questions from CSV: {e}")
        return []


def get_random_question(csv_path: str = None) -> str:
    """
    Get a random question from public.csv.
    
    Args:
        csv_path: Path to the CSV file. If None, uses data/public.csv relative to project root.
    
    Returns:
        A random question string, or a default question if CSV cannot be loaded.
    """
    questions = load_questions_from_csv(csv_path)
    
    if questions:
        question = random.choice(questions)
        print(f"@@@ Selected random question: {question[:100]}...")
        return question
    else:
        # Fallback to default question if CSV loading fails
        default = "What was Apple's revenue in 2023?"
        print(f"@@@ WARNING: Could not load questions from CSV, using default: {default}")
        return default


# Finance agent specific types (replacing tau_bench types)
class FinanceAgentResult:
    """Result from finance agent execution."""

    def __init__(self, reward: float, info: Dict[str, Any], messages: list, total_cost: float):
        self.reward = reward
        self.info = info
        self.messages = messages
        self.total_cost = total_cost


def load_agent_card_toml(agent_name):
    current_dir = __file__.rsplit("/", 1)[0]
    with open(f"{current_dir}/{agent_name}.toml", "rb") as f:
        card_dict = tomllib.load(f)
    
    # Convert camelCase TOML fields to snake_case for AgentCard
    # A2A TOML spec uses camelCase, but Python AgentCard uses snake_case
    field_mapping = {
        "defaultInputModes": "default_input_modes",
        "defaultOutputModes": "default_output_modes",
    }
    
    converted_dict = {}
    for key, value in card_dict.items():
        if key in field_mapping:
            converted_dict[field_mapping[key]] = value
        elif key == "capabilities" and isinstance(value, dict):
            # Convert capabilities dict to AgentCapabilities object
            converted_dict["capabilities"] = AgentCapabilities(**value)
        elif key == "skills" and isinstance(value, list):
            # Convert skills list of dicts to list of AgentSkill objects
            converted_dict["skills"] = [AgentSkill(**skill) for skill in value]
        else:
            converted_dict[key] = value
    
    return converted_dict


async def ask_white_agent_to_solve(
    white_agent_url: str,
    question: str,
    context: str = "",
    max_iterations: int = 30,
) -> FinanceAgentResult:
    """
    Ask a white agent (external agent) to solve a task via A2A protocol.

    Args:
        white_agent_url (str): URL of the white agent
        question (str): The question to ask
        context (str): Optional context/instructions
        max_iterations (int): Maximum number of message exchanges

    Returns:
        FinanceAgentResult: The result from the white agent
    """
    print(f"@@@ Green agent: Communicating with white agent at {white_agent_url}")

    # Prepare initial message
    if context:
        initial_message = f"{context}\n\nQuestion: {question}"
    else:
        initial_message = question

    # Track conversation
    conversation_history = []
    context_id = None
    final_answer = None

    # Communicate with white agent
    for iteration in range(max_iterations):
        print(f"@@@ Green agent: Iteration {iteration + 1}/{max_iterations}")

        # Send message to white agent
        message_to_send = initial_message if iteration == 0 else question
        print(f"@@@ Green agent: Sending to white agent:\n{message_to_send[:200]}...")

        white_agent_response = await my_a2a.send_message(
            white_agent_url, message_to_send, context_id=context_id
        )

        # Parse response
        res_root = white_agent_response.root
        if not isinstance(res_root, SendMessageSuccessResponse):
            print(f"@@@ Green agent: Unexpected response type: {type(res_root)}")
            break

        res_result = res_root.result
        if not isinstance(res_result, Message):
            print(f"@@@ Green agent: Unexpected result type: {type(res_result)}")
            break

        # Update context_id
        if context_id is None:
            context_id = res_result.context_id
        else:
            if context_id != res_result.context_id:
                print("@@@ Green agent: Warning - context ID changed")

        # Extract text from response
        text_parts = get_text_parts(res_result.parts)
        if not text_parts:
            print("@@@ Green agent: No text in response")
            break

        white_text = "\n".join(text_parts)
        print(f"@@@ White agent response:\n{white_text[:500]}...")

        conversation_history.append({
            "iteration": iteration + 1,
            "question": message_to_send,
            "response": white_text,
        })

        # Check if this looks like a final answer
        # (You can customize this logic based on your white agent's response format)
        final_answer = white_text
        break  # For now, just take the first response

    # Create result
    result = FinanceAgentResult(
        reward=1.0 if final_answer else 0.0,
        info={
            "conversation_history": conversation_history,
            "iterations": len(conversation_history),
            "context_id": context_id,
        },
        messages=conversation_history,
        total_cost=0.0,
    )

    return result




class FinanceAgentExecutor(AgentExecutor):
    def __init__(self, max_num_steps: int = 30, mode: str = "direct"):
        """
        Initialize the Finance Agent Executor.

        Args:
            max_num_steps (int): Maximum number of steps for evaluation (default: 30)
            mode (str): Execution mode - "direct" or "white_agent" (default: "direct")
                - "direct": Runs the finance agent directly (faster, simpler)
                - "white_agent": Tests an external white agent via A2A protocol
        """
        self.max_num_steps = max_num_steps
        self.mode = mode

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        # parse the task
        print(f"Finance agent: Received a task, parsing... (mode: {self.mode})")
        user_input = context.get_user_input()
        print(f"@@@ Raw user input: {user_input[:500]}...")
        tags = parse_tags(user_input)
        print(f"@@@ Parsed tags: {list(tags.keys())}")

        # Extract configuration from user input
        config_str = tags.get("config", tags.get("env_config", "{}"))
        config = json.loads(config_str)

        # Check for mode override in config
        mode = config.get("mode", self.mode)

        # Extract question and white agent URL
        # Priority: 1) <question> tag, 2) config.question, 3) extract from user_input, 4) random from CSV
        question = tags.get("question") or config.get("question")
        
        # If no explicit question, try to extract from user_input
        # (skip if user_input is just instructions about assessing agents)
        if not question:
            # Check if user_input contains actual question content (not just instructions)
            user_input_lower = user_input.lower()
            # If it's just instructions about assessing agents, use a random question from CSV
            if ("assess" in user_input_lower or "task is to" in user_input_lower) and "agent" in user_input_lower:
                # This looks like instructions, not a question - use random question from CSV
                csv_path = config.get("csv_path")  # Allow override via config
                question = get_random_question(csv_path)
                print(f"@@@ WARNING: No explicit question found in message. Using random question from CSV")
            else:
                # Try to use user_input, but clean it up
                question = user_input.strip()
                # Remove common instruction patterns
                if question.startswith("Your task is to") or question.startswith("assess"):
                    csv_path = config.get("csv_path")  # Allow override via config
                    question = get_random_question(csv_path)
                    print(f"@@@ WARNING: User input appears to be instructions. Using random question from CSV")
        
        print(f"@@@ Final question to ask white agent: {question}")
        white_agent_url = tags.get("white_agent_url")

        metrics = {}

        print("Finance agent: Starting evaluation...")
        start_time = datetime.now()
        timestamp_started = time.time()

        # Choose execution mode
        if mode == "white_agent" and white_agent_url:
            # White agent mode: Test an external agent
            print(f"@@@ Green agent mode: Testing white agent at {white_agent_url}")

            context_text = config.get("context", "")
            res = await ask_white_agent_to_solve(
                white_agent_url=white_agent_url,
                question=question,
                context=context_text,
                max_iterations=config.get("max_iterations", self.max_num_steps),
            )

            final_answer = res.info.get("conversation_history", [{}])[-1].get(
                "response", "No response"
            )
            agent_metadata = res.info

        else:
            # Direct mode: Run finance agent directly
            print("@@@ Direct mode: Running finance agent locally")

            # Create agent using get_agent function
            agent = await get_agent(
                model_name=config.get("model", "anthropic/claude-3-7-sonnet-20250219"),
                parameters={
                    "max_turns": config.get("max_turns", self.max_num_steps),
                    "tools": config.get(
                        "tools",
                        [
                            "google_web_search",
                            "retrieve_information",
                            "parse_html_page",
                            "edgar_search",
                        ],
                    ),
                    "max_output_tokens": config.get("max_output_tokens", 16384),
                    "temperature": config.get("temperature", 0.0),
                },
            )

            # Run the agent
            final_answer, agent_metadata = await agent.run(question)

            # Create result object
            res = FinanceAgentResult(
                reward=1.0 if final_answer else 0.0,
                info=agent_metadata,
                messages=[],
                total_cost=0.0,
            )

        end_time = datetime.now()
        duration_seconds = (end_time - start_time).total_seconds()

        metrics["mode"] = mode
        metrics["time_used"] = time.time() - timestamp_started
        metrics["start_time"] = start_time.isoformat()
        metrics["end_time"] = end_time.isoformat()
        metrics["duration_seconds"] = duration_seconds
        metrics["final_answer"] = final_answer
        metrics["agent_metadata"] = agent_metadata
        result_bool = metrics["success"] = res.reward == 1
        result_emoji = "✅" if result_bool else "❌"

        print(f"Finance agent: Evaluation complete. Duration: {duration_seconds:.2f}s")
        await event_queue.enqueue_event(
            new_agent_text_message(
                f"Finished. Agent success: {result_emoji}\nMode: {mode}\nMetrics: {json.dumps(metrics, indent=2)}\nAnswer: {final_answer}\n"
            )
        )

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Cancel the current execution."""
        raise NotImplementedError


def start_finance_agent(
    agent_name="finance_agent",
    host="localhost",
    port=9001,
    mode="direct",
    max_num_steps=30,
):
    """
    Start the Finance Agent A2A server.

    Args:
        agent_name (str): Name of agent config file (default: "finance_agent")
        host (str): Host to bind to (default: "localhost")
        port (int): Port to bind to (default: 9001)
        mode (str): Execution mode - "direct" or "white_agent" (default: "direct")
        max_num_steps (int): Maximum steps/iterations (default: 30)
    """
    print(f"Starting finance agent in {mode} mode...")
    agent_card_dict = load_agent_card_toml(agent_name)
    
    # Following tau-bench example: controller sets AGENT_URL environment variable
    # The controller reads CLOUDRUN_HOST to know its public URL, then sets AGENT_URL for the agent
    # # without controller (for direct mode):
    # url = f"http://{host}:{port}"
    # agent_card_dict["url"] = url
    
    agent_url = os.environ.get("AGENT_URL")
    if agent_url:
        agent_card_dict["url"] = agent_url
        url = agent_url
        print(f"Using AGENT_URL from controller: {agent_url}")
    else:
        # Fallback: construct from host/port (for local development without controller)
        url = f"http://{host}:{port}"
        agent_card_dict["url"] = url
        print(f"WARNING: AGENT_URL not set, using fallback: {url}")
        print(f"  This should be set by the controller when running with agentbeats run_ctrl")
    
    # complete all required card fields

    request_handler = DefaultRequestHandler(
        agent_executor=FinanceAgentExecutor(max_num_steps=max_num_steps, mode=mode),
        task_store=InMemoryTaskStore(),
    )

    app = A2AStarletteApplication(
        agent_card=AgentCard(**agent_card_dict),
        http_handler=request_handler,
    )

    print(f"Mode: {mode}")
    print(f"URL: {url}")
    if mode == "direct":
        print("→ Will run finance agent locally")
    else:
        print("→ Will test external white agents via A2A")

    uvicorn.run(app.build(), host=host, port=port)