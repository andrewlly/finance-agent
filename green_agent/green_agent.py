"""Finance agent implementation - manages assessment and evaluation."""

import sys
import os
from pathlib import Path

# ==============================================================================
# PATH SETUP (MUST BE FIRST)
# ==============================================================================
# 1. Get the directory where this script is located (green_agent/)
current_dir = Path(__file__).resolve().parent

# 2. Get the project root (finance-agent/)
project_root = current_dir.parent

# 3. Insert project root at the VERY START of sys.path
# This ensures we import 'agent.py' and 'tools.py' from the root, 
# not from the local folder if names conflict.
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Debug print to confirm path is correct
print(f"@@@ System Path Updated. Root: {project_root}")

# ==============================================================================
# IMPORTS
# ==============================================================================

import csv
import json
import random
import time
import tomllib
import asyncio
import traceback
import dotenv
import uvicorn
from datetime import datetime
from typing import Any, Dict, List

# --- A2A IMPORTS ---
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.apps import A2AStarletteApplication
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCard, AgentCapabilities, AgentSkill, Message, SendMessageSuccessResponse
from a2a.utils import get_text_parts, new_agent_text_message

# --- CUSTOM IMPORTS ---
# Now these should work because project_root is in sys.path
try:
    from agent import Agent  # This finds agent.py in the root
    from model_library.base import LLM
    from tools import get_tool_by_name 
    from my_util import my_a2a, parse_tags
except ImportError as e:
    print("\n" + "="*60)
    print(f"CRITICAL IMPORT ERROR: {e}")
    print(f"Looking in: {project_root}")
    print("If you have a file named 'agent.py' in the 'green_agent' folder, RENAME IT.")
    print("It is blocking the import of the main 'agent.py'.")
    print("="*60 + "\n")
    sys.exit(1)

dotenv.load_dotenv()

# -------------------------------------------------------------------------
# HELPER FUNCTIONS
# -------------------------------------------------------------------------

_questions_cache: List[str] = None

def load_questions_from_csv(csv_path: str = None) -> List[str]:
    global _questions_cache
    if _questions_cache is not None:
        return _questions_cache
    
    if csv_path is None:
        csv_path = project_root / "data" / "public.csv"
    else:
        csv_path = Path(csv_path)
    
    questions = []
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                q = row.get('Question', '').strip()
                if q: questions.append(q)
        _questions_cache = questions
        return questions
    except Exception as e:
        print(f"@@@ ERROR loading questions: {e}")
        return []

def get_random_question(csv_path: str = None) -> str:
    questions = load_questions_from_csv(csv_path)
    if questions:
        return random.choice(questions)
    return "What was Apple's revenue in 2023?"

def load_agent_card_toml(agent_name):
    # Load config from the same directory as this script
    toml_path = current_dir / f"{agent_name}.toml"
    with open(toml_path, "rb") as f:
        card_dict = tomllib.load(f)
    
    field_mapping = {
        "defaultInputModes": "default_input_modes",
        "defaultOutputModes": "default_output_modes",
    }
    converted_dict = {}
    for key, value in card_dict.items():
        if key in field_mapping:
            converted_dict[field_mapping[key]] = value
        elif key == "capabilities" and isinstance(value, dict):
            converted_dict["capabilities"] = AgentCapabilities(**value)
        elif key == "skills" and isinstance(value, list):
            converted_dict["skills"] = [AgentSkill(**skill) for skill in value]
        else:
            converted_dict[key] = value
    return converted_dict

# -------------------------------------------------------------------------
# EXECUTOR LOGIC
# -------------------------------------------------------------------------

class FinanceAgentResult:
    def __init__(self, reward: float, info: Dict[str, Any], messages: list, total_cost: float):
        self.reward = reward
        self.info = info
        self.messages = messages
        self.total_cost = total_cost

async def ask_white_agent_to_solve(white_agent_url: str, question: str, context: str = "", max_iterations: int = 30) -> FinanceAgentResult:
    print(f"@@@ Green agent: Communicating with white agent at {white_agent_url}")
    initial_message = f"{context}\n\nQuestion: {question}" if context else question
    context_id = None
    final_answer = None
    history = []

    for i in range(max_iterations):
        msg = initial_message if i == 0 else question
        try:
            resp = await my_a2a.send_message(white_agent_url, msg, context_id=context_id)
            if not isinstance(resp.root, SendMessageSuccessResponse): break
            res_result = resp.root.result
            if not isinstance(res_result, Message): break
            
            context_id = res_result.context_id
            text_parts = get_text_parts(res_result.parts)
            if not text_parts: break
            
            white_text = "\n".join(text_parts)
            history.append({"iteration": i+1, "response": white_text})
            final_answer = white_text
            break 
        except Exception as e:
            print(f"Error communicating with white agent: {e}")
            break

    return FinanceAgentResult(1.0 if final_answer else 0.0, {"history": history}, history, 0.0)


class FinanceAgentExecutor(AgentExecutor):
    def __init__(self, max_num_steps: int = 30, mode: str = "direct"):
        self.max_num_steps = max_num_steps
        self.mode = mode

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        user_input = context.get_user_input()
        tags = parse_tags(user_input)
        
        config = json.loads(tags.get("config", tags.get("env_config", "{}")))
        mode = config.get("mode", self.mode)
        
        question = tags.get("question") or config.get("question")
        if not question:
            if ("assess" in user_input.lower() or "task is to" in user_input.lower()):
                question = get_random_question(config.get("csv_path"))
            else:
                question = user_input.strip()

        print(f"@@@ Executing Task. Mode: {mode}")
        print(f"@@@ Question: {question}")

        metrics = {}
        start_time = datetime.now()

        # ---------------------------------------------------------
        # MODE 1: WHITE AGENT
        # ---------------------------------------------------------
        if mode == "white_agent" and tags.get("white_agent_url"):
            print(f"@@@ Mode: White Agent ({tags.get('white_agent_url')})")
            res = await ask_white_agent_to_solve(
                tags.get("white_agent_url"), 
                question, 
                config.get("context", ""), 
                config.get("max_iterations", self.max_num_steps)
            )
            final_answer = res.info.get("history", [{}])[-1].get("response", "No response")
            agent_metadata = res.info
            result_bool = res.reward == 1.0

        # ---------------------------------------------------------
        # MODE 2: DIRECT (Internal Agent)
        # ---------------------------------------------------------
        else:
            print("@@@ Mode: Direct (Running internal agent.py)")
            
            model_name = config.get("model", "anthropic/claude-3-7-sonnet-20250219")
            temperature = config.get("temperature", 0.0)
            llm_instance = LLM(model=model_name, temperature=temperature)
            
            tool_names = config.get("tools", ["google_web_search", "retrieve_information", "parse_html_page", "edgar_search"])
            tools_dict = {}
            for name in tool_names:
                try:
                    tools_dict[name] = get_tool_by_name(name) 
                except Exception as e:
                    print(f"Warning: Could not load tool '{name}': {e}")

            agent = Agent(
                tools=tools_dict,
                llm=llm_instance,
                max_turns=config.get("max_turns", self.max_num_steps)
            )

            final_answer, agent_metadata = await agent.run(question)
            
            result_bool = True if final_answer and "Max turns reached" not in final_answer else False

        # ---------------------------------------------------------
        # REPORTING
        # ---------------------------------------------------------
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        metrics.update({
            "mode": mode,
            "duration_seconds": duration,
            "final_answer": final_answer,
            "agent_metadata": agent_metadata,
            "success": result_bool
        })

        result_emoji = "✅" if result_bool else "❌"
        
        await event_queue.enqueue_event(
            new_agent_text_message(
                f"Finished. Agent success: {result_emoji}\n"
                f"Mode: {mode}\n"
                f"Answer: {final_answer}\n"
                f"Metrics: {json.dumps(metrics, indent=2)}\n"
            )
        )

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        pass

# -------------------------------------------------------------------------
# SERVER STARTUP
# -------------------------------------------------------------------------

def start_finance_agent(agent_name="finance_agent", host="localhost", port=9001, mode="direct", max_num_steps=30):
    print(f"Starting finance agent server in {mode} mode...")
    
    agent_card_dict = load_agent_card_toml(agent_name)
    agent_url = os.environ.get("AGENT_URL") or f"http://{host}:{port}"
    agent_card_dict["url"] = agent_url
    
    request_handler = DefaultRequestHandler(
        agent_executor=FinanceAgentExecutor(max_num_steps=max_num_steps, mode=mode),
        task_store=InMemoryTaskStore(),
    )

    app = A2AStarletteApplication(
        agent_card=AgentCard(**agent_card_dict),
        http_handler=request_handler,
    )
    
    print(f"Server running at {agent_url}")
    uvicorn.run(app.build(), host=host, port=port)

if __name__ == "__main__":
    start_finance_agent()