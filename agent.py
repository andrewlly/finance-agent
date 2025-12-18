import json
import os
import re
import traceback
import uuid
from abc import ABC
from datetime import datetime
from collections import defaultdict

from model_library.base import (
    LLM,
    ToolCall,
    ToolResult,
    QueryResult,
    InputItem,
    TextInput,
)
from model_library.exceptions import MaxContextWindowExceededError

from logger import get_logger
from tools import Tool
from utils import INSTRUCTIONS_PROMPT, _merge_statistics, TOKEN_KEYS, COST_KEYS

agent_logger = get_logger(__name__)


def dict_replace_none_with_zero(d: dict) -> dict:
    result = {}
    for k, v in d.items():
        if isinstance(v, dict):
            result[k] = dict_replace_none_with_zero(v)
        else:
            result[k] = 0 if v is None else v
    return result


class ModelException(Exception):
    """
    Raised on model errors - not retried by default
    """
    pass


class Agent(ABC):
    def __init__(
        self,
        tools: dict[str, Tool],
        llm: LLM,
        max_turns: int = 20,
        instructions_prompt: str = INSTRUCTIONS_PROMPT,
    ):
        self.tools = tools
        self.llm = llm
        self.max_turns = max_turns
        self.instructions_prompt = instructions_prompt

        # hijack llm logger
        self.llm.logger = agent_logger

    async def _find_final_answer(self, response_text: str) -> tuple[str, float]:
        """
        Search for 'FINAL ANSWER:', extract the text, the JSON sources, 
        and the Confidence Score.
        
        Returns:
            tuple: (full_answer_string, confidence_score)
        """
        final_answer_pattern = re.compile(r"FINAL ANSWER:", re.IGNORECASE)

        if isinstance(response_text, str) and final_answer_pattern.search(response_text):
            
            confidence_match = re.search(r"CONFIDENCE:\s*(\d+)", response_text, re.IGNORECASE)
            confidence = float(confidence_match.group(1)) if confidence_match else 0.0

            final_answer_match = re.search(
                r"FINAL ANSWER:(.*?)(?:CONFIDENCE:|\{\"sources\"|\Z)",
                response_text,
                re.DOTALL | re.IGNORECASE
            )
            
            sources_match = re.search(r"(\{\"sources\".*\})", response_text, re.DOTALL)

            answer_text = (
                final_answer_match.group(1).strip() if final_answer_match else ""
            )

            sources_text = sources_match.group(1) if sources_match else ""

            # Reconstruct the final string (Answer + Sources)
            final_answer = answer_text
            if sources_text:
                final_answer = f"{answer_text}\n\n{sources_text}"

            agent_logger.info(f"\033[1;32m[FINAL ANSWER]\033[0m {final_answer}")
            agent_logger.info(f"\033[1;32m[CONFIDENCE]\033[0m {confidence}")
            
            return final_answer, confidence

        return None, 0.0

    async def _process_tool_calls(
        self, tool_calls: list[ToolCall], data_storage: dict, turn_metadata: dict
    ):
        """
        Helper method to process tool calls, handling errors, validating arguments,
        and generating the results.
        """

        tool_results: list[ToolResult] = []
        tool_call_metadatas: list[dict] = []
        errors: list[str] = []

        for tool_call in tool_calls:
            tool_name = tool_call.name

            # unpacks tool call arguments
            arguments = tool_call.args
            tool_call_metadata = {
                "tool_name": tool_name,
                "arguments": arguments,
                "success": False,
                "error": None,
                "tool_output": None 
            }

            # Validate tool_name exists
            if tool_name not in self.tools:
                error_msg = f"Tool '{tool_name}' not found. Available tools: {list(self.tools.keys())}"

                tool_call_metadata["error"] = error_msg
                tool_call_metadatas.append(tool_call_metadata)
                turn_metadata["errors"].append(error_msg)

                tool_result = ToolResult(tool_call=tool_call, result=error_msg)
                tool_results.append(tool_result)
                continue

            # Validate tool arguments are JSON-parseable
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    error_msg = f"Tool call arguments were not valid json: {arguments}"

                    tool_call_metadata["error"] = error_msg
                    tool_call_metadatas.append(tool_call_metadata)
                    errors.append(error_msg)

                    tool_result = ToolResult(tool_call=tool_call, result=error_msg)
                    tool_results.append(tool_result)
                    continue

            if tool_name == "retrieve_information":
                # Requires LLM and Data Storage
                raw_tool_result = await self.tools[tool_name](
                    arguments, data_storage, self.llm
                )
                
                # Token Tracking for Retrieval
                if "usage" in raw_tool_result:
                    tool_token_usage = raw_tool_result["usage"]
                    turn_metadata["retrieval_metadata"] = {**tool_token_usage}
                    for key in TOKEN_KEYS:
                        turn_metadata["combined_metadata"][key] += (
                            tool_token_usage.get(key, 0) or 0
                        )
                    for key in COST_KEYS:
                        turn_metadata["combined_metadata"]["cost"][key] += (
                            tool_token_usage.get("cost", {}).get(key, 0) or 0
                        )
                    turn_metadata["total_cost"] += tool_token_usage["cost"]["total"]

            elif tool_name in ["parse_html_page", "parse_pdf"]:
                # Requires Data Storage
                raw_tool_result = await self.tools[tool_name](arguments, data_storage)
            
            else:
                raw_tool_result = await self.tools[tool_name](arguments)
            if raw_tool_result["success"]:
                tool_call_metadata["success"] = True
                
                try:
                    if tool_name in ["parse_html_page", "parse_pdf"]:
                        key = arguments.get("key")
                        if key and key in data_storage:
                            content = str(data_storage[key])
                            tool_call_metadata["tool_output"] = content[:10000]
                    
                    else:
                        content = str(raw_tool_result.get("result", ""))
                        tool_call_metadata["tool_output"] = content[:10000]
                except Exception as e:
                    agent_logger.warning(f"Failed to capture tool output: {e}")

            else:
                tool_call_metadata["error"] = raw_tool_result["result"]
                errors.append(raw_tool_result["result"])

            tool_result = ToolResult(
                tool_call=tool_call, result=raw_tool_result["result"]
            )
            tool_results.append(tool_result)
            tool_call_metadatas.append(tool_call_metadata)

        turn_metadata["tool_calls"].extend(tool_call_metadatas)

        return tool_results

    async def _process_turn(self, turn_count, data_storage):
        """
        Process a single turn. Returns (final_answer, metadata, should_continue)
        """
        agent_logger.info(f"\033[1;34m[TURN {turn_count}]\033[0m")

        tool_definitions = [tool.get_tool_definition() for tool in self.tools.values()]
        
        try:
            response: QueryResult = await self.llm.query(
                input=self.messages, tools=tool_definitions
            )
        except MaxContextWindowExceededError:
            raise
        except Exception as e:
            agent_logger.critical(f"Error: {e}")
            agent_logger.critical(f"Traceback: {traceback.format_exc()}")
            raise ModelException(e)

        self.messages = response.history

        response_text = response.output_text
        reasoning_text = response.reasoning
        tool_calls: list[ToolCall] = response.tool_calls

        agent_logger.info(f"\033[1;36m[TOOL CALLS]\033[0m {len(tool_calls)}: {[tc.name for tc in tool_calls]}")

        turn_metadata = {
            "tool_calls": [],
            "errors": [],
            "query_metadata": dict_replace_none_with_zero(response.metadata.model_dump()),
            "retrieval_metadata": defaultdict(int),
            "combined_metadata": dict_replace_none_with_zero(response.metadata.model_dump()),
            "total_cost": response.metadata.cost.total,
        }

        if reasoning_text:
            agent_logger.info(f"\033[1;33m[REASONING]\033[0m {reasoning_text}")

        if response_text:
            agent_logger.info(f"\033[1;33m[RESPONSE]\033[0m {response_text}")

        if tool_calls:
            tool_results = await self._process_tool_calls(
                tool_calls, data_storage, turn_metadata
            )
            self.messages.extend(tool_results)
            return None, turn_metadata, True

        else:
            final_answer, confidence = await self._find_final_answer(response_text)

            if final_answer:
                turn_metadata["confidence"] = confidence
                return final_answer, turn_metadata, False

        return None, turn_metadata, True

    async def run(self, question: str, session_id: str = None) -> tuple[str, dict]:
        """
        Run the agent loop.
        """
        session_id = session_id or str(uuid.uuid4())
        metadata = {
            "session_id": session_id,
            "model_key": self.llm._registry_key,
            "user_input": question,
            "start_time": datetime.now().isoformat(),
            "end_time": None,
            "total_duration_seconds": 0,
            "total_tokens": defaultdict(int),
            "total_tokens_retrieval": defaultdict(int),
            "total_tokens_query": defaultdict(int),
            "turns": [],
            "tool_usage": {},
            "tool_calls_count": 0,
            "api_calls_count": 0,
            "error_count": 0,
            "total_cost": 0,
            "final_confidence": 0.0 # New field
        }

        data_storage = {}

        # Prepare initial prompt 
        # (Assuming INSTRUCTIONS_PROMPT in utils.py handles the formatting instructions now)
        prompt_instruction = self.instructions_prompt.format(question=question)
        
        initial_message = TextInput(text=prompt_instruction)
        self.messages: list[InputItem] = [initial_message]

        agent_logger.info(f"\033[1;34m[USER]\033[0m {prompt_instruction}")

        turn_count = 0
        final_answer = None

        while turn_count < self.max_turns:
            turn_count += 1
            try:
                result, turn_metadata, should_continue = await self._process_turn(
                    turn_count, data_storage
                )

                metadata["turns"].append(turn_metadata)

                if not should_continue:
                    final_answer = result
                    # Capture final confidence if available
                    if "confidence" in turn_metadata:
                        metadata["final_confidence"] = turn_metadata["confidence"]
                    break

            except MaxContextWindowExceededError:
                agent_logger.warning("Max Context Window Exceeded. Pruning history...")
                self.messages.pop(1)
                while len(self.messages) > 1 and isinstance(self.messages[1], ToolResult):
                    self.messages.pop(1)
                should_continue = True

            except Exception as e:
                metadata["error_count"] += 1
                agent_logger.error(f"Error: {e}")
                error_message = TextInput(text=f"System Error: {e}. Please retry.")
                self.messages.append(error_message)
                should_continue = True

        metadata["end_time"] = datetime.now().isoformat()
        
        # Calculate Duration
        try:
            start_dt = datetime.fromisoformat(metadata["start_time"])
            end_dt = datetime.fromisoformat(metadata["end_time"])
            metadata["total_duration_seconds"] = (end_dt - start_dt).total_seconds()
        except:
            pass

        if final_answer:
            metadata["final_answer"] = final_answer

        metadata = _merge_statistics(metadata)

        # Save results to file
        os.makedirs("logs/trajectories", exist_ok=True)
        log_path = os.path.join("logs", "trajectories", f"{session_id}.json")
        with open(log_path, "w") as f:
            json.dump(metadata, f, indent=2)

        return (final_answer if final_answer else "Max turns reached."), metadata