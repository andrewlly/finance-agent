from datetime import datetime

TOKEN_KEYS = [
    "in_tokens",
    "out_tokens",
    "reasoning_tokens",
    "cache_read_tokens",
    "cache_write_tokens",
    "total_input_tokens",
    "total_output_tokens",
]

COST_KEYS = [
    "input",
    "output",
    "reasoning",
    "cache_read",
    "cache_write",
    "total_input",
    "total_output",
    "total",
]

INSTRUCTIONS_PROMPT = """You are a professional Financial Agent. Today is DEC 17, 2025. 
You are given a question and you need to answer it using the tools provided.

GUIDELINES:
1. **Accuracy is paramount.** Do not hallucinate numbers. 
2. **Admit failure.** If the answer is not found in the tools/documents, state "I could not find the answer" and explain why. Do not make up a number.
3. **Be efficient.** Do not retrieve the same document multiple times.

FORMATTING REQUIREMENTS:
1. When you have the answer, respond with 'FINAL ANSWER:' followed by your answer text.
2. On a NEW LINE after the text, provide a confidence score (0-100) indicating how certain you are based on the evidence. Format: 'CONFIDENCE: <score>'
3. Finally, append your sources in a dictionary with the exact format below.

EXAMPLE OUTPUT:
FINAL ANSWER: 
The FY2024 revenue was $5.2 billion, an increase of 10% YoY.

CONFIDENCE: 95

{
    "sources": [
        {
            "url": "https://sec.gov/...",
            "name": "2024 10-K Filing"
        }
    ]
}

Question:
{question}
"""


def _merge_statistics(metadata: dict) -> dict:
    """
    Merge turn-level statistics into session-level statistics.

    Args:
        metadata (dict): The metadata with turn-level statistics

    Returns:
        dict: Updated metadata with merged statistics
    """
    # Aggregate statistics from all turns
    for turn in metadata["turns"]:
        metadata["total_cost"] += turn["total_cost"]
        for key in TOKEN_KEYS:
            metadata["total_tokens"][key] += turn["combined_metadata"].get(key, 0) or 0

        for key in TOKEN_KEYS:
            metadata["total_tokens_query"][key] += (
                turn["query_metadata"].get(key, 0) or 0
            )

        if "retrieval_metadata" in turn:
            rm = turn["retrieval_metadata"]
            for key in TOKEN_KEYS:
                metadata["total_tokens_retrieval"][key] += rm.get(key, 0) or 0

        metadata["error_count"] += len(turn["errors"])

        # Aggregate tool usage
        for tool_call in turn["tool_calls"]:
            tool_name = tool_call["tool_name"]
            if tool_name not in metadata["tool_usage"]:
                metadata["tool_usage"][tool_name] = 0
            metadata["tool_usage"][tool_name] += 1
            metadata["tool_calls_count"] += 1

    # Calculate total duration
    if metadata["start_time"] and metadata["end_time"]:
        start = datetime.fromisoformat(metadata["start_time"])
        end = datetime.fromisoformat(metadata["end_time"])
        metadata["total_duration_seconds"] = (end - start).total_seconds()

    return metadata
