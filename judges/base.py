from abc import ABC, abstractmethod
from openai import OpenAI

class BaseJudge(ABC):
    def __init__(self, llm_client: OpenAI = None):
        self.client = llm_client

    @abstractmethod
    def evaluate(self, question: dict, prediction: dict) -> dict:
        """
        Main evaluation method.
        Must return a dict with keys: 'score', 'reason', 'metadata'
        """
        pass

    def render(self, result: dict) -> str:
        """
        Default render method for human-readable reports.
        """
        return f"{self.__class__.__name__}: {result.get('score')}/100 - {result.get('reason')}"

    def _build_output(self, score: float, reasons: list[str] | str, metadata: dict = None) -> dict:
        """
        Helper to standardize judge output.
        """
        if isinstance(reasons, list):
            reason_text = "; ".join(reasons)
        else:
            reason_text = str(reasons)

        return {
            "score": float(score),
            "reason": reason_text,
            "metadata": metadata or {}
        }