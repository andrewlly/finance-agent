from abc import ABC, abstractmethod
from typing import Dict, Any

class BaseJudge(ABC):
    def __init__(self, llm_client=None):
        """
        Initialize the judge with an optional LLM client.
        """
        self.client = llm_client

    @abstractmethod
    def evaluate(self, question: Dict, prediction: Dict) -> Dict[str, Any]:
        """
        Abstract method that must be implemented by all judges.
        """
        pass