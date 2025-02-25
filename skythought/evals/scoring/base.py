from abc import ABC, abstractmethod
from typing import Any, Dict, List


class Scorer(ABC):
    """
    Abstract base class for scorers.
    """

    @abstractmethod
    def score(self, row: dict) -> bool:
        pass

    def __call__(self, row: dict) -> bool:
        return {**row, "score": self.score(row)}


class BatchScorer(ABC):
    """
    Abstract base class for batch scorers.
    """

    @abstractmethod
    def score(self, batch: Dict[str, Any]) -> List[bool]:
        pass

    def __call__(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        return {**batch, "score": self.score(batch)}
