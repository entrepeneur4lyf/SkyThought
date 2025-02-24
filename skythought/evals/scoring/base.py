from abc import ABC, abstractmethod


class Scorer(ABC):
    """
    Abstract base class for scorers.
    """

    @abstractmethod
    def __call__(self, row: dict) -> bool:
        pass
