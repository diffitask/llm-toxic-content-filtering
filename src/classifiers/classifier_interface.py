from abc import ABC, abstractmethod
from typing import Annotated
from src.schemas import ClassifierOutput

class ClassifierInterface(ABC):
    """Abstract base class to define the classifier interface."""

    @abstractmethod
    def classify(self, input) -> ClassifierOutput:
        """Takes a text input and returns a dict with predictions."""
        pass
