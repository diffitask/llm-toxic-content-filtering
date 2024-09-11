from abc import ABC, abstractmethod
from typing import Annotated
from schemas import ClassifierInput, ClassifierOutput

class ClassifierInterface(ABC):
    """Abstract base class to define the classifier interface."""

    @abstractmethod
    def classify(self, input: ClassifierInput) -> ClassifierOutput:
        """Takes a text input and returns a dict with predictions."""
        pass
