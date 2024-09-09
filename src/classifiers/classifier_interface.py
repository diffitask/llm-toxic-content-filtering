from abc import ABC, abstractmethod
from schemas import ClassifierResult, UserPrompt
from pydantic import validate_arguments

class ClassifierInterface(ABC):
    """Abstract base class to define the classifier interface."""
    
    @validate_arguments
    @abstractmethod
    def classify(self, user_prompt: UserPrompt) -> ClassifierResult:
        """Takes a text input and returns a dict with predictions."""
        pass
