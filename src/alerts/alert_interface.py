from abc import ABC, abstractmethod
from src.schemas import ClassifierOutput

class AlertInterface(ABC):
    """Abstract base class to define the alert interface."""

    @abstractmethod
    def alert(self, classifier_output_dict: ClassifierOutput):
        """Takes a classifier output and implements alert mechanism"""
        pass
