import random
from schemas import ClassifierResult, UserPrompt

class BaselineToxicClassifier:
    def check(self, user_prompt: UserPrompt) -> ClassifierResult:
        toxic = random.randint(0, 1)
        return ClassifierResult(toxic=toxic)