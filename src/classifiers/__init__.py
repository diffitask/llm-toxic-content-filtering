from langserve import add_routes
from langchain_core.runnables import RunnableLambda
from schemas import UserPrompt, ClassifierResult

from classifiers.baseline import BaselineToxicClassifier

available_classifiers = {
    'baseline': BaselineToxicClassifier()
}

def configure_classifiers(app):
    for classifier_name, classifier in available_classifiers.items():
        runnable = RunnableLambda(classifier.check).with_types(input_type=UserPrompt, output_type=ClassifierResult)
        add_routes(app, runnable, path=f"/{classifier_name}")