from langserve import add_routes
from langchain_core.runnables import RunnableLambda
from schemas import UserPrompt, ClassifierResult

from classifiers.tf_idf_baseline import get_tf_idf_baseline_classifier

available_classifiers = {
    'baseline': get_tf_idf_baseline_classifier(),
}

def configure_classifiers(app):
    for classifier_name, classifier in available_classifiers.items():
        runnable = RunnableLambda(classifier.classify).with_types(input_type=UserPrompt, output_type=ClassifierResult)
        add_routes(app, runnable, path=f"/{classifier_name}")