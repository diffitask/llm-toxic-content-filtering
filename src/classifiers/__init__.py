from langchain_core.runnables import RunnableLambda
from langserve import add_routes

from src.classifiers.tf_idf_baseline import get_tf_idf_baseline_classifier
from src.schemas import ClassifierInput, ClassifierOutput

available_classifiers = {
    'baseline': get_tf_idf_baseline_classifier(),
}

def alert(input):
    classifier_output = ClassifierOutput.parse_obj(input)
    if classifier_output.predicted_class == "vanilla_harmful":
        print('alert here', flush=True)
    return input


def configure_classifiers(app):
    for classifier_name, classifier in available_classifiers.items():
        classifier_runnable = RunnableLambda(classifier.classify)
        alert_runnable = RunnableLambda(alert)
        add_routes(app, classifier_runnable | alert_runnable, path=f"/{classifier_name}")