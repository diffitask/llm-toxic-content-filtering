from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
from langserve import add_routes

from src.classifiers.mistral_classifier import get_mistral_classifier


available_classifiers = {
    'mistral': get_mistral_classifier(),
}

def configure_classifiers(app):
    for classifier_name, classifier in available_classifiers.items():
        classifier_runnable = RunnableLambda(classifier.classify, name=f'classifier_{classifier_name}')
        add_routes(
            app, 
            classifier_runnable, 
            path=f"/{classifier_name}",
            disabled_endpoints=['playground']
        )
