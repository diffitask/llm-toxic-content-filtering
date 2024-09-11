from langchain_core.runnables import RunnableLambda
from langserve import add_routes

from src.classifiers.tf_idf_baseline import get_tf_idf_baseline_classifier

available_classifiers = {
    'baseline': get_tf_idf_baseline_classifier(),
}

def configure_classifiers(app):
    for classifier_name, classifier in available_classifiers.items():
        classifier_runnable = RunnableLambda(classifier.classify, name='classfier_fn')
        add_routes(app, classifier_runnable, path=f"/{classifier_name}")