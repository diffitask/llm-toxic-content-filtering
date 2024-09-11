from langchain_core.runnables import RunnableLambda
from langserve import add_routes

from src.classifiers.tf_idf_baseline import get_tf_idf_baseline_classifier
from src.schemas import ClassifierInput, ClassifierOutput
from src.logging import logger

available_classifiers = {
    'baseline': get_tf_idf_baseline_classifier(),
}


def alert(classifier_output_obj):
    classifier_output: ClassifierOutput = ClassifierOutput.parse_obj(classifier_output_obj)

    # logging the request
    # TODO: user_prompt_is_harmful = "Yes" if classifier_output.predicted_class else "No"
    user_prompt_is_harmful = "Yes" if (classifier_output.predicted_class == "vanilla_harmful") else "No"

    # configure logging: saving 1) initial user prompt, 2) filtering model answer
    logger.warning(f"Initial text: \"{classifier_output.initial_text}\" - "
                   f"Is the text harmful: \"{user_prompt_is_harmful}\"")

    # TODO: if classifier_output.predicted_class == 0:
    if classifier_output.predicted_class == "vanilla_harmful":
        # TODO: to send a notification to the Telegram here
        print('alert here', flush=True)

    return classifier_output_obj


def configure_classifiers(app):
    for classifier_name, classifier in available_classifiers.items():
        classifier_runnable = RunnableLambda(classifier.classify, name='classfier_fn')
        alert_runnable = RunnableLambda(alert, name='alert_fn')
        add_routes(app, classifier_runnable | alert_runnable, path=f"/{classifier_name}")
