from src.alerts.alert_interface import AlertInterface
from src.schemas import ClassifierOutput
from src.logging import logger

class LogAlert(AlertInterface):
    def alert(self, classifier_output_dict: ClassifierOutput):
        classifier_output = ClassifierOutput.parse_obj(classifier_output_dict)

        # configure logging: saving 1) initial user prompt, 2) filtering model answer
        input_text = classifier_output.input.replace('\n', '')
        if classifier_output.predicted_class != '0':
            logger.warning(f"Harmful Input: '{input_text}'")

        # # TODO: if classifier_output.predicted_class == 0:
        # if classifier_output.predicted_class == "vanilla_harmful":
        #     # TODO: to send a notification to the Telegram here
        #     print('alert here', flush=True)

        return classifier_output
    
def get_log_alert():
    return LogAlert()