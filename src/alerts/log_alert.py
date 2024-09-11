from src.alerts.alert_interface import AlertInterface
from src.schemas import ClassifierOutput

class LogAlert(AlertInterface):
    def alert(self, input: ClassifierOutput):
        classifier_output = ClassifierOutput.parse_obj(input)
        if classifier_output.predicted_class == "vanilla_harmful":
            print('HELLO', flush=True)
    
def get_log_alert():
    return LogAlert()