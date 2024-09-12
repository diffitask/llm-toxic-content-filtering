from langchain_core.runnables import RunnableLambda
from langserve import add_routes

from src.alerts.log_alert import get_log_alert

available_alerts = {
    'log': get_log_alert(),
}

def configure_alerts(app):
    for alert_name, alert_cls in available_alerts.items():
        alert_runnable = RunnableLambda(alert_cls.alert, name=f'alert_{alert_name}')
        add_routes(
            app, 
            alert_runnable, 
            path=f"/{alert_name}", 
            disabled_endpoints=['playground']
        )
