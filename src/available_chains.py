from langchain_core.runnables import RunnableParallel, RunnablePick
from langchain_core.output_parsers import StrOutputParser
from langchain_core.language_models import BaseChatModel
from langserve import RemoteRunnable

def classifier_and_alert_chain(classifier, alert):
    assert isinstance(classifier, RemoteRunnable)
    assert isinstance(alert, RemoteRunnable)
    return classifier | alert

def check_only_user_input_chain(llm, classifier, alert):
    assert isinstance(llm, BaseChatModel)
    classifier_and_alert = classifier_and_alert_chain(classifier, alert)
    chain = RunnableParallel(
        classifier=classifier_and_alert,
        llm=llm | StrOutputParser()
    )
    return chain

def check_everything_chain(llm, classifier, alert):
    get_llm_output_chain = RunnablePick(keys='llm') | StrOutputParser()
    chain = check_only_user_input_chain(llm, classifier, alert) | RunnableParallel(
        input_classifier_result=RunnablePick(keys='classifier'),
        output=get_llm_output_chain,
        llm_classifier_result=get_llm_output_chain | classifier_and_alert_chain(classifier, alert)
    )
    return chain