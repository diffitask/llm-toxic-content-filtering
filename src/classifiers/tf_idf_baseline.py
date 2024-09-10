import joblib
from src.classifiers.classifier_interface import ClassifierInterface
from src.schemas import ClassifierInput, ClassifierOutput

class TfidfLogRegClassifier(ClassifierInterface):
    """Classifier implementation using TF-IDF vectorizer and Logistic Regression."""
    
    def __init__(self, tfidf_path: str, logreg_path: str):
        """
        Initialize the classifier with paths to the pre-trained TF-IDF vectorizer 
        and Logistic Regression model.
        """
        self.tfidf_vectorizer = joblib.load(tfidf_path)
        self.logreg_model = joblib.load(logreg_path)
    
    def classify(self, input: ClassifierInput) -> ClassifierOutput:
        """
        Classify the input text using the TF-IDF vectorizer and Logistic Regression model.
        """
        # Transform the input text to a TF-IDF vector
        tfidf_vector = self.tfidf_vectorizer.transform([input['text']])
        
        # Predict class using the logistic regression model
        predicted_class = self.logreg_model.predict(tfidf_vector)[0]
        
        return ClassifierOutput(predicted_class=predicted_class)
    
def get_tf_idf_baseline_classifier():
    tfidf_path = "src/artifacts/ru_tfidf_vectorizer.joblib"
    logreg_path = "src/artifacts/ru_jailbreak_logreg.joblib"
    return TfidfLogRegClassifier(tfidf_path, logreg_path)