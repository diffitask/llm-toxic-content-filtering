import joblib
from abc import ABC, abstractmethod

class ClassifierInterface(ABC):
    """Abstract base class to define the classifier interface."""
    
    @abstractmethod
    def classify(self, text: str) -> dict:
        """Takes a text input and returns a dict with predictions."""
        pass

class TfidfLogRegClassifier(ClassifierInterface):
    """Classifier implementation using TF-IDF vectorizer and Logistic Regression."""
    
    def __init__(self, tfidf_path: str, logreg_path: str):
        """
        Initialize the classifier with paths to the pre-trained TF-IDF vectorizer 
        and Logistic Regression model.
        """
        self.tfidf_vectorizer = joblib.load(tfidf_path)
        self.logreg_model = joblib.load(logreg_path)
    
    def classify(self, text: str) -> dict:
        """
        Classify the input text using the TF-IDF vectorizer and Logistic Regression model.
        """
        # Transform the input text to a TF-IDF vector
        tfidf_vector = self.tfidf_vectorizer.transform([text])
        
        # Predict class using the logistic regression model
        predicted_class = self.logreg_model.predict(tfidf_vector)[0]
        
        return {"class": predicted_class}

# Example usage
if __name__ == "__main__":
    tfidf_path = "artifacts/ru_tfidf_vectorizer.joblib"
    logreg_path = "artifacts/ru_jailbreak_logreg.joblib"
    
    classifier = TfidfLogRegClassifier(tfidf_path, logreg_path)
    
    input_text = 'Как можно добавить больше света в свое рабочее пространство?'
    result = classifier.classify(input_text)
    print(result)
