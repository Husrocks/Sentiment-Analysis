# models.py
from abc import ABC, abstractmethod
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# 1. The Abstract Base Class (The Parent)
# This fulfills the "Abstract Base Class" requirement 
class BaseMLModel(ABC):
    @abstractmethod
    def train(self, X, y):
        """Abstract method to train the model."""
        pass

    @abstractmethod
    def predict(self, X):
        """Abstract method to make predictions."""
        pass

# 2. Subclass 1: Logistic Regression (The First Child)
# This fulfills the "Subclasses" requirement 
class LogRegModel(BaseMLModel):
    def __init__(self):
        # We initialize the actual Scikit-Learn model here
        self.model = LogisticRegression(max_iter=1000)

    def train(self, X, y):
        print("Training Logistic Regression...")
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

# 3. Subclass 2: Naive Bayes (The Second Child)
# This fulfills the requirement for "at least two different ML models" 
class NaiveBayesModel(BaseMLModel):
    def __init__(self):
        self.model = MultinomialNB()

    def train(self, X, y):
        print("Training Naive Bayes...")
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)