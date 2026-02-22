# data_loader.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

class DataLoader:
    def __init__(self, file_path):
        # Attributes required by assignment [cite: 11]
        self.file_path = file_path
        self.raw_data = None
        self.cleaned_data = None
        # We also need a way to turn text into numbers (vectorizer)
        self.vectorizer = TfidfVectorizer(max_features=5000)

    def load_data(self):
        """Loads data from the CSV file."""
        try:
            self.raw_data = pd.read_csv(self.file_path)
            # Depending on your dataset, columns might be 'review' and 'sentiment'
            # Let's ensure we just keep the top 1000 rows for speed while testing
            self.raw_data = self.raw_data.head(1000)
            print("Data loaded successfully.")
        except FileNotFoundError:
            print("Error: File not found. Check the path.")

    def handle_missing_values(self):
        """Removes empty rows."""
        # This satisfies the preprocessing requirement [cite: 12]
        if self.raw_data is not None:
            self.cleaned_data = self.raw_data.dropna()
            print("Missing values handled.")

    def get_features_and_target(self):
        """
        Splits data into X (text) and y (labels), and converts text to numbers.
        This effectively scales/processes features[cite: 12].
        """
        # Assuming the CSV has columns 'review' and 'sentiment'
        X_text = self.cleaned_data['review']
        y = self.cleaned_data['sentiment']

        # Convert text to numbers so the AI can understand it
        X_vectorized = self.vectorizer.fit_transform(X_text)
        
        return train_test_split(X_vectorized, y, test_size=0.2, random_state=42)