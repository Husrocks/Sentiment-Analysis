import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, auc

class ModelEvaluator:
    def __init__(self, y_true, y_pred, y_probs=None):
        self.y_true = y_true
        self.y_pred = y_pred
        self.y_probs = y_probs  
    def calculate_metrics(self):
        return accuracy_score(self.y_true, self.y_pred)

    def plot_confusion_matrix(self):
        cm = confusion_matrix(self.y_true, self.y_pred)
        fig = px.imshow(
            cm, text_auto=True, aspect="auto", color_continuous_scale='Blues',
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=['Negative', 'Positive'], y=['Negative', 'Positive'],
            template='plotly_dark'
        )
        fig.update_layout(title='Confusion Matrix', height=400, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        return fig

    def plot_roc_curve(self, pos_label='positive'):
        """Generates the ROC Curve."""
        if self.y_probs is None:
            return None
        
        # Check if model classes are available in the model object
        # Since evaluator only receives y_true and y_probs, we can't easily 
        # get classes_ without passing the model. However, we can use the 
        # unique values of y_true to try and figure it out.
        
        # A simple fallback: if pos_label is in y_true, we assume its probability
        # is at index 1 (or we can just fallback to index 1 if pos_label is string)
        # To be robust, let's assume y_probs has shape (n_samples, 2)
        # We will plot ROC using probability of the positive class.
        
        # Assuming Positive class is column 1 (standard for scikit-learn binary classifiers)
        fpr, tpr, thresholds = roc_curve(self.y_true, self.y_probs[:, 1], pos_label=pos_label)
        roc_auc = auc(fpr, tpr)

        fig = go.Figure()
        
        # Add the main curve
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'AUC = {roc_auc:.2f}', line=dict(color='darkorange', width=2)))
        
        # Add the random guessing line
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random Guess', line=dict(color='navy', width=2, dash='dash')))
        
        fig.update_layout(
            title='ROC Curve (Receiver Operating Characteristic)',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            height=400,
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        return fig

    @staticmethod
    def plot_feature_importance(model, vectorizer, n=10):
        """
        Extracts and plots the top positive and negative words.
        Only works for Linear Models (Logistic Regression).
        """
        # Check if model has coefficients (Naive Bayes uses different logic)
        if not hasattr(model.model, 'coef_'):
            return None
            
        # Get feature names (words) and their coefficients (weights)
        feature_names = vectorizer.get_feature_names_out()
        coefs = model.model.coef_[0]
        
        # Create a DataFrame
        df = pd.DataFrame({'word': feature_names, 'weight': coefs})
        
        # Sort by weight
        top_pos = df.nlargest(n, 'weight')
        top_neg = df.nsmallest(n, 'weight')
        
        # Combine them
        top_features = pd.concat([top_pos, top_neg])
        
        fig = px.bar(
            top_features, 
            x='weight', 
            y='word', 
            orientation='h', 
            title=f'Top {n} Most Important Words',
            color='weight',
            color_continuous_scale='RdBu',
            template='plotly_dark'
        )
        fig.update_layout(height=500, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        return fig