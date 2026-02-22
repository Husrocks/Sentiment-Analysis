# test.py
import matplotlib.pyplot as plt
from data_loader import DataLoader
from models import LogRegModel
from evaluator import ModelEvaluator

# 1. Load Data
print("Loading data...")
loader = DataLoader('reviews.csv')
loader.load_data()
loader.handle_missing_values()
X_train, X_test, y_train, y_test = loader.get_features_and_target()

# 2. Train Model (Using Logistic Regression)
print("Training model...")
model = LogRegModel()
model.train(X_train, y_train)
predictions = model.predict(X_test)

# 3. Test Evaluator
print("Generating Report...")
evaluator = ModelEvaluator(y_test, predictions)

# Print Accuracy
print(evaluator.calculate_metrics())

# Generate Plot
fig = evaluator.plot_confusion_matrix()
print("Plot generated successfully!")

# Optional: Show the plot (a window will pop up)
plt.show()