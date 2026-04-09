# 🧠 Sentify Pro: Object-Oriented Sentiment Analysis System
https://husrocks-sentiment-analysis-app-nhufzq.streamlit.app/

## 📌 Project Overview
**Sentify Pro** is a Final Year Project designed to classify movie reviews as **Positive** or **Negative** using Machine Learning. 

[cite_start]Unlike simple scripts, this project uses a modular **Object-Oriented Programming (OOP)** architecture[cite: 27]. It features an interactive dashboard that allows users to swap between models in real-time (Polymorphism) and visualize performance using advanced metrics like ROC Curves and Feature Importance.

## 🚀 Key Features
* **OOP Architecture:** Uses Abstract Base Classes (`BaseMLModel`) to enforce structure.
* **Multi-Model Support:** switch between **Logistic Regression** and **Naive Bayes** instantly.
* **Advanced Analytics:**
    * **Interactive Confusion Matrix** (Plotly).
    * **ROC Curve** (Receiver Operating Characteristic) to measure sensitivity.
    * **Feature Importance** to explain *why* a review was classified as Positive/Negative.
* **Real-Time Interface:** Built with Streamlit for a professional user experience.

## 📂 Project Structure
[cite_start]The system is divided into 4 modular components[cite: 9]:

| File | Description |
| :--- | :--- |
| `data_loader.py` | **The Data Engine:** Handles loading CSVs, cleaning text, and vectorization (TF-IDF). |
| `models.py` | **The Logic:** Contains the `BaseMLModel` (Parent) and child classes (`LogRegModel`, `NaiveBayesModel`). |
| `evaluator.py` | **The Analytics:** Calculates accuracy and generates Plotly charts (ROC, Heatmaps). |
| `app.py` | **The Dashboard:** The main entry point that connects the user to the backend logic. |

## 🛠️ Installation & Setup

### 1. Clone the Repository
Download this folder to your local machine.

### 2. Install Dependencies
Open your terminal/command prompt in the project folder and run:
```bash
pip install -r requirements.txt
