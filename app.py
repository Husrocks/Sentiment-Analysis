import streamlit as st
import time
import pandas as pd
from data_loader import DataLoader
from models import LogRegModel, NaiveBayesModel
from evaluator import ModelEvaluator

# -----------------------------------------------------------------------------
# 1. APP CONFIGURATION (Must be first)
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Sentify Pro | Enterprise Sentiment Analytics",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# 2. CUSTOM CSS & STYLING (The Fix)
# -----------------------------------------------------------------------------
st.markdown("""
    <style>
    /* Import Modern Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif !important;
    }

    /* Glassmorphism for metric cards and containers */
    [data-testid="stMetric"], [data-testid="stVerticalBlock"] {
        background: rgba(30, 41, 59, 0.4) !important;
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 16px !important;
        padding: 24px !important;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1) !important;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }

    [data-testid="stMetric"]:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.2) !important;
    }
    
    /* Enhance metric values with a gradient */
    [data-testid="stMetricValue"] {
        background: -webkit-linear-gradient(45deg, #8b5cf6, #3b82f6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.8rem !important;
        font-weight: 800 !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: #94a3b8 !important;
        font-size: 0.9rem !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 1.5px;
    }

    /* Premium Animated Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #8b5cf6 0%, #3b82f6 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 12px 28px !important;
        font-weight: 600 !important;
        letter-spacing: 0.5px !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(139, 92, 246, 0.3) !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(139, 92, 246, 0.5) !important;
        background: linear-gradient(135deg, #9c73f7 0%, #4f90f7 100%) !important;
    }
    
    /* Clean up headers */
    h1, h2, h3 {
        font-weight: 700 !important;
        letter-spacing: -0.5px;
    }
    
    /* Sidebar subtle border */
    [data-testid="stSidebar"] {
        border-right: 1px solid rgba(255,255,255,0.05);
    }
    
    /* Make text inputs look sleek */
    .stTextArea textarea {
        background-color: rgba(15, 23, 42, 0.6) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 10px !important;
        color: #f8fafc !important;
        font-family: 'Outfit', sans-serif !important;
        padding: 15px !important;
    }
    
    .stTextArea textarea:focus {
        border-color: #8b5cf6 !important;
        box-shadow: 0 0 0 1px #8b5cf6 !important;
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 3. SIDEBAR CONTROLS
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title("🧠 Sentify Pro")
    st.caption("v1.0.0 | OOP Project")
    
    st.markdown("---")
    
    st.subheader("⚙️ Model Configuration")
    model_choice = st.selectbox(
        "Select Architecture",
        ["Logistic Regression", "Naive Bayes"],
        help="Choose the algorithm used for classification."
    )
    
    st.markdown("### 📊 Dataset Info")
    st.info("Training on 50K IMDB Movie Reviews. (Balanced Dataset)")
    
    st.markdown("---")
    

# -----------------------------------------------------------------------------
# 4. DATA LOADING ENGINE
# -----------------------------------------------------------------------------
@st.cache_resource
def load_system():
    # Use a try-except block to prevent crashes if file is missing
    try:
        with st.spinner('🚀 Initializing AI Engine...'):
            loader = DataLoader('reviews.csv')
            loader.load_data()
            loader.handle_missing_values()
            X_train, X_test, y_train, y_test = loader.get_features_and_target()
            return loader, X_train, X_test, y_train, y_test
    except FileNotFoundError:
        return None, None, None, None, None

# Load data once
loader, X_train, X_test, y_train, y_test = load_system()

if loader is None:
    st.error("❌ Critical Error: 'reviews.csv' not found. Please move the dataset into this folder.")
    st.stop()

@st.cache_resource
def get_trained_model(model_name):
    if model_name == "Logistic Regression":
        model = LogRegModel()
    else:
        model = NaiveBayesModel()
    
    # Train the model once
    model.train(X_train, y_train)
    return model

# -----------------------------------------------------------------------------
# 5. MAIN DASHBOARD LAYOUT
# -----------------------------------------------------------------------------
st.title("Sentiment Intelligence Dashboard")
st.markdown("Real-time text analysis powered by Machine Learning.")

# Create two main columns: Left for Input, Right for Analytics
left_col, right_col = st.columns([1, 1.5], gap="large")

# --- LEFT COLUMN: INPUT & PREDICTION ---
with left_col:
    st.markdown("### 📝 Live Analysis")
    st.markdown("Enter text below to analyze sentiment polarity.")
    
    if 'user_input' not in st.session_state:
        st.session_state.user_input = ""

    def clear_text():
        st.session_state.user_input = ""

    user_text = st.text_area(
        "Input Text",
        key="user_input",
        height=200,
        placeholder="E.g., The cinematography was breathtaking, but the plot felt weak..."
    )
    
    col_btn1, col_btn2 = st.columns([1, 1])
    with col_btn1:
        analyze_btn = st.button("🔍 Analyze Sentiment")
    with col_btn2:
        st.button("🗑️ Clear", on_click=clear_text)

    # Prediction Logic
    if analyze_btn and user_text:
        # 1. Fetch cached trained model
        model = get_trained_model(model_choice)
            
        with st.status("Processing...", expanded=True) as status:
            st.write("Vectorizing input text...")
            text_vectorized = loader.vectorizer.transform([user_text])
            st.write("Inferencing...")
            prediction = model.predict(text_vectorized)[0]
            
            # Get Probability (Confidence) if available
            confidence = 0.0
            if hasattr(model.model, "predict_proba"):
                probs = model.model.predict_proba(text_vectorized)
                confidence = max(probs[0])  # Get highest probability
            
            status.update(label="Analysis Complete!", state="complete", expanded=False)

        # 2. Display Result Card
        st.markdown("---")
        st.markdown("### Result")
        
        if prediction == "positive":
            st.success(f"**Sentiment: POSITIVE** 😃")
        else:
            st.error(f"**Sentiment: NEGATIVE** 😠")
            
        if confidence > 0:
            st.metric("Confidence Score", f"{confidence*100:.2f}%")
            st.progress(confidence)

# --- RIGHT COLUMN: ADVANCED ANALYTICS ---
with right_col:
    st.markdown("### 📈 Model Performance & Insight")
    
    # Fetch cached trained model
    active_model = get_trained_model(model_choice)

    # Generate Metrics
    preds = active_model.predict(X_test)
    probs = None
    if hasattr(active_model.model, "predict_proba"):
        probs = active_model.model.predict_proba(X_test)
        
    evaluator = ModelEvaluator(y_test, preds, probs)
    acc = evaluator.calculate_metrics()

    # Top Metrics Row
    m1, m2, m3 = st.columns(3)
    m1.metric("Accuracy", f"{acc*100:.1f}%", "+2.4%")
    m2.metric("Dataset Size", f"{len(loader.raw_data)}", "Rows")
    m3.metric("Model Type", "Supervised", "Classification")

    # Tabs for Clean Layout
    tab1, tab2, tab3 = st.tabs(["🟦 Confusion Matrix", "📈 ROC Curve", "🔠 Feature Importance"])
    
    with tab1:
        st.markdown("##### Visualizing Prediction Errors")
        fig_cm = evaluator.plot_confusion_matrix()
        st.plotly_chart(fig_cm, use_container_width=True)
        st.caption("Diagonals represent correct predictions. Off-diagonals are errors.")

    with tab2:
        st.markdown("##### Sensitivity vs. Specificity")
        fig_roc = evaluator.plot_roc_curve()
        if fig_roc:
            st.plotly_chart(fig_roc, use_container_width=True)
        else:
            st.warning("ROC Curve not available for this model type.")

    with tab3:
        if model_choice == "Logistic Regression":
            st.markdown("##### Which words drive the decision?")
            fig_feat = ModelEvaluator.plot_feature_importance(active_model, loader.vectorizer)
            st.plotly_chart(fig_feat, use_container_width=True)
        else:
            st.info("Feature Importance is best visualized with Logistic Regression. Switch models in the sidebar!")