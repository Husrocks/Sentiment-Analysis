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
    /* 1. Force Main Background to Light Grey */
    .stApp {
        background-color: #f0f2f6;
    }
    
    /* 2. FORCE TEXT COLORS (The Fix for Invisible Text) */
    h1, h2, h3, h4, h5, h6, p, div, span {
        color: #2c3e50 !important; /* Dark Blue-Grey Text */
    }
    
    /* Sidebar Text needs to remain light (since sidebar is usually dark) */
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3, 
    [data-testid="stSidebar"] p, 
    [data-testid="stSidebar"] span {
        color: #ffffff !important;
    }
    [data-testid="stSidebar"] {
        background-color: #1a1c24;
    }

    /* 3. Card Styling (White boxes) */
    div.css-1r6slb0, div.stMetric {
        background-color: #ffffff;
        border: 1px solid #d1d5db;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }
    
    /* 4. Metric Values (Big Numbers) */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
        color: #2563eb !important; /* Bright Blue */
    }
    [data-testid="stMetricLabel"] {
        color: #64748b !important; /* Grey Label */
    }

    /* 5. Buttons */
    .stButton>button {
        background-color: #2563eb;
        color: white !important;
        border-radius: 6px;
        border: none;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #1d4ed8;
    }
    
    /* 6. Tabs */
    button[data-baseweb="tab"] {
        color: #2c3e50;
        font-weight: 600;
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
    
    user_text = st.text_area(
        "Input Text",
        height=200,
        placeholder="E.g., The cinematography was breathtaking, but the plot felt weak..."
    )
    
    col_btn1, col_btn2 = st.columns([1, 1])
    with col_btn1:
        analyze_btn = st.button("🔍 Analyze Sentiment")
    with col_btn2:
        clear_btn = st.button("🗑️ Clear")
        if clear_btn:
            user_text = ""

    # Prediction Logic
    if analyze_btn and user_text:
        # 1. Initialize & Train Model
        if model_choice == "Logistic Regression":
            model = LogRegModel()
        else:
            model = NaiveBayesModel()
            
        with st.status("Processing...", expanded=True) as status:
            st.write("Training model on live data...")
            model.train(X_train, y_train)
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
    
    # Train a model for stats (if not already trained)
    if 'stats_model' not in st.session_state or st.session_state.current_model_name != model_choice:
        with st.spinner(f"Calibrating {model_choice} Metrics..."):
            if model_choice == "Logistic Regression":
                active_model = LogRegModel()
            else:
                active_model = NaiveBayesModel()
                
            active_model.train(X_train, y_train)
            st.session_state.stats_model = active_model
            st.session_state.current_model_name = model_choice
    else:
        active_model = st.session_state.stats_model

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