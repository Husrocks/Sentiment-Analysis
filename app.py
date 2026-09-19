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
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# 2. CUSTOM CSS & RESPONSIVE STYLING
# -----------------------------------------------------------------------------
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Geist:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Geist', sans-serif !important;
    }

    /* Base Theme */
    .stApp {
        background-color: #000000 !important;
        color: #ededed !important;
    }

    /* Hide Streamlit Header */
    header[data-testid="stHeader"] {
        display: none !important;
    }
    
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 2rem !important;
        max-width: 1400px !important;
    }

    /* Custom HTML Dashboard CSS */
    .dashboard-header {
        margin-bottom: 2rem;
        padding-bottom: 1rem;
        border-bottom: 1px solid #333;
    }
    .dashboard-header h1 {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        background: linear-gradient(90deg, #fff, #888);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .dashboard-header p {
        color: #a1a1aa;
        margin-top: 0.5rem;
        font-size: 1rem;
    }

    /* Custom Metrics Grid (Responsive) */
    .custom-metrics-grid {
        display: grid;
        grid-template-columns: repeat(1, 1fr);
        gap: 1rem;
        margin-bottom: 2rem;
    }
    @media (min-width: 768px) {
        .custom-metrics-grid {
            grid-template-columns: repeat(3, 1fr);
        }
    }
    
    .custom-metric-card {
        background-color: #0a0a0a;
        border: 1px solid #27272a;
        border-radius: 12px;
        padding: 1.5rem;
        transition: all 0.2s ease;
    }
    .custom-metric-card:hover {
        border-color: #52525b;
        transform: translateY(-2px);
    }
    .custom-metric-title {
        color: #a1a1aa;
        font-size: 0.875rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.5rem;
    }
    .custom-metric-value {
        color: #ffffff;
        font-size: 2.25rem;
        font-weight: 700;
        line-height: 1;
    }

    /* Custom Input and Button Overrides */
    .stTextArea textarea {
        background-color: #0a0a0a !important;
        border: 1px solid #27272a !important;
        border-radius: 8px !important;
        color: #fff !important;
        padding: 1rem !important;
        font-size: 1rem !important;
    }
    .stTextArea textarea:focus {
        border-color: #ededed !important;
        box-shadow: 0 0 0 1px #ededed !important;
    }
    
    .stButton > button {
        background-color: #ededed !important;
        color: #000 !important;
        border: none !important;
        border-radius: 6px !important;
        padding: 0.5rem 1.5rem !important;
        font-weight: 600 !important;
        width: 100% !important;
    }
    .stButton > button:hover {
        background-color: #a1a1aa !important;
    }
    
    /* Result Banner */
    .result-banner {
        padding: 1.5rem;
        border-radius: 8px;
        margin-top: 1rem;
        display: flex;
        flex-direction: column;
        gap: 0.5rem;
        border-left: 4px solid;
    }
    .result-banner.positive {
        background-color: rgba(34, 197, 94, 0.1);
        border-color: #22c55e;
    }
    .result-banner.negative {
        background-color: rgba(239, 68, 68, 0.1);
        border-color: #ef4444;
    }
    .result-title {
        font-size: 1.25rem;
        font-weight: 600;
        color: #fff;
    }
    .result-conf {
        font-size: 0.875rem;
        color: #a1a1aa;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #050505 !important;
        border-right: 1px solid #27272a !important;
    }
    
    /* Desktop vs Mobile Column Overrides */
    @media (max-width: 992px) {
        div[data-testid="column"] {
            width: 100% !important;
            flex: 1 1 100% !important;
            min-width: 100% !important;
        }
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 3. SIDEBAR CONTROLS
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚡ Sentify")
    st.caption("Engineered for Scale")
    st.markdown("---")
    
    model_choice = st.selectbox(
        "AI Architecture",
        ["Logistic Regression", "Naive Bayes"]
    )
    
    st.markdown("---")
    st.markdown("#### System Status")
    st.success("🟢 API Operational")
    st.info("📚 50,000 Records Loaded")

# -----------------------------------------------------------------------------
# 4. DATA LOADING ENGINE
# -----------------------------------------------------------------------------
@st.cache_resource
def load_system():
    try:
        with st.spinner('Waking up AI models...'):
            loader = DataLoader('reviews.csv')
            loader.load_data()
            loader.handle_missing_values()
            X_train, X_test, y_train, y_test = loader.get_features_and_target()
            return loader, X_train, X_test, y_train, y_test
    except FileNotFoundError:
        return None, None, None, None, None

loader, X_train, X_test, y_train, y_test = load_system()

if loader is None:
    st.error("Dataset 'reviews.csv' not found. Please upload it to the root directory.")
    st.stop()

@st.cache_resource
def get_trained_model(model_name):
    model = LogRegModel() if model_name == "Logistic Regression" else NaiveBayesModel()
    model.train(X_train, y_train)
    return model

# -----------------------------------------------------------------------------
# 5. DASHBOARD LAYOUT (Custom HTML injected)
# -----------------------------------------------------------------------------

st.markdown("""
<div class="dashboard-header">
    <h1>Sentiment Intelligence Center</h1>
    <p>Real-time natural language processing powered by Scikit-Learn.</p>
</div>
""", unsafe_allow_html=True)

# Fetch Model
active_model = get_trained_model(model_choice)
preds = active_model.predict(X_test)
probs = active_model.model.predict_proba(X_test) if hasattr(active_model.model, "predict_proba") else None
evaluator = ModelEvaluator(y_test, preds, probs)
acc = evaluator.calculate_metrics()

# Custom Responsive Metrics Grid
metrics_html = f"""
<div class="custom-metrics-grid">
    <div class="custom-metric-card">
        <div class="custom-metric-title">Model Accuracy</div>
        <div class="custom-metric-value">{acc*100:.1f}%</div>
    </div>
    <div class="custom-metric-card">
        <div class="custom-metric-title">Training Volume</div>
        <div class="custom-metric-value">{len(loader.raw_data):,}</div>
    </div>
    <div class="custom-metric-card">
        <div class="custom-metric-title">Active Engine</div>
        <div class="custom-metric-value">{"LR" if model_choice == "Logistic Regression" else "NB"}</div>
    </div>
</div>
"""
st.markdown(metrics_html, unsafe_allow_html=True)

# Layout Split (Using CSS media query to force stacking on tablet/mobile)
left_col, right_col = st.columns([1, 1], gap="large")

with left_col:
    st.markdown("### Inference Engine")
    
    if 'user_input' not in st.session_state:
        st.session_state.user_input = ""

    def clear_text():
        st.session_state.user_input = ""

    user_text = st.text_area(
        "Payload",
        key="user_input",
        height=150,
        placeholder="Enter unstructured text data here for immediate classification...",
        label_visibility="collapsed"
    )
    
    c1, c2 = st.columns(2)
    with c1:
        analyze_btn = st.button("Run Inference")
    with c2:
        st.button("Clear Buffer", on_click=clear_text)

    if analyze_btn and user_text:
        text_vectorized = loader.vectorizer.transform([user_text])
        prediction = active_model.predict(text_vectorized)[0]
        
        confidence = 0.0
        if hasattr(active_model.model, "predict_proba"):
            p = active_model.model.predict_proba(text_vectorized)[0]
            confidence = max(p) * 100
            
        if prediction == "positive":
            st.markdown(f"""
            <div class="result-banner positive">
                <div class="result-title">Positive Sentiment Detected</div>
                <div class="result-conf">Confidence Score: {confidence:.2f}%</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-banner negative">
                <div class="result-title">Negative Sentiment Detected</div>
                <div class="result-conf">Confidence Score: {confidence:.2f}%</div>
            </div>
            """, unsafe_allow_html=True)

with right_col:
    st.markdown("### Telemetry & Diagnostics")
    
    tab1, tab2, tab3 = st.tabs(["Matrix", "ROC", "Features"])
    
    with tab1:
        fig_cm = evaluator.plot_confusion_matrix()
        # Force background transparency in Plotly for new theme
        fig_cm.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#ededed')
        st.plotly_chart(fig_cm, use_container_width=True)

    with tab2:
        fig_roc = evaluator.plot_roc_curve()
        if fig_roc:
            fig_roc.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#ededed')
            st.plotly_chart(fig_roc, use_container_width=True)
        else:
            st.warning("ROC not available.")

    with tab3:
        if model_choice == "Logistic Regression":
            fig_feat = ModelEvaluator.plot_feature_importance(active_model, loader.vectorizer)
            fig_feat.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#ededed')
            st.plotly_chart(fig_feat, use_container_width=True)
        else:
            st.info("Feature Importance requires Logistic Regression.")