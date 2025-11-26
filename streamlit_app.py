import streamlit as st
import pandas as pd
import numpy as np
import joblib
from app.Utils.features import extract_features_from_window
import plotly.graph_objects as go
import plotly.express as px

# ==========================
# Load Model
# ==========================
MODEL_PATH = "app/model/cme_model.joblib"
model = joblib.load(MODEL_PATH)
THRESHOLD = 0.45

# ==========================
# Custom CSS for Enhanced UI
# ==========================
st.markdown("""
<style>
    /* Main background gradient */
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
        border-right: 2px solid #4a5568;
    }
    
    /* Headers with glow effect */
    h1, h2, h3 {
        color: #ffffff !important;
        text-shadow: 0 0 20px rgba(100, 181, 246, 0.6);
    }
    
    /* Metric cards */
    [data-testid="stMetricValue"] {
        font-size: 2.5rem !important;
        font-weight: bold !important;
        color: #64b5f6 !important;
    }
    
    /* Info boxes */
    .stAlert {
        border-radius: 10px;
        border-left: 5px solid #64b5f6;
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.5rem 2rem;
        font-weight: 600;
        transition: transform 0.2s;
    }
    
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }
    
    /* Data frames */
    .dataframe {
        border-radius: 10px;
        overflow: hidden;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: 8px;
        padding: 10px 20px;
        color: white;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
</style>
""", unsafe_allow_html=True)

# ==========================
# Streamlit Page Setup
# ==========================
st.set_page_config(
    page_title="Halo CME Detection System",
    page_icon="🌞",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================
# Sidebar Navigation
# ==========================
st.sidebar.markdown("# 🌌 CME Detection")
st.sidebar.markdown("### AI-Powered Space Weather")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate to:",
    ("🔮 Prediction Interface", "🧠 How It Works", "📊 About the Model"),
    label_visibility="collapsed"
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style='text-align: center; padding: 20px;'>
    <h4 style='color: #64b5f6; margin-bottom: 10px;'>👨‍💻 Developers</h4>
    <p style='color: #ffffff; font-weight: 600;'>Arnav Joshi | Pulkit Garg</p>
    <p style='color: #a0aec0; font-size: 0.9rem;'>3rd Year | Thapar University</p>
    <div style='margin-top: 15px; padding: 10px; background: rgba(100, 181, 246, 0.1); border-radius: 8px;'>
        <p style='color: #64b5f6; font-size: 0.85rem; margin: 0;'>🚀 Aditya-L1 SWIS Data</p>
    </div>
</div>
""", unsafe_allow_html=True)

# =====================================================
# 🔮 PREDICTION PAGE
# =====================================================
if page == "🔮 Prediction Interface":
    # Header
    st.markdown("""
    <div style='text-align: center; padding: 20px 0;'>
        <h1 style='font-size: 3rem; margin-bottom: 10px;'>🌌 Halo CME Detection System</h1>
        <p style='font-size: 1.2rem; color: #a0aec0;'>Real-time Solar Wind Plasma Analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Info cards
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("📡 **Data Source**\n\nAditya-L1 SWIS L2")
    with col2:
        st.info("🎯 **Resolution**\n\n5-minute intervals")
    with col3:
        st.info("🔬 **Method**\n\nPhysics-informed ML")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # File uploader with enhanced UI
    st.markdown("### 📁 Upload Solar Wind Data")
    uploaded_file = st.file_uploader(
        "Select your CSV file containing plasma parameters",
        type=["csv"],
        help="File should contain: proton_density, proton_speed, proton_temperature, alpha_density"
    )

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            
            # Create tabs for better organization
            tab1, tab2, tab3 = st.tabs(["📊 Data Preview", "🔮 Prediction Results", "📈 Feature Analysis"])
            
            with tab1:
                st.markdown("#### Uploaded Dataset")
                st.dataframe(df.head(20), use_container_width=True, height=400)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total Records", len(df))
                with col2:
                    st.metric("Columns", len(df.columns))

            required_cols = {"proton_density", "proton_speed", "proton_temperature", "alpha_density"}
            if not required_cols.issubset(df.columns):
                st.error(f"❌ Missing required columns: {required_cols - set(df.columns)}")
            else:
                features_df = extract_features_from_window(df)

                if features_df.isnull().values.any():
                    st.error("❌ Not enough valid data available for feature computation (~15 min needed).")
                else:
                    prob = model.predict_proba(features_df)[0][1]
                    prediction = "CME" if prob >= THRESHOLD else "Non-CME"
                    
                    with tab2:
                        st.markdown("### 🎯 Detection Results")
                        
                        # Create gauge chart for confidence
                        fig = go.Figure(go.Indicator(
                            mode = "gauge+number+delta",
                            value = prob * 100,
                            domain = {'x': [0, 1], 'y': [0, 1]},
                            title = {'text': "CME Confidence Score", 'font': {'size': 24, 'color': 'white'}},
                            delta = {'reference': THRESHOLD * 100, 'increasing': {'color': "red"}},
                            gauge = {
                                'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "white"},
                                'bar': {'color': "red" if prob >= THRESHOLD else "green"},
                                'bgcolor': "rgba(0,0,0,0)",
                                'borderwidth': 2,
                                'bordercolor': "white",
                                'steps': [
                                    {'range': [0, THRESHOLD*100], 'color': 'rgba(0, 255, 0, 0.2)'},
                                    {'range': [THRESHOLD*100, 100], 'color': 'rgba(255, 0, 0, 0.2)'}
                                ],
                                'threshold': {
                                    'line': {'color': "yellow", 'width': 4},
                                    'thickness': 0.75,
                                    'value': THRESHOLD * 100
                                }
                            }
                        ))
                        
                        fig.update_layout(
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)",
                            font={'color': "white", 'family': "Arial"},
                            height=400
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Result display
                        st.markdown("<br>", unsafe_allow_html=True)
                        result_col1, result_col2 = st.columns(2)
                        
                        with result_col1:
                            if prediction == "CME":
                                st.error(f"### ⚠️ {prediction} DETECTED")
                                st.markdown("**Recommendation:** Space weather alert protocols should be activated.")
                            else:
                                st.success(f"### ✅ {prediction}")
                                st.markdown("**Status:** Normal solar wind conditions detected.")
                        
                        with result_col2:
                            st.metric("Probability Score", f"{prob*100:.2f}%")
                            st.metric("Decision Threshold", f"{THRESHOLD*100:.0f}%")
                    
                    with tab3:
                        st.markdown("### 🔬 Physics-Informed Features")
                        
                        # Display features in a nice format
                        st.dataframe(
                            features_df.style.background_gradient(cmap='RdYlGn_r', axis=1),
                            use_container_width=True
                        )
                        
                        # Create feature importance visualization
                        st.markdown("#### Feature Values Visualization")
                        
                        feature_names = features_df.columns.tolist()
                        feature_values = features_df.iloc[0].tolist()
                        
                        fig_features = go.Figure(data=[
                            go.Bar(
                                x=feature_names,
                                y=feature_values,
                                marker=dict(
                                    color=feature_values,
                                    colorscale='Viridis',
                                    showscale=True
                                ),
                                text=[f'{val:.4f}' for val in feature_values],
                                textposition='auto',
                            )
                        ])
                        
                        fig_features.update_layout(
                            title="Extracted Feature Values",
                            xaxis_title="Feature Name",
                            yaxis_title="Value",
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(30,30,30,0.5)",
                            font={'color': "white"},
                            height=400
                        )
                        
                        st.plotly_chart(fig_features, use_container_width=True)
                        
                        # Feature descriptions
                        with st.expander("📖 Feature Descriptions"):
                            st.markdown("""
                            | Feature | Scientific Significance |
                            |---------|------------------------|
                            | **Alpha-Proton Ratio** | Higher ratios indicate CME ejecta composition |
                            | **Speed Variability** | Increased turbulence during CME passage |
                            | **Alpha/VpStd Index** | Identifies alpha-rich coherent plasma structures |
                            | **Alpha/Temperature Ratio** | Detects cool, dense CME material |
                            """)

        except Exception as e:
            st.error(f"⚠️ Error processing file: {str(e)}")
            st.info("Please ensure your CSV file contains the required columns with valid numerical data.")

    else:
        # Landing state with instructions
        st.markdown("""
        <div style='text-align: center; padding: 40px; background: rgba(100, 181, 246, 0.05); border-radius: 15px; margin: 20px 0;'>
            <h3 style='color: #64b5f6;'>👆 Upload a CSV file to begin analysis</h3>
            <p style='color: #a0aec0; margin-top: 15px;'>
                The system requires solar wind plasma parameters including:<br>
                <strong>proton_density, proton_speed, proton_temperature, alpha_density</strong>
            </p>
        </div>
        """, unsafe_allow_html=True)

# =====================================================
# 🧠 HOW IT WORKS PAGE
# =====================================================
elif page == "🧠 How It Works":
    st.markdown("""
    <div style='text-align: center; padding: 20px 0;'>
        <h1 style='font-size: 3rem;'>🧠 System Architecture & Methodology</h1>
    </div>
    """, unsafe_allow_html=True)
    
    # Interactive sections with tabs
    workflow_tab, physics_tab, performance_tab = st.tabs(["🔄 Workflow", "⚛️ Physics Features", "📈 Performance"])
    
    with workflow_tab:
        st.markdown("### 🔄 Detection Pipeline")
        
        # Flowchart visualization
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            <div style='background: rgba(74, 144, 226, 0.15); padding: 15px; border-radius: 10px; text-align: center; height: 240px; display: flex; flex-direction: column; justify-content: center; border: 1px solid rgba(74, 144, 226, 0.3);'>
                <h2 style='margin: 0 0 12px 0; font-size: 2.5rem;'>📡</h2>
                <h4 style='margin: 0 0 12px 0; color: #64b5f6; font-size: 1.1rem;'>Data Acquisition</h4>
                <p style='font-size: 0.8rem; margin: 0; line-height: 1.6; color: #c8d8e8;'>Aditya-L1 SWIS L2<br>Plasma Data<br>5-min resolution</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div style='background: rgba(100, 181, 246, 0.15); padding: 15px; border-radius: 10px; text-align: center; height: 240px; display: flex; flex-direction: column; justify-content: center; border: 1px solid rgba(100, 181, 246, 0.3);'>
                <h2 style='margin: 0 0 12px 0; font-size: 2.5rem;'>🏷️</h2>
                <h4 style='margin: 0 0 12px 0; color: #81d4fa; font-size: 1.1rem;'>Event Labeling</h4>
                <p style='font-size: 0.8rem; margin: 0; line-height: 1.6; color: #c8d8e8;'>CACTUS CME Catalog<br>Halo CME events<br>±1-2 day windows</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div style='background: rgba(129, 212, 250, 0.15); padding: 15px; border-radius: 10px; text-align: center; height: 240px; display: flex; flex-direction: column; justify-content: center; border: 1px solid rgba(129, 212, 250, 0.3);'>
                <h2 style='margin: 0 0 12px 0; font-size: 2.5rem;'>🔬</h2>
                <h4 style='margin: 0 0 12px 0; color: #4fc3f7; font-size: 1.1rem;'>Feature Engineering</h4>
                <p style='font-size: 0.8rem; margin: 0; line-height: 1.6; color: #c8d8e8;'>Physics-informed<br>4 key features<br>Rolling statistics</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown("""
            <div style='background: rgba(186, 218, 247, 0.15); padding: 15px; border-radius: 10px; text-align: center; height: 240px; display: flex; flex-direction: column; justify-content: center; border: 1px solid rgba(186, 218, 247, 0.3);'>
                <h2 style='margin: 0 0 12px 0; font-size: 2.5rem;'>🤖</h2>
                <h4 style='margin: 0 0 12px 0; color: #4dd0e1; font-size: 1.1rem;'>ML Classification</h4>
                <p style='font-size: 0.8rem; margin: 0; line-height: 1.6; color: #c8d8e8;'>Ensemble Model<br>CME/Non-CME<br>Soft voting</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br><br>", unsafe_allow_html=True)
        
        # Detailed steps
        with st.expander("📋 Detailed Pipeline Steps", expanded=True):
            st.markdown("""
            1. **Raw Data Collection**
               - **Aditya-L1 SWIS L2 Data**: Proton density (Np), speed (Vp), temperature (Tp), Alpha particle density (α)
               - **CACTUS CME Catalog**: Halo CME event timestamps and classifications
               - Continuous 5-minute cadence measurements
            
            2. **Event Window Extraction**
               - Halo CME timestamps from CACTUS catalog
               - Extract time windows: -1 to +2 days around events
               - Balanced dataset: 10 CME + 30 Non-CME windows
            
            3. **Feature Computation**
               - Rolling statistics (1-hour windows)
               - Ratio-based plasma composition markers
               - Turbulence and variability indicators
            
            4. **Model Inference**
               - Soft voting ensemble (RF + XGBoost + LogReg)
               - Probability-based classification
               - Threshold optimization for high recall
            """)
    
    with physics_tab:
        st.markdown("### ⚛️ Physics-Informed Features")
        
        # Feature cards with professional styling
        features_data = [
            {
                "name": "Alpha-Proton Ratio (α/Np)",
                "icon": "⚛️",
                "formula": "α / Np",
                "insight": "CME ejecta typically shows enhanced alpha particle abundance compared to normal solar wind. Higher ratios are strong CME indicators.",
            },
            {
                "name": "Speed Variability (VpStd)",
                "icon": "🌪️",
                "formula": "rolling_std(Vp, 1hr)",
                "insight": "CME-driven shocks create turbulence and speed fluctuations. Increased variability signals disturbed plasma conditions.",
            },
            {
                "name": "Alpha/VpStd Index",
                "icon": "🎯",
                "formula": "(α/Np) ÷ VpStd",
                "insight": "Combines composition and coherence. High values indicate alpha-rich, low-turbulence CME ejecta cores.",
            },
            {
                "name": "Alpha/Temperature Ratio",
                "icon": "🌡️",
                "formula": "α ÷ Tp",
                "insight": "CME material is often cooler and denser than ambient solar wind. This ratio identifies thermodynamic signatures.",
            }
        ]
        
        for i, feature in enumerate(features_data):
            with st.expander(f"{feature['icon']} {feature['name']}", expanded=(i==0)):
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.markdown(f"""
                    <div style='background: rgba(74, 144, 226, 0.15); padding: 25px; border-radius: 10px; text-align: center; border: 1px solid rgba(74, 144, 226, 0.3);'>
                        <h2 style='font-size: 3rem; margin: 0;'>{feature['icon']}</h2>
                        <code style='font-size: 1.1rem; color: #64b5f6; background: rgba(0,0,0,0.3); padding: 8px 12px; border-radius: 5px; display: inline-block; margin-top: 15px;'>{feature['formula']}</code>
                    </div>
                    """, unsafe_allow_html=True)
                with col2:
                    st.markdown(f"""
                    <div style='padding: 15px;'>
                        <h4 style='color: #64b5f6; margin-bottom: 10px;'>Scientific Insight:</h4>
                        <p style='line-height: 1.7; color: #e0e0e0;'>{feature['insight']}</p>
                    </div>
                    """, unsafe_allow_html=True)
    
    with performance_tab:
        st.markdown("### 📈 Model Performance Metrics")
        
        # Performance metrics
        metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
        
        metrics_col1.metric("Accuracy", "85%", delta="Balanced")
        metrics_col2.metric("Precision", "67% (CME)", delta="Conservative")
        metrics_col3.metric("Recall", "100% (CME)", delta="Perfect")
        metrics_col4.metric("F1-Score", "80% (CME)", delta="Optimized")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Model architecture
        st.markdown("#### 🤖 Ensemble Architecture")
        
        model_col1, model_col2, model_col3 = st.columns(3)
        
        with model_col1:
            st.markdown("""
            <div style='background: rgba(74, 144, 226, 0.1); padding: 20px; border-radius: 10px; border: 2px solid rgba(74, 144, 226, 0.4);'>
                <h4 style='color: #64b5f6; margin-bottom: 15px;'>🌲 Random Forest</h4>
                <p style='font-size: 0.9rem; line-height: 1.6;'>• Balanced class weights<br>• Robust to outliers<br>• Feature importance</p>
            </div>
            """, unsafe_allow_html=True)
        
        with model_col2:
            st.markdown("""
            <div style='background: rgba(100, 181, 246, 0.1); padding: 20px; border-radius: 10px; border: 2px solid rgba(100, 181, 246, 0.4);'>
                <h4 style='color: #81d4fa; margin-bottom: 15px;'>⚡ XGBoost</h4>
                <p style='font-size: 0.9rem; line-height: 1.6;'>• scale_pos_weight<br>• Gradient boosting<br>• Bias reduction</p>
            </div>
            """, unsafe_allow_html=True)
        
        with model_col3:
            st.markdown("""
            <div style='background: rgba(129, 212, 250, 0.1); padding: 20px; border-radius: 10px; border: 2px solid rgba(129, 212, 250, 0.4);'>
                <h4 style='color: #4fc3f7; margin-bottom: 15px;'>📊 Logistic Regression</h4>
                <p style='font-size: 0.9rem; line-height: 1.6;'>• Interpretable<br>• Linear boundary<br>• Probability calibration</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Innovation highlights
        st.markdown("### 💡 Key Innovations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.success("""
            **✅ Advantages**
            - No imaging required → real-time capability
            - Physics-informed → scientifically interpretable
            - Lightweight → suitable for spacecraft edge computing
            - Early warning → pre-impact detection possible
            """)
        
        with col2:
            st.info("""
            **🚀 Future Enhancements**
            - Shock region & sheath detection
            - Multi-year SWIS dataset expansion
            - Real-time onboard inference (ONNX)
            - Integration with forecasting models
            """)

# =====================================================
# 📊 ABOUT THE MODEL PAGE
# =====================================================
elif page == "📊 About the Model":
    st.markdown("""
    <div style='text-align: center; padding: 20px 0;'>
        <h1 style='font-size: 3rem;'>📊 Model Information</h1>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 🎯 Project Overview")
    
    st.markdown("""
    This system represents a novel approach to **Coronal Mass Ejection (CME) detection** using
    machine learning applied to in-situ plasma measurements rather than traditional coronagraph imaging.
    
    **Key Objective:** Detect Halo-type CMEs from solar wind plasma data collected by the Aditya-L1 
    spacecraft's Solar Wind Ion Spectrometer (SWIS) instrument, validated against the CACTUS CME catalog.
    """)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📚 Dataset Details")
        st.markdown("""
        **Primary Data Sources:**
        - **Aditya-L1 SWIS L2 Data**: Solar wind plasma parameters
        - **CACTUS CME Catalog**: Halo CME event ground truth
        
        **Data Specifications:**
        - **Time Resolution:** 5 minutes
        - **CME Events:** 10 Halo CMEs
        - **Non-CME Windows:** 30 quiet periods
        - **Window Size:** -1 to +2 days around events
        """)
        
        st.markdown("### 🎯 Model Parameters")
        st.markdown(f"""
        - **Algorithm:** Soft Voting Ensemble
        - **Components:** RF + XGBoost + LogReg
        - **Decision Threshold:** {THRESHOLD}
        - **Optimization Goal:** Maximize recall (minimize false negatives)
        - **Model File:** `{MODEL_PATH}`
        """)
    
    with col2:
        st.markdown("### 🔬 Scientific Context")
        st.markdown("""
        **What are Halo CMEs?**
        
        Coronal Mass Ejections that appear to surround the Sun in coronagraph images,
        indicating they're directed toward or away from Earth. These are the most
        geo-effective solar events, capable of:
        
        - Disrupting satellite operations
        - Affecting GPS navigation
        - Inducing geomagnetic storms
        - Impacting power grid infrastructure
        
        **Why This Matters:**
        
        Traditional detection relies on imaging instruments (LASCO, CACTUS), which have
        delays and require ground processing. This ML system enables **real-time onboard
        detection** using only plasma measurements, providing earlier warnings for space
        weather events.
        """)
    
    st.markdown("---")
    
    st.markdown("### 🏆 Achievements & Validation")
    
    achievement_col1, achievement_col2, achievement_col3 = st.columns(3)
    
    with achievement_col1:
        st.markdown("""
        <div style='background: rgba(76, 175, 80, 0.15); padding: 25px; border-radius: 10px; text-align: center; border: 1px solid rgba(76, 175, 80, 0.3);'>
            <h2 style='font-size: 2.5rem; margin: 0; color: #81c784;'>0</h2>
            <p style='margin: 10px 0 5px 0; color: #a5d6a7; font-weight: 600;'>False Negatives</p>
            <small style='color: #c8e6c9;'>Critical for safety</small>
        </div>
        """, unsafe_allow_html=True)
    
    with achievement_col2:
        st.markdown("""
        <div style='background: rgba(33, 150, 243, 0.15); padding: 25px; border-radius: 10px; text-align: center; border: 1px solid rgba(33, 150, 243, 0.3);'>
            <h2 style='font-size: 2.5rem; margin: 0; color: #64b5f6;'>4</h2>
            <p style='margin: 10px 0 5px 0; color: #90caf9; font-weight: 600;'>Key Features</p>
            <small style='color: #bbdefb;'>Physics-informed</small>
        </div>
        """, unsafe_allow_html=True)
    
    with achievement_col3:
        st.markdown("""
        <div style='background: rgba(255, 152, 0, 0.15); padding: 25px; border-radius: 10px; text-align: center; border: 1px solid rgba(255, 152, 0, 0.3);'>
            <h2 style='font-size: 2.5rem; margin: 0; color: #ffb74d;'>85%</h2>
            <p style='margin: 10px 0 5px 0; color: #ffcc80; font-weight: 600;'>Accuracy</p>
            <small style='color: #ffe0b2;'>Cross-validated</small>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.success("""
    **🎓 Academic Contribution:** This work demonstrates that physics-informed machine learning
    can match or exceed traditional detection methods while enabling real-time, onboard processing
    capabilities crucial for future space weather forecasting systems.
    """)