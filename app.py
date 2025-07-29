import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Page configuration
st.set_page_config(
    page_title="AI Salary Predictor",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for dark theme styling
st.markdown("""
<style>
    /* Dark theme styling */
    .stApp {
        background-color: #1a1a1a !important;
        color: #ffffff !important;
    }
    
    /* Main content background */
    .main .block-container {
        background-color: #1a1a1a !important;
        padding-top: 2rem;
        color: #ffffff !important;
    }
    
    .main-header {
        font-size: 3rem;
        color: #00d4ff !important;
        text-align: center;
        font-weight: bold;
        margin-bottom: 0.5rem;
        text-shadow: 0 0 20px rgba(0, 212, 255, 0.5);
    }
    
    .sub-header {
        text-align: center;
        color: #b0b0b0 !important;
        font-size: 1.2rem;
        margin-bottom: 2rem;
        font-weight: 500;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #2a2a2a 0%, #3a3a3a 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: #00d4ff !important;
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
        border: 1px solid #444;
    }
    
    .metric-card h3 {
        color: #00d4ff !important;
        font-size: 2rem;
        margin-bottom: 0.5rem;
        font-weight: bold;
    }
    
    .metric-card p {
        color: #b0b0b0 !important;
        font-size: 0.9rem;
        margin: 0;
        font-weight: 500;
    }
    
    .feature-card {
        background: #2a2a2a !important;
        padding: 1.5rem;
        border-radius: 10px;
        border: 2px solid #00d4ff;
        margin: 1rem 0;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
        color: #ffffff !important;
    }
    
    .feature-card strong {
        color: #00d4ff !important;
        font-size: 1.2rem;
        font-weight: bold;
    }
    
    .feature-card h4 {
        color: #00d4ff !important;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    
    .feature-card li {
        color: #b0b0b0 !important;
        margin-bottom: 0.5rem;
        font-weight: 500;
    }
    
    .input-section {
        background: #2a2a2a !important;
        padding: 2rem;
        border-radius: 15px;
        border: 2px solid #00d4ff;
        margin: 1rem 0;
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.4);
    }
    
    .input-header {
        color: #00d4ff !important;
        font-size: 1.5rem;
        font-weight: bold;
        margin-bottom: 1rem;
        text-align: center;
        border-bottom: 2px solid #00d4ff;
        padding-bottom: 0.5rem;
    }
    
    .section-header {
        background: #333 !important;
        color: #00d4ff !important;
        padding: 0.8rem;
        border-radius: 8px;
        margin: 1rem 0;
        font-weight: bold;
        border-left: 4px solid #00d4ff;
    }
    
    .prediction-success {
        background: linear-gradient(135deg, #27ae60 0%, #2ecc71 100%) !important;
        padding: 2.5rem;
        border-radius: 15px;
        color: white !important;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.4);
        border: 2px solid #27ae60;
    }
    
    .prediction-success h2, .prediction-success h3, .prediction-success p {
        color: white !important;
        font-weight: bold;
    }
    
    .prediction-info {
        background: linear-gradient(135deg, #3498db 0%, #2980b9 100%) !important;
        padding: 2.5rem;
        border-radius: 15px;
        color: white !important;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.4);
        border: 2px solid #3498db;
    }
    
    .prediction-info h2, .prediction-info h3, .prediction-info p {
        color: white !important;
        font-weight: bold;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #00d4ff 0%, #0099cc 100%) !important;
        color: #1a1a1a !important;
        border: none;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-weight: bold;
        font-size: 1rem;
        transition: all 0.3s;
        box-shadow: 0 4px 8px rgba(0, 212, 255, 0.3);
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #0099cc 0%, #00d4ff 100%) !important;
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 212, 255, 0.5);
        color: #1a1a1a !important;
    }
    
    .stButton > button:focus {
        color: #1a1a1a !important;
        outline: 2px solid #00d4ff;
        background: linear-gradient(135deg, #00d4ff 0%, #0099cc 100%) !important;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
        background-color: #1a1a1a !important;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: #2a2a2a !important;
        color: #ffffff !important;
        border-radius: 8px;
        font-weight: 600;
        border: 2px solid #444;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #00d4ff !important;
        color: #1a1a1a !important;
        border: 2px solid #00d4ff !important;
    }
    
    /* Text color overrides */
    .stMarkdown, .stMarkdown p, .stMarkdown li, .stMarkdown span, 
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4, 
    .stText, p, span, div {
        color: #ffffff !important;
    }
    
    /* Metric widget styling */
    .css-1xarl3l, .css-1wivap2, .metric-container {
        color: #ffffff !important;
        background-color: #2a2a2a !important;
    }
    
    /* Dataframe styling */
    .stDataFrame, .stDataFrame td, .stDataFrame th {
        color: #ffffff !important;
        background-color: #2a2a2a !important;
    }
    
    /* Input widgets */
    .stSelectbox > div > div {
        background-color: #2a2a2a !important;
        color: #ffffff !important;
        border: 1px solid #444 !important;
    }
    
    .stSlider > div > div > div {
        color: #ffffff !important;
    }
    
    .stNumberInput > div > div > input {
        color: #ffffff !important;
        background-color: #2a2a2a !important;
        border: 1px solid #444 !important;
    }
    
    /* Progress bar */
    .stProgress > div > div > div {
        background-color: #00d4ff !important;
    }
    
    /* File uploader */
    .stFileUploader > div {
        background-color: #2a2a2a !important;
        color: #ffffff !important;
        border: 2px dashed #00d4ff !important;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background-color: #2a2a2a !important;
        color: #ffffff !important;
    }
    
    /* Override any remaining text */
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] span,
    [data-testid="stMarkdownContainer"] div {
        color: #ffffff !important;
    }
    
    /* Special styling for input labels */
    label {
        color: #00d4ff !important;
        font-weight: bold !important;
    }
</style>
""", unsafe_allow_html=True)

# Load the trained model
@st.cache_resource
def load_model():
    try:
        model_data = joblib.load("improved_salary_model.pkl")
        return model_data
    except FileNotFoundError:
        st.error("⚠️ Model file not found. Please train the model first by running 'improved_salary_model.py'")
        return None

model_data = load_model()

# Header section
st.markdown('<h1 class="main-header">🚀 AI-Powered Salary Predictor</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Predict employee salary classification using advanced machine learning | Internship Project 2025</p>', unsafe_allow_html=True)

# Info cards
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown("""
    <div class="metric-card">
        <h3>86.32%</h3>
        <p>Model Accuracy</p>
    </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown("""
    <div class="metric-card">
        <h3>48K+</h3>
        <p>Training Records</p>
    </div>
    """, unsafe_allow_html=True)
with col3:
    st.markdown("""
    <div class="metric-card">
        <h3>13</h3>
        <p>Input Features</p>
    </div>
    """, unsafe_allow_html=True)
with col4:
    st.markdown("""
    <div class="metric-card">
        <h3>ML</h3>
        <p>Ensemble Model</p>
    </div>
    """, unsafe_allow_html=True)

if model_data is None:
    st.stop()

# Create tabs for better organization
tab1, tab2 = st.tabs(["🔮 Single Prediction", "📁 Batch Processing"])

with tab1:
    # Center the input form
    input_col1, input_col2, input_col3 = st.columns([0.5, 2, 0.5])
    
    with input_col2:
        st.markdown('<div class="input-section">', unsafe_allow_html=True)
        st.markdown('<h3 class="input-header">📊 Employee Information</h3>', unsafe_allow_html=True)
        
        # Personal Information Section
        st.markdown('<div class="section-header">👤 Personal Information</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            age = st.slider("Age", 17, 75, 35, help="Employee's age")
            gender = st.selectbox("Gender", ["Male", "Female"], help="Gender")
        with col2:
            race = st.selectbox("Race", [
                "White", "Asian-Pac-Islander", "Amer-Indian-Eskimo", 
                "Other", "Black"
            ], help="Race/ethnicity")
            native_country = st.selectbox("Native Country", [
                "United-States", "Cambodia", "England", "Puerto-Rico", "Canada", 
                "Germany", "India", "Japan", "China", "Cuba", "Philippines", 
                "Italy", "Mexico", "Unknown"
            ], index=0, help="Country of origin")
        
        # Education & Work Section
        st.markdown('<div class="section-header">🎓 Education & Work</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            educational_num = st.selectbox("Education Level", [
                (9, "High School Graduate"),
                (10, "Some College"),
                (11, "Associates Degree"),
                (12, "Associates Academic"),
                (13, "Bachelors"),
                (14, "Masters"),
                (15, "Professional School"),
                (16, "Doctorate")
            ], format_func=lambda x: x[1], help="Highest education level achieved")[0]
            
            workclass = st.selectbox("Work Class", [
                "Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov", 
                "Local-gov", "State-gov", "Unknown"
            ], help="Type of employment")
        
        with col2:
            occupation = st.selectbox("Occupation", [
                "Tech-support", "Craft-repair", "Other-service", "Sales",
                "Exec-managerial", "Prof-specialty", "Handlers-cleaners", 
                "Machine-op-inspct", "Adm-clerical", "Farming-fishing", 
                "Transport-moving", "Priv-house-serv", "Protective-serv", 
                "Armed-Forces", "Unknown"
            ], help="Primary occupation")
            
            hours_per_week = st.slider("Hours per Week", 1, 99, 40, 
                                     help="Average hours worked per week")
        
        # Financial & Family Section
        st.markdown('<div class="section-header">💰 Financial & Family</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            marital_status = st.selectbox("Marital Status", [
                "Married-civ-spouse", "Divorced", "Never-married", 
                "Separated", "Widowed", "Married-spouse-absent", 
                "Married-AF-spouse"
            ], help="Current marital status")
            
            capital_gain = st.number_input("Capital Gain", 0, 100000, 0, step=1000, 
                                         help="Capital gains from investments")
        
        with col2:
            relationship = st.selectbox("Relationship", [
                "Wife", "Own-child", "Husband", "Not-in-family", 
                "Other-relative", "Unmarried"
            ], help="Relationship status within family")
            
            capital_loss = st.number_input("Capital Loss", 0, 5000, 0, step=100,
                                         help="Capital losses from investments")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Final weight (hidden from user, using average)
        fnlwgt = 189778  # Average value from dataset
        
        # Build input DataFrame matching the training data
        input_df = pd.DataFrame({
            'age': [age],
            'workclass': [workclass],
            'fnlwgt': [fnlwgt],
            'educational-num': [educational_num],
            'marital-status': [marital_status],
            'occupation': [occupation],
            'relationship': [relationship],
            'race': [race],
            'gender': [gender],
            'capital-gain': [capital_gain],
            'capital-loss': [capital_loss],
            'hours-per-week': [hours_per_week],
            'native-country': [native_country]
        })
        
        # Enhanced display area in the center
        st.markdown('<div style="background-color: #2a2a2a; padding: 1rem; border-radius: 8px; border: 2px solid #00d4ff; margin: 2rem 0;"><h3 style="color: #00d4ff !important; margin: 0;">📋 Input Summary</h3></div>', unsafe_allow_html=True)
        
        # Create a nice display of input data
        display_data = {
            "Personal": f"👤 {age} years old {gender} from {native_country}",
            "Education": f"🎓 Level {educational_num} ({[x[1] for x in [(9, 'High School Graduate'), (10, 'Some College'), (11, 'Associates Degree'), (12, 'Associates Academic'), (13, 'Bachelors'), (14, 'Masters'), (15, 'Professional School'), (16, 'Doctorate')] if x[0] == educational_num][0]})",
            "Employment": f"💼 {workclass} - {occupation}",
            "Work Schedule": f"⏰ {hours_per_week} hours/week",
            "Family": f"👫 {marital_status} - {relationship}",
            "Financial": f"💰 Gain: ${capital_gain:,} | Loss: ${capital_loss:,}"
        }
        
        for category, info in display_data.items():
            st.markdown(f"""
            <div class="feature-card">
                <strong>{category}</strong><br>
                {info}
            </div>
            """, unsafe_allow_html=True)
        
        # Prediction section with enhanced styling
        st.markdown("---")
        
        predict_col1, predict_col2, predict_col3 = st.columns([1, 2, 1])
        with predict_col2:
            if st.button("🔮 Predict Salary Class", type="primary", use_container_width=True):
                try:
                    # Create predictor object and load model
                    from improved_salary_model import SalaryPredictor
                    predictor = SalaryPredictor()
                    predictor.model = model_data['model']
                    predictor.label_encoders = model_data['label_encoders']
                    predictor.scaler = model_data['scaler']
                    predictor.feature_names = model_data['feature_names']
                    
                    # Make prediction
                    prediction = predictor.predict(input_df)
                    probabilities = predictor.predict_proba(input_df)
                    
                    # Display results with enhanced styling
                    if prediction[0] == '>50K':
                        st.markdown(f"""
                        <div class="prediction-success">
                            <h2>💰 HIGH EARNER</h2>
                            <h3>Predicted Salary: <strong>{prediction[0]}</strong></h3>
                            <p>This employee is predicted to earn more than $50,000 annually</p>
                        </div>
                        """, unsafe_allow_html=True)
                        st.balloons()
                    else:
                        st.markdown(f"""
                        <div class="prediction-info">
                            <h2>💼 STANDARD EARNER</h2>
                            <h3>Predicted Salary: <strong>{prediction[0]}</strong></h3>
                            <p>This employee is predicted to earn $50,000 or less annually</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    if probabilities is not None:
                        prob_high = probabilities[0][1] if probabilities[0][1] > probabilities[0][0] else probabilities[0][0]
                        
                        # Enhanced probability display
                        prob_col1, prob_col2, prob_col3 = st.columns(3)
                        with prob_col1:
                            st.metric("≤$50K Probability", f"{probabilities[0][0]:.1%}", 
                                    delta=f"{probabilities[0][0]-0.5:.1%}")
                        with prob_col2:
                            st.metric("Confidence Level", f"{prob_high:.1%}", 
                                    delta="High" if prob_high > 0.8 else "Medium")
                        with prob_col3:
                            st.metric(">$50K Probability", f"{probabilities[0][1]:.1%}", 
                                    delta=f"{probabilities[0][1]-0.5:.1%}")
                        
                        # Enhanced progress bar
                        confidence_color = "🟢" if prob_high > 0.8 else "🟡" if prob_high > 0.6 else "🔴"
                        st.progress(prob_high, text=f"{confidence_color} Prediction Confidence: {prob_high:.1%}")
                        
                except Exception as e:
                    st.error(f"❌ Error making prediction: {str(e)}")
                    st.info("Please make sure the improved model is trained and saved.")
        
        # Feature insights section
        st.markdown("---")
        st.markdown('<div style="background-color: #2a2a2a; padding: 1rem; border-radius: 8px; border: 2px solid #27ae60; margin: 1rem 0;"><h3 style="color: #00d4ff !important; margin: 0;">📈 Key Insights</h3></div>', unsafe_allow_html=True)
        
        insight_col1, insight_col2 = st.columns(2)
        
        with insight_col1:
            st.markdown("""
            <div class="feature-card">
                <h4>🎯 High Earning Factors</h4>
                <ul>
                    <li>🎓 Bachelor's degree or higher</li>
                    <li>👔 Executive/Professional roles</li>
                    <li>⏰ 40+ hours per week</li>
                    <li>💍 Married status</li>
                    <li>🎂 Age 35-55 years</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with insight_col2:
            st.markdown("""
            <div class="feature-card">
                <h4>📊 Model Performance</h4>
                <ul>
                    <li>🎯 86.32% Overall Accuracy</li>
                    <li>📊 Gradient Boosting Algorithm</li>
                    <li>🔢 13 Input Features</li>
                    <li>📚 48,000+ Training Records</li>
                    <li>🏆 Ensemble Learning</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

with tab2:
    # Enhanced batch prediction section
    st.markdown('<div style="background-color: #2a2a2a; padding: 1rem; border-radius: 8px; border: 2px solid #9b59b6; margin-bottom: 1rem;"><h3 style="color: #00d4ff !important; margin: 0;">📁 Batch Salary Prediction</h3><p style="color: #b0b0b0 !important; margin: 0.5rem 0 0 0;">Upload a CSV file with employee data for bulk predictions</p></div>', unsafe_allow_html=True)
    
    # File upload with better styling
    uploaded_file = st.file_uploader(
        "Choose a CSV file", 
        type="csv", 
        help="Upload a CSV file with the same columns as the training data"
    )
    
    if uploaded_file is not None:
        try:
            batch_data = pd.read_csv(uploaded_file)
            
            # Enhanced data preview
            st.markdown("#### 📊 Data Preview")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Records", len(batch_data))
            with col2:
                st.metric("Features", len(batch_data.columns))
            with col3:
                st.metric("File Size", f"{uploaded_file.size} bytes")
            
            st.dataframe(batch_data.head(10), use_container_width=True)
            
            # Prediction button with progress
            if st.button("🔄 Process Batch Predictions", type="primary", use_container_width=True):
                with st.spinner("🔄 Processing predictions..."):
                    # Create predictor object and load model
                    from improved_salary_model import SalaryPredictor
                    predictor = SalaryPredictor()
                    predictor.model = model_data['model']
                    predictor.label_encoders = model_data['label_encoders']
                    predictor.scaler = model_data['scaler']
                    predictor.feature_names = model_data['feature_names']
                    
                    # Make predictions
                    batch_preds = predictor.predict(batch_data)
                    batch_data['Predicted_Income'] = batch_preds
                    
                    # Enhanced results display
                    st.success("✅ Batch predictions completed successfully!")
                    
                    # Summary statistics
                    col1, col2, col3, col4 = st.columns(4)
                    high_earners = (batch_data['Predicted_Income'] == '>50K').sum()
                    low_earners = (batch_data['Predicted_Income'] == '<=50K').sum()
                    high_percentage = (high_earners / len(batch_data)) * 100
                    
                    with col1:
                        st.metric("High Earners (>50K)", high_earners, f"{high_percentage:.1f}%")
                    with col2:
                        st.metric("Standard Earners (≤50K)", low_earners, f"{100-high_percentage:.1f}%")
                    with col3:
                        st.metric("Total Processed", len(batch_data))
                    with col4:
                        st.metric("Success Rate", "100%", "✅")
                    
                    # Results table
                    st.markdown("#### 📋 Prediction Results")
                    st.dataframe(batch_data, use_container_width=True)
                    
                    # Download section
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        csv = batch_data.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="📥 Download Complete Results",
                            data=csv,
                            file_name='salary_predictions_complete.csv',
                            mime='text/csv',
                            use_container_width=True
                        )
                    
                    with col2:
                        # Summary only download
                        summary_data = batch_data[['Predicted_Income']].copy()
                        summary_data['Record_ID'] = range(1, len(summary_data) + 1)
                        summary_csv = summary_data.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="📋 Download Summary Only",
                            data=summary_csv,
                            file_name='salary_predictions_summary.csv',
                            mime='text/csv',
                            use_container_width=True
                        )
                    
        except Exception as e:
            st.error(f"❌ Error processing batch data: {str(e)}")
            st.info("Please ensure your CSV file has the correct format and column names.")

# Enhanced footer with dark theme
st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 2.5rem; background: linear-gradient(135deg, #2a2a2a 0%, #3a3a3a 100%); border-radius: 15px; color: #00d4ff; margin-top: 2rem; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3); border: 2px solid #00d4ff;'>
    <h3 style='color: #00d4ff !important; margin-bottom: 1rem; text-shadow: 0 0 10px rgba(0, 212, 255, 0.5);'>🎓 Machine Learning Internship Project 2025</h3>
    <p style='color: #ffffff !important; font-size: 1.1rem; margin-bottom: 0.5rem;'><strong>AI-Powered Salary Classification System</strong></p>
    <p style='color: #b0b0b0 !important; margin-bottom: 0.5rem;'>Built with ❤️ using Streamlit, Scikit-learn & Advanced ML Algorithms</p>
    <p style='color: #ffffff !important; font-weight: 600;'>🏆 Gradient Boosting Model | 86.32% Accuracy | Ensemble Learning</p>
    <p style='color: #888 !important; font-style: italic; margin-top: 1rem;'><em>Developed for Educational & Professional Portfolio Purposes</em></p>
</div>
""", unsafe_allow_html=True)

# Quick stats and information section
col1, col2 = st.columns(2)

with col1:
    with st.expander("📊 Model Performance Stats", expanded=False):
        st.markdown("""
        <div style='background: #2a2a2a; padding: 1rem; border-radius: 8px; border: 1px solid #00d4ff;'>
        <p style='color: #ffffff !important;'><strong>Model Performance:</strong></p>
        </div>
        """, unsafe_allow_html=True)
        st.progress(0.86, text="Accuracy: 86.3%")
        st.progress(0.88, text="Precision: 88%")
        st.progress(0.95, text="Recall: 95%")

with col2:
    with st.expander("🎯 Sample Test Cases", expanded=False):
        st.markdown("""
        <div style='background: #2a2a2a; padding: 1rem; border-radius: 8px; border: 1px solid #00d4ff;'>
        <p style='color: #ffffff !important;'><strong>High Earner Example:</strong></p>
        </div>
        """, unsafe_allow_html=True)
        st.code("Age: 45, Masters, Executive, 50h/week")
        st.markdown("**Standard Earner Example:**")
        st.code("Age: 25, High School, Service, 30h/week")
