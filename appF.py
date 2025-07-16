import streamlit as st
import pandas as pd
import pickle
import io
from datetime import datetime
import numpy as np

# Configure page
st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stDataFrame {
        background-color: #f8f9fa;
        border-radius: 5px;
        padding: 10px;
    }
    .success-message {
        padding: 1rem;
        border-radius: 5px;
        background-color: #d4edda;
        color: #155724;
        border: 1px solid #c3e6cb;
    }
    .error-message {
        padding: 1rem;
        border-radius: 5px;
        background-color: #f8d7da;
        color: #721c24;
        border: 1px solid #f5c6cb;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar configuration
st.sidebar.markdown("## 🔍 Fraud Detection System")
st.sidebar.markdown("---")

# Model description
st.sidebar.markdown("### 📊 Model Information")
st.sidebar.markdown("""
**Model Type:** CatBoost Classifier  
**Purpose:** Detect fraudulent insurance claims  
**Accuracy:** High-performance gradient boosting model  
**Status:** Production Ready
""")

st.sidebar.markdown("### 📋 Instructions")
st.sidebar.markdown("""
1. **Upload Excel File**: Select your data file containing the required features
2. **Verify Data**: Ensure all required columns are present
3. **Get Predictions**: Model will analyze each row for fraud probability
4. **Download Results**: Export predictions as Excel file
""")

st.sidebar.markdown("### 📁 Required Features")
required_features = [
    'incident_severity',
    'insured_hobbies', 
    'incident_hour_of_the_day',
    'Pin_code',
    'insured_zip',
    'property_damage',
    'vehicle_claim',
    'incident_city'
]

for feature in required_features:
    st.sidebar.markdown(f"• `{feature}`")

st.sidebar.markdown("---")
st.sidebar.markdown("*Categorical features will be automatically converted to numerical format*")

st.sidebar.markdown("### 🔄 Data Processing")
st.sidebar.markdown("""
**Categorical Features**: Will be automatically encoded using factorize()
- `incident_severity`
- `Pin_code`
- `insured_hobbies`
- `property_damage`
- `incident_city`

**Numerical Features**: Used as-is
- `vehicle_claim`
- `incident_hour_of_the_day`
- `insured_zip`
""")

# Main content
st.markdown('<h1 class="main-header">🔍 Fraud Detection System</h1>', unsafe_allow_html=True)

# Function to load model
@st.cache_resource
def load_model():
    """Load the CatBoost model from pickle file"""
    try:
        with open('fraud_detection_catboost_model.pkl', 'rb') as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        st.error("❌ Model file 'fraud_detection_catboost_model.pkl' not found. Please ensure the file is in the same directory as this script.")
        return None
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None

# Function to validate data
def validate_data(df):
    """Validate if the uploaded data contains all required features"""
    missing_features = []
    for feature in required_features:
        if feature not in df.columns:
            missing_features.append(feature)
    
    if missing_features:
        return False, missing_features
    
    return True, None

# Function to preprocess data (convert categorical to numerical)
def preprocess_data(df):
    """Convert categorical features to numerical using factorize (same as training)"""
    processed_df = df.copy()
    
    # Define categorical features that need encoding
    categorical_features = [
        'incident_severity',
        'Pin_code', 
        'insured_hobbies',
        'property_damage',
        'incident_city'
    ]
    
    # Define numerical features that should remain as-is
    numerical_features = [
        'vehicle_claim',
        'incident_hour_of_the_day',
        'insured_zip'
    ]
    
    # Convert categorical features using factorize
    for feature in categorical_features:
        if feature in processed_df.columns:
            # Convert to string first to handle any data type issues
            processed_df[feature] = processed_df[feature].astype(str)
            # Use factorize to convert to numerical
            processed_df[feature], _ = pd.factorize(processed_df[feature])
    
    # Ensure numerical features are numeric
    for feature in numerical_features:
        if feature in processed_df.columns:
            processed_df[feature] = pd.to_numeric(processed_df[feature], errors='coerce')
    
    return processed_df

# Function to make predictions
def make_predictions(model, df):
    """Make fraud predictions on the dataframe"""
    try:
        # Select only the required features in the correct order
        feature_data = df[required_features]
        
        # Make predictions
        predictions = model.predict(feature_data)
        prediction_proba = model.predict_proba(feature_data)
        
        # Create results dataframe
        results_df = df.copy()
        results_df['is_fraudulent'] = predictions
        results_df['fraud_probability'] = prediction_proba[:, 1]  # Probability of fraud (class 1)
        results_df['risk_level'] = results_df['fraud_probability'].apply(
            lambda x: 'High' if x > 0.7 else 'Medium' if x > 0.3 else 'Low'
        )
        
        return results_df
    except Exception as e:
        st.error(f"❌ Error making predictions: {str(e)}")
        return None

# Function to convert dataframe to Excel
def to_excel(df):
    """Convert dataframe to Excel format for download"""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, sheet_name='Fraud_Predictions', index=False)
    return output.getvalue()

# Main application logic
def main():
    # Load model automatically
    model = load_model()
    
    if model is None:
        st.stop()
    
    st.success("✅ Model loaded successfully!")
    
    # File upload
    st.markdown("## 📁 Upload Your Data")
    uploaded_file = st.file_uploader(
        "Choose an Excel or CSV file",
        type=['xlsx', 'xls', 'csv'],
        help="Upload a file containing all required features (categorical data will be automatically converted)"
    )
    
    if uploaded_file is not None:
        try:
            # Read the file (Excel or CSV)
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            # Display basic info about the uploaded file
            st.markdown("### 📊 Data Overview")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Records", len(df))
            with col2:
                st.metric("Total Columns", len(df.columns))
            with col3:
                st.metric("File Size", f"{uploaded_file.size / 1024:.1f} KB")
            
            # Show first few rows of original data
            st.markdown("### 🔍 Original Data Preview")
            st.dataframe(df.head(), use_container_width=True)
            
            # Validate data
            is_valid, error_info = validate_data(df)
            
            if not is_valid:
                if isinstance(error_info, list):
                    st.error(f"❌ Missing required features: {', '.join(error_info)}")
                else:
                    st.error(f"❌ Data validation error: {error_info}")
                st.stop()
            
            st.success("✅ Data validation passed!")
            
            # Preprocess data (convert categorical to numerical)
            st.markdown("### 🔄 Data Processing")
            with st.spinner("Converting categorical data to numerical format..."):
                processed_df = preprocess_data(df)
            
            # Show processed data preview
            st.markdown("### 📋 Processed Data Preview")
            st.info("Categorical features have been converted to numerical format using factorize()")
            
            # Show comparison of original vs processed for categorical columns
            categorical_features = ['incident_severity', 'Pin_code', 'insured_hobbies', 'property_damage', 'incident_city']
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Original (Categorical):**")
                original_sample = df[categorical_features].head(3)
                st.dataframe(original_sample, use_container_width=True)
            
            with col2:
                st.markdown("**Processed (Numerical):**")
                processed_sample = processed_df[categorical_features].head(3)
                st.dataframe(processed_sample, use_container_width=True)
            
            st.success("✅ Data preprocessing completed!")
            
            # Make predictions button
            if st.button("🚀 Run Fraud Detection", type="primary"):
                with st.spinner("Analyzing data for fraudulent patterns..."):
                    results_df = make_predictions(model, processed_df)
                
                if results_df is not None:
                    # Add original categorical data back to results for better readability
                    for feature in categorical_features:
                        if feature in df.columns:
                            results_df[f'{feature}_original'] = df[feature]
                    
                    # Display results
                    st.markdown("### 📈 Prediction Results")
                    
                    # Summary statistics
                    fraud_count = (results_df['is_fraudulent'] == 1).sum()
                    total_count = len(results_df)
                    fraud_percentage = (fraud_count / total_count) * 100
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Total Analyzed", total_count)
                    with col2:
                        st.metric("Fraudulent Cases", fraud_count)
                    with col3:
                        st.metric("Fraud Rate", f"{fraud_percentage:.1f}%")
                    with col4:
                        avg_fraud_prob = results_df['fraud_probability'].mean()
                        st.metric("Avg Fraud Probability", f"{avg_fraud_prob:.3f}")
                    
                    # Risk distribution
                    st.markdown("### 🎯 Risk Distribution")
                    risk_dist = results_df['risk_level'].value_counts()
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("High Risk", risk_dist.get('High', 0), delta_color="inverse")
                    with col2:
                        st.metric("Medium Risk", risk_dist.get('Medium', 0), delta_color="off")
                    with col3:
                        st.metric("Low Risk", risk_dist.get('Low', 0), delta_color="normal")
                    
                    # Display results table
                    st.markdown("### 📋 Detailed Results")
                    
                    # Add filters for better user experience
                    col1, col2 = st.columns(2)
                    with col1:
                        filter_risk = st.selectbox(
                            "Filter by Risk Level",
                            options=['All', 'High', 'Medium', 'Low'],
                            index=0
                        )
                    with col2:
                        filter_fraud = st.selectbox(
                            "Filter by Fraud Status",
                            options=['All', 'Fraudulent', 'Non-Fraudulent'],
                            index=0
                        )
                    
                    # Apply filters
                    filtered_df = results_df.copy()
                    if filter_risk != 'All':
                        filtered_df = filtered_df[filtered_df['risk_level'] == filter_risk]
                    if filter_fraud == 'Fraudulent':
                        filtered_df = filtered_df[filtered_df['is_fraudulent'] == 1]
                    elif filter_fraud == 'Non-Fraudulent':
                        filtered_df = filtered_df[filtered_df['is_fraudulent'] == 0]
                    
                    # Display filtered results
                    st.dataframe(
                        filtered_df.style.format({
                            'fraud_probability': '{:.3f}'
                        }).background_gradient(subset=['fraud_probability'], cmap='RdYlBu_r'),
                        use_container_width=True
                    )
                    
                    # Download button
                    st.markdown("### 💾 Download Results")
                    excel_data = to_excel(results_df)
                    
                    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"fraud_detection_results_{current_time}.xlsx"
                    
                    st.download_button(
                        label="📥 Download Results as Excel",
                        data=excel_data,
                        file_name=filename,
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        type="primary"
                    )
                    
                    st.success(f"✅ Analysis complete! {fraud_count} potentially fraudulent cases detected out of {total_count} records.")
        
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
            st.info("Please ensure your file is a valid Excel/CSV file with all required features.")

if __name__ == "__main__":
    main()