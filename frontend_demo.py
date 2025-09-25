# ================================================================================
# 🌐 FRONTEND DEMO - EPILEPSY DETECTION WEB APP
# ================================================================================

import streamlit as st
import requests
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import json
from datetime import datetime
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="Epilepsy Seizure Detection",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #1f77b4;
        font-size: 3rem;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Configuration
DEFAULT_API_URL = "http://localhost:5000"  # Change to your deployed URL

# Title and header
st.markdown('<h1 class="main-header">🧠 Epilepsy Seizure Detection System</h1>', unsafe_allow_html=True)
st.markdown("### AI-Powered EEG Analysis for Seizure Detection")

# Sidebar configuration
st.sidebar.header("⚙️ Configuration")
api_url = st.sidebar.text_input("API URL", value=DEFAULT_API_URL, help="Enter your deployed API URL")

# API health check
@st.cache_data(ttl=60)  # Cache for 1 minute
def check_api_health(url):
    try:
        response = requests.get(f"{url}/", timeout=5)
        return response.status_code == 200, response.json() if response.status_code == 200 else None
    except:
        return False, None

# Generate sample EEG data
def generate_sample_eeg():
    """Generate realistic EEG data for demo"""
    np.random.seed(42)
    time_points = np.linspace(0, 1, 178)
    
    # Create realistic EEG signal
    signal = (
        2 * np.sin(2 * np.pi * 10 * time_points) +  # 10 Hz alpha waves
        1.5 * np.sin(2 * np.pi * 4 * time_points) +  # 4 Hz theta waves
        0.5 * np.sin(2 * np.pi * 20 * time_points) +  # 20 Hz beta waves
        np.random.normal(0, 0.3, 178)  # Noise
    )
    
    return signal

def generate_seizure_like_eeg():
    """Generate seizure-like EEG data for demo"""
    np.random.seed(123)
    time_points = np.linspace(0, 1, 178)
    
    # Create seizure-like signal (high amplitude, sharp spikes)
    signal = (
        5 * np.sin(2 * np.pi * 15 * time_points) +  # High frequency
        3 * np.sin(2 * np.pi * 8 * time_points) +   # Medium frequency
        np.random.normal(0, 1.5, 178)  # Higher noise
    )
    
    # Add sharp spikes characteristic of seizures
    spike_indices = np.random.choice(178, 20, replace=False)
    signal[spike_indices] += np.random.normal(0, 5, 20)
    
    return signal

# Check API status
is_healthy, api_info = check_api_health(api_url)

if is_healthy:
    st.sidebar.success("✅ API is online")
    if api_info and 'available_models' in api_info:
        st.sidebar.info(f"Available models: {len(api_info['available_models'])}")
else:
    st.sidebar.error("❌ API is offline")
    st.sidebar.warning("Please check the API URL and ensure the server is running")

# Main interface
tab1, tab2, tab3 = st.tabs(["🧠 Single Prediction", "📊 Batch Analysis", "ℹ️ Model Info"])

with tab1:
    st.header("Single EEG Sample Analysis")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("EEG Signal Input")
        
        # Input method selection
        input_method = st.radio(
            "Choose input method:",
            ["📊 Use Demo Data", "📁 Upload File", "✏️ Manual Input"]
        )
        
        if input_method == "📊 Use Demo Data":
            demo_type = st.selectbox(
                "Select demo data type:",
                ["Normal EEG", "Seizure-like EEG"]
            )
            
            if demo_type == "Normal EEG":
                eeg_data = generate_sample_eeg()
                st.info("Using normal EEG demo data")
            else:
                eeg_data = generate_seizure_like_eeg()
                st.warning("Using seizure-like EEG demo data")
            
            # Plot the EEG signal
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                y=eeg_data,
                mode='lines',
                name='EEG Signal',
                line=dict(color='blue', width=1)
            ))
            fig.update_layout(
                title="EEG Signal Visualization",
                xaxis_title="Time Points",
                yaxis_title="Amplitude",
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)
            
        elif input_method == "📁 Upload File":
            uploaded_file = st.file_uploader("Upload EEG data (CSV/TXT)", type=['csv', 'txt'])
            if uploaded_file is not None:
                try:
                    data = pd.read_csv(uploaded_file, header=None)
                    if len(data.columns) >= 178:
                        eeg_data = data.iloc[0, :178].values.tolist()
                        st.success(f"Loaded {len(eeg_data)} features from file")
                    else:
                        st.error(f"File must contain at least 178 features. Found: {len(data.columns)}")
                        eeg_data = None
                except Exception as e:
                    st.error(f"Error reading file: {str(e)}")
                    eeg_data = None
            else:
                eeg_data = None
                
        else:  # Manual Input
            st.info("Manual input: Enter 178 comma-separated values")
            manual_input = st.text_area(
                "EEG Features (178 values):",
                placeholder="Enter 178 comma-separated numbers...",
                height=100
            )
            
            if manual_input:
                try:
                    eeg_data = [float(x.strip()) for x in manual_input.split(',')]
                    if len(eeg_data) != 178:
                        st.error(f"Expected 178 values, got {len(eeg_data)}")
                        eeg_data = None
                    else:
                        st.success(f"Parsed {len(eeg_data)} features")
                except ValueError:
                    st.error("Please enter valid numbers separated by commas")
                    eeg_data = None
            else:
                eeg_data = None
    
    with col2:
        st.subheader("Analysis Settings")
        
        model_type = st.selectbox(
            "Select Model:",
            ["cnn_3class", "bilstm_3class", "cnn_binary", "bilstm_binary"],
            help="Choose the model for prediction"
        )
        
        prediction_type = "binary" if "binary" in model_type else "3class"
        
        st.info(f"Using {prediction_type} classification")
        
        # Prediction button
        if st.button("🔍 Analyze EEG Signal", type="primary", disabled=not is_healthy or eeg_data is None):
            if eeg_data is not None:
                try:
                    # Prepare API request
                    endpoint = "/predict/binary" if prediction_type == "binary" else "/predict"
                    payload = {
                        "features": eeg_data,
                        "model": model_type
                    }
                    
                    # Make prediction
                    with st.spinner("Analyzing EEG signal..."):
                        response = requests.post(
                            f"{api_url}{endpoint}",
                            json=payload,
                            headers={"Content-Type": "application/json"},
                            timeout=30
                        )
                    
                    if response.status_code == 200:
                        result = response.json()
                        prediction = result['prediction']
                        
                        # Display results
                        st.success("Analysis Complete!")
                        
                        # Metrics
                        col_metrics1, col_metrics2, col_metrics3 = st.columns(3)
                        
                        with col_metrics1:
                            st.metric(
                                "Predicted Class",
                                prediction['class'],
                                help="Classification result"
                            )
                        
                        with col_metrics2:
                            st.metric(
                                "Confidence",
                                f"{prediction['confidence']:.2%}",
                                help="Model confidence in prediction"
                            )
                        
                        with col_metrics3:
                            if 'is_seizure' in prediction:
                                seizure_status = "⚡ SEIZURE" if prediction['is_seizure'] else "✅ NORMAL"
                                st.metric("Status", seizure_status)
                        
                        # Label and additional info
                        st.subheader("Analysis Results")
                        
                        if prediction['confidence'] > 0.8:
                            confidence_level = "High"
                            confidence_color = "success"
                        elif prediction['confidence'] > 0.6:
                            confidence_level = "Medium"
                            confidence_color = "warning"
                        else:
                            confidence_level = "Low"
                            confidence_color = "error"
                        
                        st.markdown(f"""
                        **Prediction**: {prediction['label']}  
                        **Confidence Level**: {confidence_level} ({prediction['confidence']:.2%})  
                        **Model Used**: {result['model_used']}  
                        **Analysis Time**: {result['timestamp']}
                        """)
                        
                        # Probability distribution
                        if 'probabilities' in result:
                            prob_data = result['probabilities']
                            labels = ['Class 0', 'Class 1'] if len(prob_data) == 2 else [f'Class {i}' for i in range(len(prob_data))]
                            
                            fig_prob = px.bar(
                                x=labels,
                                y=prob_data,
                                title="Class Probabilities",
                                color=prob_data,
                                color_continuous_scale="viridis"
                            )
                            fig_prob.update_layout(height=300)
                            st.plotly_chart(fig_prob, use_container_width=True)
                    
                    else:
                        st.error(f"Prediction failed: {response.text}")
                
                except requests.exceptions.RequestException as e:
                    st.error(f"API connection error: {str(e)}")
                except Exception as e:
                    st.error(f"Error during prediction: {str(e)}")

with tab2:
    st.header("Batch EEG Analysis")
    st.info("Analyze multiple EEG samples simultaneously")
    
    # Batch demo data generation
    col1, col2 = st.columns([1, 1])
    
    with col1:
        num_samples = st.number_input("Number of samples to generate:", min_value=1, max_value=10, value=3)
        
    with col2:
        batch_model = st.selectbox(
            "Batch Analysis Model:",
            ["cnn_3class", "bilstm_3class", "cnn_binary", "bilstm_binary"],
            key="batch_model"
        )
    
    if st.button("🔄 Generate & Analyze Batch", disabled=not is_healthy):
        try:
            # Generate batch data
            batch_data = []
            for i in range(num_samples):
                np.random.seed(42 + i)  # Different seed for each sample
                if i % 2 == 0:
                    sample = generate_sample_eeg()
                else:
                    sample = generate_seizure_like_eeg()
                batch_data.append(sample.tolist())
            
            # Make batch prediction
            payload = {
                "features": batch_data,
                "model": batch_model
            }
            
            with st.spinner(f"Analyzing {num_samples} EEG samples..."):
                response = requests.post(
                    f"{api_url}/predict/batch",
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=60
                )
            
            if response.status_code == 200:
                result = response.json()
                predictions = result['predictions']
                
                st.success(f"Batch analysis complete! Processed {len(predictions)} samples")
                
                # Create results DataFrame
                results_df = pd.DataFrame([
                    {
                        'Sample': f"Sample {p['sample_id'] + 1}",
                        'Predicted Class': p['class'],
                        'Confidence': f"{p['confidence']:.2%}",
                        'Confidence Score': p['confidence']
                    }
                    for p in predictions
                ])
                
                # Display results table
                st.subheader("Batch Results Summary")
                st.dataframe(results_df, use_container_width=True)
                
                # Visualization
                fig_batch = px.bar(
                    results_df,
                    x='Sample',
                    y='Confidence Score',
                    color='Predicted Class',
                    title=f"Batch Analysis Results ({batch_model})",
                    text='Confidence'
                )
                fig_batch.update_layout(height=400)
                st.plotly_chart(fig_batch, use_container_width=True)
                
            else:
                st.error(f"Batch prediction failed: {response.text}")
                
        except Exception as e:
            st.error(f"Batch analysis error: {str(e)}")

with tab3:
    st.header("Model Information & System Status")
    
    if is_healthy:
        try:
            # Get model info
            response = requests.get(f"{api_url}/info", timeout=10)
            
            if response.status_code == 200:
                model_info = response.json()
                
                st.subheader("📊 System Overview")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Loaded Models", model_info.get('loaded_models', 0))
                
                with col2:
                    total_params = sum(
                        details.get('parameters', 0) 
                        for details in model_info.get('model_details', {}).values()
                    )
                    st.metric("Total Parameters", f"{total_params:,}")
                
                with col3:
                    st.metric("API Status", "🟢 Online")
                
                # Model details
                st.subheader("🤖 Available Models")
                
                for model_name, details in model_info.get('model_details', {}).items():
                    with st.expander(f"📋 {model_name.upper()}", expanded=False):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write("**Model Architecture:**")
                            st.write(f"- Input Shape: `{details.get('input_shape', 'N/A')}`")
                            st.write(f"- Output Shape: `{details.get('output_shape', 'N/A')}`")
                            st.write(f"- Parameters: `{details.get('parameters', 0):,}`")
                        
                        with col2:
                            st.write("**Model Type:**")
                            if 'cnn' in model_name.lower():
                                st.write("🧠 Convolutional Neural Network")
                            elif 'lstm' in model_name.lower():
                                st.write("🔄 Bidirectional LSTM")
                            
                            if 'binary' in model_name:
                                st.write("📊 Binary Classification (Seizure/Normal)")
                            else:
                                st.write("📊 Multi-class Classification")
                
                # Training information (if available)
                if 'training_info' in model_info and model_info['training_info']:
                    st.subheader("📈 Training Performance")
                    
                    training_data = []
                    for model_name, info in model_info['training_info'].items():
                        if info.get('final_train_acc') is not None:
                            training_data.append({
                                'Model': model_name,
                                'Training Accuracy': f"{info['final_train_acc']:.2%}",
                                'Validation Accuracy': f"{info['final_val_acc']:.2%}"
                            })
                    
                    if training_data:
                        training_df = pd.DataFrame(training_data)
                        st.dataframe(training_df, use_container_width=True)
            
            else:
                st.error("Failed to retrieve model information")
                
        except Exception as e:
            st.error(f"Error retrieving model info: {str(e)}")
    
    else:
        st.error("API is not accessible. Please check the connection.")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        🧠 Epilepsy Seizure Detection System | Powered by Deep Learning & Flask API
    </div>
    """,
    unsafe_allow_html=True
)