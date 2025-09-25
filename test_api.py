# ================================================================================
# 🧪 API TESTING SCRIPT - EPILEPSY SEIZURE DETECTION
# ================================================================================

import requests
import json
import numpy as np
from datetime import datetime

# Configuration
API_BASE_URL = "http://localhost:5000"  # Change to your Render URL when deployed
# Example: API_BASE_URL = "https://your-app-name.onrender.com"

def test_api_health():
    """Test if API is running"""
    print("🏥 Testing API Health...")
    try:
        response = requests.get(f"{API_BASE_URL}/")
        if response.status_code == 200:
            data = response.json()
            print("✅ API is healthy!")
            print(f"   Available models: {data.get('available_models', [])}")
            return True
        else:
            print(f"❌ API health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Cannot connect to API: {str(e)}")
        return False

def test_model_info():
    """Test model information endpoint"""
    print("\n📊 Testing Model Info...")
    try:
        response = requests.get(f"{API_BASE_URL}/info")
        if response.status_code == 200:
            data = response.json()
            print("✅ Model info retrieved successfully!")
            print(f"   Loaded models: {data.get('loaded_models', 0)}")
            for model_name, details in data.get('model_details', {}).items():
                print(f"   📋 {model_name}: {details.get('input_shape')} -> {details.get('output_shape')}")
        else:
            print(f"❌ Model info failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Model info error: {str(e)}")

def generate_sample_eeg_data():
    """Generate sample EEG data for testing"""
    # Generate realistic-looking EEG signal with 178 features
    np.random.seed(42)  # For reproducibility
    
    # Simulate EEG signal characteristics
    time_points = np.linspace(0, 1, 178)
    
    # Base signal with multiple frequency components
    signal = (
        2 * np.sin(2 * np.pi * 10 * time_points) +  # 10 Hz alpha waves
        1.5 * np.sin(2 * np.pi * 4 * time_points) +  # 4 Hz theta waves
        0.5 * np.sin(2 * np.pi * 20 * time_points) +  # 20 Hz beta waves
        np.random.normal(0, 0.3, 178)  # Noise
    )
    
    # Add some seizure-like characteristics (high amplitude spikes)
    seizure_indices = np.random.choice(178, 10, replace=False)
    signal[seizure_indices] += np.random.normal(0, 3, 10)
    
    return signal.tolist()

def test_3class_prediction():
    """Test 3-class seizure classification"""
    print("\n🧠 Testing 3-Class Prediction...")
    
    sample_data = generate_sample_eeg_data()
    
    payload = {
        "features": sample_data,
        "model": "cnn_3class"  # or "bilstm_3class"
    }
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/predict",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            result = response.json()
            prediction = result['prediction']
            print("✅ 3-Class prediction successful!")
            print(f"   📊 Predicted Class: {prediction['class']}")
            print(f"   🏷️  Label: {prediction['label']}")
            print(f"   🎯 Confidence: {prediction['confidence']:.4f}")
            print(f"   🤖 Model Used: {result['model_used']}")
        else:
            print(f"❌ 3-Class prediction failed: {response.status_code}")
            print(f"   Error: {response.text}")
            
    except Exception as e:
        print(f"❌ 3-Class prediction error: {str(e)}")

def test_binary_prediction():
    """Test binary seizure detection"""
    print("\n⚡ Testing Binary Prediction (Seizure Detection)...")
    
    sample_data = generate_sample_eeg_data()
    
    payload = {
        "features": sample_data,
        "model": "cnn_binary"  # or "bilstm_binary"
    }
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/predict/binary",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            result = response.json()
            prediction = result['prediction']
            print("✅ Binary prediction successful!")
            print(f"   📊 Predicted Class: {prediction['class']}")
            print(f"   🏷️  Label: {prediction['label']}")
            print(f"   ⚡ Is Seizure: {prediction['is_seizure']}")
            print(f"   🎯 Confidence: {prediction['confidence']:.4f}")
            print(f"   🤖 Model Used: {result['model_used']}")
        else:
            print(f"❌ Binary prediction failed: {response.status_code}")
            print(f"   Error: {response.text}")
            
    except Exception as e:
        print(f"❌ Binary prediction error: {str(e)}")

def test_batch_prediction():
    """Test batch prediction"""
    print("\n📦 Testing Batch Prediction...")
    
    # Generate multiple samples
    batch_data = [generate_sample_eeg_data() for _ in range(3)]
    
    payload = {
        "features": batch_data,
        "model": "cnn_3class"
    }
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/predict/batch",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Batch prediction successful!")
            print(f"   📦 Batch Size: {result['batch_size']}")
            print(f"   🤖 Model Used: {result['model_used']}")
            
            for i, pred in enumerate(result['predictions'][:3]):  # Show first 3
                print(f"   📊 Sample {i+1}: Class {pred['class']}, Confidence: {pred['confidence']:.4f}")
                
        else:
            print(f"❌ Batch prediction failed: {response.status_code}")
            print(f"   Error: {response.text}")
            
    except Exception as e:
        print(f"❌ Batch prediction error: {str(e)}")

def main():
    """Run all API tests"""
    print("🚀 EPILEPSY SEIZURE DETECTION API - TESTING SUITE")
    print("=" * 60)
    print(f"🌐 Testing API at: {API_BASE_URL}")
    print(f"⏰ Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # Test API health first
    if not test_api_health():
        print("\n❌ API is not accessible. Please check:")
        print("   1. Is the Flask app running?")
        print("   2. Is the URL correct?")
        print("   3. Are there any network issues?")
        return
    
    # Run all tests
    test_model_info()
    test_3class_prediction()
    test_binary_prediction()
    test_batch_prediction()
    
    print("\n" + "=" * 60)
    print("🎉 API Testing Completed!")
    print("=" * 60)

if __name__ == "__main__":
    main()