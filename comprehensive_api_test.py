# ================================================================================
# 🧪 COMPREHENSIVE API TESTING SCRIPT - EPILEPSY SEIZURE DETECTION
# ================================================================================
# This script thoroughly tests the API with proper input ranges and validates model correctness

import requests
import json
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

# Configuration
API_BASE_URL = "https://seizure-detection-using-eeg-deployment.onrender.com"

def print_header(title):
    """Print formatted header"""
    print(f"\n{'='*60}")
    print(f"🧪 {title}")
    print(f"{'='*60}")

def test_api_health():
    """Test if API is running"""
    print("🏥 Testing API Health...")
    try:
        response = requests.get(f"{API_BASE_URL}/")
        if response.status_code == 200:
            data = response.json()
            print("✅ API is healthy!")
            print(f"   Available models: {data.get('available_models', [])}")
            return True, data.get('available_models', [])
        else:
            print(f"❌ API health check failed: {response.status_code}")
            return False, []
    except Exception as e:
        print(f"❌ Cannot connect to API: {str(e)}")
        return False, []

def generate_realistic_eeg_data(signal_type="normal", seed=None):
    """
    Generate realistic EEG data based on actual dataset characteristics
    
    Dataset info:
    - Raw value range: -1885 to 2047
    - Mean around -7.72
    - Standard deviation: 153.88 to 172.44
    - 178 features per sample
    """
    if seed:
        np.random.seed(seed)
    
    if signal_type == "normal":
        # Normal EEG: smaller amplitudes, more regular patterns
        base_signal = np.random.normal(-10, 50, 178)  # Closer to dataset mean
        # Add some alpha and beta wave patterns
        time_points = np.linspace(0, 1, 178)
        base_signal += 20 * np.sin(2 * np.pi * 10 * time_points)  # Alpha waves
        base_signal += 10 * np.sin(2 * np.pi * 15 * time_points)  # Beta waves
        
    elif signal_type == "seizure":
        # Seizure EEG: higher amplitudes, more chaotic
        base_signal = np.random.normal(-5, 80, 178)
        # Add high-amplitude spikes characteristic of seizures
        spike_indices = np.random.choice(178, 15, replace=False)
        base_signal[spike_indices] += np.random.normal(0, 200, 15)
        # Add some rapid oscillations
        time_points = np.linspace(0, 1, 178)
        base_signal += 50 * np.sin(2 * np.pi * 25 * time_points)  # Fast waves
        
    elif signal_type == "artifact":
        # High amplitude artifacts (should be detected as abnormal)
        base_signal = np.random.normal(100, 300, 178)
        
    elif signal_type == "flat":
        # Flat signal (minimal variation)
        base_signal = np.full(178, -10) + np.random.normal(0, 5, 178)
        
    else:  # random
        # Completely random within dataset range
        base_signal = np.random.uniform(-1000, 1000, 178)
    
    return base_signal.tolist()

def test_prediction_with_various_inputs():
    """Test API with various types of input data"""
    print_header("TESTING WITH VARIOUS INPUT TYPES")
    
    test_cases = [
        ("normal_1", "normal", 42),
        ("normal_2", "normal", 123),
        ("seizure_1", "seizure", 456),
        ("seizure_2", "seizure", 789),
        ("artifact", "artifact", 999),
        ("flat_signal", "flat", 111),
        ("random_data", "random", 222),
    ]
    
    results = {}
    
    for test_name, signal_type, seed in test_cases:
        print(f"\n🔬 Testing {test_name} ({signal_type}) data...")
        
        # Generate test data
        sample_data = generate_realistic_eeg_data(signal_type, seed)
        
        # Test both binary and 3-class predictions
        for endpoint, model in [("/predict/binary", "cnn_binary"), ("/predict", "cnn_3class")]:
            payload = {
                "features": sample_data,
                "model": model
            }
            
            try:
                response = requests.post(
                    f"{API_BASE_URL}{endpoint}",
                    json=payload,
                    headers={"Content-Type": "application/json"}
                )
                
                if response.status_code == 200:
                    result = response.json()
                    prediction = result['prediction']
                    
                    # Store results
                    key = f"{test_name}_{model}"
                    results[key] = {
                        'signal_type': signal_type,
                        'class': prediction['class'],
                        'confidence': prediction['confidence'],
                        'label': prediction['label'],
                        'probabilities': result.get('probabilities', []),
                        'raw_output': result
                    }
                    
                    print(f"   ✅ {model}: Class {prediction['class']} ({prediction['label']}) - Confidence: {prediction['confidence']:.4f}")
                    
                    # Check for negative confidence issue
                    if prediction['confidence'] < 0:
                        print(f"   ⚠️  WARNING: Negative confidence detected! This suggests model output isn't properly processed.")
                        print(f"       Raw probabilities: {result.get('probabilities', 'N/A')}")
                        
                else:
                    print(f"   ❌ {model} failed: {response.status_code} - {response.text}")
                    
            except Exception as e:
                print(f"   ❌ {model} error: {str(e)}")
    
    return results

def analyze_model_behavior(results):
    """Analyze the model behavior from test results"""
    print_header("MODEL BEHAVIOR ANALYSIS")
    
    # Group results by model
    binary_results = {k: v for k, v in results.items() if 'cnn_binary' in k}
    class3_results = {k: v for k, v in results.items() if 'cnn_3class' in k}
    
    print("📊 Binary Classification Results:")
    print("-" * 40)
    for key, result in binary_results.items():
        signal_type = result['signal_type']
        confidence = result['confidence']
        is_seizure = result['raw_output']['prediction'].get('is_seizure', False)
        print(f"   {signal_type:10} -> {'SEIZURE' if is_seizure else 'NORMAL':8} (conf: {confidence:8.4f})")
    
    print("\n📊 3-Class Classification Results:")
    print("-" * 40)
    for key, result in class3_results.items():
        signal_type = result['signal_type']
        confidence = result['confidence']
        label = result['label']
        print(f"   {signal_type:10} -> {label:15} (conf: {confidence:8.4f})")
    
    # Check for issues
    print("\n🔍 POTENTIAL ISSUES DETECTED:")
    print("-" * 35)
    
    negative_confidences = [r for r in results.values() if r['confidence'] < 0]
    if negative_confidences:
        print(f"❌ Negative confidences found: {len(negative_confidences)} cases")
        print("   This suggests model outputs aren't properly processed through softmax/sigmoid")
    
    # Check for identical predictions
    binary_preds = [r['class'] for r in binary_results.values()]
    class3_preds = [r['class'] for r in class3_results.values()]
    
    if len(set(binary_preds)) == 1:
        print(f"❌ Binary model always predicts class {binary_preds[0]} - model may be broken")
    
    if len(set(class3_preds)) == 1:
        print(f"❌ 3-class model always predicts class {class3_preds[0]} - model may be broken")
    
    # Check confidence ranges
    all_confidences = [r['confidence'] for r in results.values()]
    print(f"\n📈 Confidence Statistics:")
    print(f"   Range: {min(all_confidences):.4f} to {max(all_confidences):.4f}")
    print(f"   Mean: {np.mean(all_confidences):.4f}")
    print(f"   Std: {np.std(all_confidences):.4f}")

def test_input_ranges():
    """Test API with different input value ranges"""
    print_header("INPUT RANGE TESTING")
    
    print("🎯 Based on dataset analysis:")
    print("   • Raw data range: -1885 to 2047")
    print("   • Mean: ~-7.72")
    print("   • Std: 153.88 to 172.44")
    print("   • Models expect standardized inputs (mean=0, std=1)")
    
    range_tests = [
        ("dataset_range", np.random.uniform(-1885, 2047, 178)),
        ("standardized", np.random.normal(0, 1, 178)),
        ("small_values", np.random.uniform(-1, 1, 178)),
        ("large_values", np.random.uniform(-10000, 10000, 178)),
        ("zeros", np.zeros(178)),
        ("ones", np.ones(178)),
    ]
    
    for test_name, test_data in range_tests:
        print(f"\n🔬 Testing {test_name} range...")
        print(f"   Data range: [{test_data.min():.2f}, {test_data.max():.2f}]")
        print(f"   Mean: {test_data.mean():.2f}, Std: {test_data.std():.2f}")
        
        payload = {
            "features": test_data.tolist(),
            "model": "cnn_binary"
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
                print(f"   ✅ Success: Class {prediction['class']} - Confidence: {prediction['confidence']:.4f}")
            else:
                print(f"   ❌ Failed: {response.status_code} - {response.text}")
                
        except Exception as e:
            print(f"   ❌ Error: {str(e)}")

def test_edge_cases():
    """Test edge cases and potential errors"""
    print_header("EDGE CASE TESTING")
    
    edge_cases = [
        ("empty_array", []),
        ("wrong_size_small", np.random.normal(0, 1, 100).tolist()),
        ("wrong_size_large", np.random.normal(0, 1, 200).tolist()),
        ("nan_values", [float('nan')] * 178),
        ("inf_values", [float('inf')] * 178),
        ("missing_features", None),
    ]
    
    for test_name, test_data in edge_cases:
        print(f"\n🔬 Testing {test_name}...")
        
        if test_data is None:
            payload = {"model": "cnn_binary"}  # Missing features
        else:
            payload = {
                "features": test_data,
                "model": "cnn_binary"
            }
        
        try:
            response = requests.post(
                f"{API_BASE_URL}/predict/binary",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"   ⚠️  Unexpected success: {result}")
            else:
                print(f"   ✅ Properly rejected: {response.status_code} - {response.text[:100]}")
                
        except Exception as e:
            print(f"   ✅ Properly failed: {str(e)}")

def generate_usage_examples():
    """Generate code examples for different input ranges"""
    print_header("USAGE EXAMPLES & RECOMMENDED INPUT RANGES")
    
    print("🎯 RECOMMENDED INPUT DATA GENERATION:")
    print("-" * 45)
    
    print("\n1️⃣ Normal EEG Signal (Non-seizure):")
    print("```python")
    print("import numpy as np")
    print("# Generate normal EEG-like signal")
    print("time_points = np.linspace(0, 1, 178)")
    print("normal_eeg = (")
    print("    np.random.normal(-10, 50, 178) +")
    print("    20 * np.sin(2 * np.pi * 10 * time_points) +  # Alpha waves")
    print("    10 * np.sin(2 * np.pi * 15 * time_points)    # Beta waves")
    print(")")
    print("```")
    
    print("\n2️⃣ Seizure-like EEG Signal:")
    print("```python")
    print("# Generate seizure-like signal with spikes")
    print("seizure_eeg = np.random.normal(-5, 80, 178)")
    print("spike_indices = np.random.choice(178, 15, replace=False)")
    print("seizure_eeg[spike_indices] += np.random.normal(0, 200, 15)")
    print("```")
    
    print("\n3️⃣ API Usage Example:")
    print("```python")
    print("import requests")
    print("import numpy as np")
    print("")
    print("# Generate test data")
    print("eeg_data = np.random.normal(-10, 60, 178).tolist()")
    print("")
    print("# Binary seizure detection")
    print("response = requests.post(")
    print("    'https://seizure-detection-using-eeg-deployment.onrender.com/predict/binary',")
    print("    json={'features': eeg_data, 'model': 'cnn_binary'}")
    print(")")
    print("result = response.json()")
    print("print(f\"Seizure: {result['prediction']['is_seizure']}\")")
    print("```")
    
    print("\n📊 INPUT RANGE RECOMMENDATIONS:")
    print("-" * 35)
    print("✅ GOOD ranges (based on training data):")
    print("   • Normal signals: mean=-10, std=50, range=[-200, 200]")
    print("   • Seizure signals: mean=-5, std=80, range=[-400, 400]")
    print("   • General use: np.random.normal(-10, 60, 178)")
    print("")
    print("⚠️  AVOID these ranges:")
    print("   • Extremely large values: >5000 or <-5000")
    print("   • All zeros or identical values")
    print("   • NaN or infinite values")
    print("   • Wrong array size (not 178 features)")

def main():
    """Run comprehensive API testing"""
    print("🚀 COMPREHENSIVE EPILEPSY SEIZURE DETECTION API TESTING")
    print("=" * 65)
    print(f"🌐 API URL: {API_BASE_URL}")
    print(f"⏰ Test started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Test API health
    is_healthy, available_models = test_api_health()
    if not is_healthy:
        print("❌ API is not responding. Exiting...")
        return
    
    # Run comprehensive tests
    results = test_prediction_with_various_inputs()
    analyze_model_behavior(results)
    test_input_ranges()
    test_edge_cases()
    generate_usage_examples()
    
    print(f"\n{'='*65}")
    print("🎉 COMPREHENSIVE TESTING COMPLETED!")
    print(f"{'='*65}")

if __name__ == "__main__":
    main()