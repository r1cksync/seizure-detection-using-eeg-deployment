
"""
Quick test to verify the negative confidence fix
"""
import requests
import numpy as np
from datetime import datetime

API_BASE_URL = "https://seizure-detection-using-eeg-deployment.onrender.com"

def test_confidence_fix():
    print("🔧 Testing Negative Confidence Fix")
    print("=" * 40)
    
    # Test case that previously gave negative confidence
    flat_signal = np.full(178, -10) + np.random.normal(0, 5, 178)
    
    payload = {
        "features": flat_signal.tolist(),
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
            
            print(f"✅ Success!")
            print(f"   Class: {prediction['class']} ({prediction['label']})")
            print(f"   Confidence: {prediction['confidence']:.6f}")
            print(f"   Is Seizure: {prediction['is_seizure']}")
            
            # Check if confidence is now positive (probability)
            if 0 <= prediction['confidence'] <= 1:
                print(f"✅ Confidence is now a valid probability!")
            else:
                print(f"❌ Confidence still not a valid probability: {prediction['confidence']}")
            
            # Show raw logits vs probabilities
            if 'raw_logits' in result:
                print(f"   Raw logits: {[f'{x:.4f}' for x in result['raw_logits']]}")
                print(f"   Probabilities: {[f'{x:.6f}' for x in result['probabilities']]}")
                
                # Verify probabilities sum to 1
                prob_sum = sum(result['probabilities'])
                print(f"   Probability sum: {prob_sum:.6f} {'✅' if abs(prob_sum - 1.0) < 0.001 else '❌'}")
            
        else:
            print(f"❌ Failed: {response.status_code} - {response.text}")
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")

def test_various_inputs():
    print(f"\n🧪 Testing Various Input Types")
    print("=" * 35)
    
    test_cases = [
        ("Normal EEG", np.random.normal(-10, 50, 178)),
        ("Seizure-like", np.random.normal(-5, 80, 178) + np.random.normal(0, 100, 178) * (np.random.rand(178) < 0.1)),
        ("Standardized", np.random.normal(0, 1, 178)),
        ("Large values", np.random.uniform(-1000, 1000, 178)),
    ]
    
    for name, data in test_cases:
        print(f"\n🔬 {name}:")
        print(f"   Range: [{data.min():.2f}, {data.max():.2f}]")
        
        payload = {
            "features": data.tolist(),
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
                confidence = prediction['confidence']
                
                status = "✅" if 0 <= confidence <= 1 else "❌"
                print(f"   {status} Confidence: {confidence:.6f}")
                print(f"   Prediction: {prediction['label']}")
                
        except Exception as e:
            print(f"   ❌ Error: {str(e)}")

if __name__ == "__main__":
    print(f"🚀 API Fix Verification")
    print(f"URL: {API_BASE_URL}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    test_confidence_fix()
    test_various_inputs()
    
    print(f"\n{'='*50}")
    print("🎉 Fix verification completed!")
    print(f"{'='*50}")