# Check if model files are properly in the repository
import os

def check_models():
    print("🔍 Checking model files in the repository...")
    
    models_dir = "models"
    if not os.path.exists(models_dir):
        print("❌ Models directory not found!")
        return False
    
    files = os.listdir(models_dir)
    print(f"📁 Files in models directory: {files}")
    
    h5_files = [f for f in files if f.endswith('.h5')]
    pkl_files = [f for f in files if f.endswith('.pkl')]
    
    print(f"🧠 H5 model files found: {len(h5_files)}")
    for file in h5_files:
        size = os.path.getsize(os.path.join(models_dir, file))
        print(f"   📊 {file}: {size:,} bytes")
    
    print(f"📋 PKL info files found: {len(pkl_files)}")
    for file in pkl_files:
        size = os.path.getsize(os.path.join(models_dir, file))
        print(f"   📊 {file}: {size:,} bytes")
    
    if len(h5_files) >= 4:
        print("✅ All model files appear to be present!")
        return True
    else:
        print("❌ Missing model files!")
        return False

if __name__ == "__main__":
    check_models()