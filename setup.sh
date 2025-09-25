#!/bin/bash

# ================================================================================
# 🚀 DEPLOYMENT SETUP SCRIPT
# ================================================================================

echo "🚀 Setting up Epilepsy Detection API for deployment..."

# Create models directory if it doesn't exist
echo "📁 Creating models directory..."
mkdir -p models

# Copy model files from the parent directory
echo "📦 Copying model files..."
if [ -d "../saved_dl_models" ]; then
    cp ../saved_dl_models/*.h5 models/ 2>/dev/null || echo "⚠️  No .h5 files found"
    cp ../saved_dl_models/*.pkl models/ 2>/dev/null || echo "⚠️  No .pkl files found"
    echo "✅ Model files copied successfully!"
else
    echo "⚠️  saved_dl_models directory not found. Please ensure your models are trained and saved."
fi

# List copied files
echo "📋 Files in models directory:"
ls -la models/

echo ""
echo "🎉 Setup complete! Next steps:"
echo "1. Install dependencies: pip install -r requirements.txt"
echo "2. Test locally: python app.py"
echo "3. Run tests: python test_api.py"
echo "4. Deploy to Render using the README instructions"