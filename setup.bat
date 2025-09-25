@echo off
REM ================================================================================
REM 🚀 DEPLOYMENT SETUP SCRIPT (Windows)
REM ================================================================================

echo 🚀 Setting up Epilepsy Detection API for deployment...

REM Create models directory if it doesn't exist
echo 📁 Creating models directory...
if not exist "models" mkdir models

REM Copy model files from the parent directory
echo 📦 Copying model files...
if exist "..\saved_dl_models" (
    copy "..\saved_dl_models\*.h5" "models\" >nul 2>&1
    copy "..\saved_dl_models\*.pkl" "models\" >nul 2>&1
    echo ✅ Model files copied successfully!
) else (
    echo ⚠️  saved_dl_models directory not found. Please ensure your models are trained and saved.
)

REM List copied files
echo 📋 Files in models directory:
dir models

echo.
echo 🎉 Setup complete! Next steps:
echo 1. Install dependencies: pip install -r requirements.txt
echo 2. Test locally: python app.py
echo 3. Run tests: python test_api.py
echo 4. Deploy to Render using the README instructions

pause