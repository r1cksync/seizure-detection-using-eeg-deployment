# Epilepsy Seizure Detection API 🧠⚡

A Flask-based REST API for epilepsy seizure detection using deep learning models (CNN and BiLSTM).

## 🎯 Features

- **Multiple Model Support**: CNN and BiLSTM models for both 3-class and binary classification
- **REST API Endpoints**: Easy-to-use HTTP endpoints for predictions
- **Batch Processing**: Support for single and batch predictions
- **Model Information**: Get details about loaded models
- **Error Handling**: Comprehensive error handling and logging
- **CORS Support**: Cross-origin requests enabled

## 📁 Project Structure

```
epilepsy_api_deployment/
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── Procfile              # Render deployment configuration
├── test_api.py           # API testing script
├── models/               # Directory for model files
│   ├── cnn_3class_epilepsy_*.h5
│   ├── bilstm_3class_epilepsy_*.h5
│   ├── cnn_binary_epilepsy_*.h5
│   ├── bilstm_binary_epilepsy_*.h5
│   └── model_info_*.pkl
└── README.md            # This file
```

## 🚀 Deployment on Render

### Step 1: Prepare Model Files

1. Copy your saved model files to the `models/` directory:
   ```bash
   cp ../saved_dl_models/*.h5 models/
   cp ../saved_dl_models/*.pkl models/
   ```

### Step 2: Deploy to Render

1. **Create a Render account**: Go to [render.com](https://render.com) and sign up
2. **Create a new Web Service**:
   - Connect your GitHub repository
   - Or upload this folder as a ZIP file
3. **Configure the service**:
   - **Environment**: Python 3
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app`
   - **Auto-Deploy**: Yes

### Step 3: Environment Variables (Optional)

You can set these environment variables in Render:
- `PORT`: Port number (automatically set by Render)
- `FLASK_ENV`: Set to `production`

## 🔧 Local Development

### Setup

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Copy model files**:
   ```bash
   # Copy your trained models to the models/ directory
   cp ../saved_dl_models/*.h5 models/
   cp ../saved_dl_models/*.pkl models/
   ```

3. **Run the application**:
   ```bash
   python app.py
   ```

4. **Test the API**:
   ```bash
   python test_api.py
   ```

## 📡 API Endpoints

### Base URL
- **Local**: `http://localhost:5000`
- **Render**: `https://your-app-name.onrender.com`

### Endpoints

#### 1. Health Check
```http
GET /
```
Returns API status and available models.

#### 2. Model Information
```http
GET /info
```
Returns detailed information about loaded models.

#### 3. 3-Class Prediction
```http
POST /predict
```
**Request Body**:
```json
{
    "features": [/* 178 EEG feature values */],
    "model": "cnn_3class"  // or "bilstm_3class"
}
```

**Response**:
```json
{
    "prediction": {
        "class": 0,
        "label": "Seizure Activity",
        "confidence": 0.85
    },
    "model_used": "cnn_3class",
    "probabilities": [0.85, 0.15],
    "timestamp": "2025-09-25T12:00:00"
}
```

#### 4. Binary Prediction (Seizure Detection)
```http
POST /predict/binary
```
**Request Body**:
```json
{
    "features": [/* 178 EEG feature values */],
    "model": "cnn_binary"  // or "bilstm_binary"
}
```

**Response**:
```json
{
    "prediction": {
        "class": 1,
        "label": "Epileptic Seizure",
        "confidence": 0.92,
        "is_seizure": true
    },
    "model_used": "cnn_binary",
    "probabilities": [0.08, 0.92],
    "timestamp": "2025-09-25T12:00:00"
}
```

#### 5. Batch Prediction
```http
POST /predict/batch
```
**Request Body**:
```json
{
    "features": [
        [/* Sample 1: 178 values */],
        [/* Sample 2: 178 values */],
        [/* Sample 3: 178 values */]
    ],
    "model": "cnn_3class"
}
```

## 🧪 Testing

Run the test script to verify all endpoints:

```bash
python test_api.py
```

This will test:
- ✅ API health check
- 📊 Model information retrieval
- 🧠 3-class prediction
- ⚡ Binary seizure detection
- 📦 Batch prediction

## 📊 Model Details

### Available Models:
1. **CNN 3-Class**: Convolutional Neural Network for 3-class classification
2. **BiLSTM 3-Class**: Bidirectional LSTM for 3-class classification
3. **CNN Binary**: CNN for binary seizure detection (Epileptic vs Others)
4. **BiLSTM Binary**: BiLSTM for binary seizure detection

### Input Format:
- **Shape**: 178 EEG features per sample
- **Type**: List of floating-point numbers
- **Range**: Raw EEG signal values

## 🔒 Security & Production Notes

- API includes CORS support for web applications
- Comprehensive error handling and logging
- Input validation and preprocessing
- Model loading optimization
- Production-ready with Gunicorn

## 🐛 Troubleshooting

### Common Issues:

1. **Models not loading**:
   - Ensure model files are in the `models/` directory
   - Check file paths in `app.py`
   - Verify TensorFlow version compatibility

2. **Memory issues on Render**:
   - Consider using smaller models
   - Optimize batch sizes
   - Use model quantization

3. **Slow predictions**:
   - Models are loaded at startup (not per request)
   - Consider caching frequently used predictions
   - Use GPU instances if available

## 📈 Performance Optimization

- Models are loaded once at startup
- Batch processing for multiple samples
- Efficient numpy operations
- Memory-optimized preprocessing

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is part of an academic seizure detection research project.

---

**🚀 Ready to deploy? Follow the Render deployment steps above!**