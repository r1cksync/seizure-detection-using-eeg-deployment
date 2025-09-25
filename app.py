# ================================================================================
# 🚀 EPILEPSY SEIZURE DETECTION API - FLASK APPLICATION
# ================================================================================

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle
import os
import logging
from datetime import datetime
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Global variables to store loaded models
models = {}
model_info = None

def load_models():
    """Load all trained models at startup"""
    global models, model_info
    
    try:
        # Model file paths (adjust these based on your uploaded model files)
        model_files = {
            'cnn_3class': 'models/cnn_3class_epilepsy_20250925_065912.h5',
            'bilstm_3class': 'models/bilstm_3class_epilepsy_20250925_065912.h5',
            'cnn_binary': 'models/cnn_binary_epilepsy_20250925_065912.h5',
            'bilstm_binary': 'models/bilstm_binary_epilepsy_20250925_065912.h5'
        }
        
        # Load each model
        for model_name, model_path in model_files.items():
            if os.path.exists(model_path):
                models[model_name] = load_model(model_path)
                logger.info(f"✅ Loaded {model_name} from {model_path}")
            else:
                logger.warning(f"⚠️ Model file not found: {model_path}")
        
        # Load model information
        info_path = 'models/model_info_20250925_065912.pkl'
        if os.path.exists(info_path):
            with open(info_path, 'rb') as f:
                model_info = pickle.load(f)
            logger.info("✅ Loaded model information")
        
        logger.info(f"🎉 Successfully loaded {len(models)} models")
        
    except Exception as e:
        logger.error(f"❌ Error loading models: {str(e)}")
        logger.error(traceback.format_exc())

def preprocess_input(data):
    """Preprocess input data for model prediction"""
    try:
        # Convert to numpy array if it's a list
        if isinstance(data, list):
            data = np.array(data)
        
        # Ensure correct shape (batch_size, 178, 1)
        if data.shape == (178,):
            # Single sample: reshape to (1, 178, 1)
            data = data.reshape(1, 178, 1)
        elif data.shape == (178, 1):
            # Single sample with correct feature dimension: reshape to (1, 178, 1)
            data = data.reshape(1, 178, 1)
        elif len(data.shape) == 2 and data.shape[1] == 178:
            # Multiple samples: reshape to (batch_size, 178, 1)
            data = np.expand_dims(data, axis=2)
        elif len(data.shape) == 3 and data.shape[1:] == (178, 1):
            # Already in correct shape
            pass
        else:
            raise ValueError(f"Invalid input shape: {data.shape}. Expected (178,) or (batch_size, 178)")
        
        # Normalize data (if needed - adjust based on your training preprocessing)
        # data = (data - np.mean(data)) / np.std(data)
        
        return data
        
    except Exception as e:
        logger.error(f"❌ Preprocessing error: {str(e)}")
        raise

@app.route('/', methods=['GET'])
def home():
    """API health check and information endpoint"""
    return jsonify({
        'status': 'active',
        'message': 'Epilepsy Seizure Detection API',
        'timestamp': datetime.now().isoformat(),
        'available_models': list(models.keys()),
        'endpoints': {
            'prediction': '/predict',
            'binary_prediction': '/predict/binary',
            'batch_prediction': '/predict/batch',
            'model_info': '/info'
        }
    })

@app.route('/info', methods=['GET'])
def get_model_info():
    """Get information about loaded models"""
    try:
        info = {
            'loaded_models': len(models),
            'model_details': {}
        }
        
        for model_name, model in models.items():
            info['model_details'][model_name] = {
                'input_shape': str(model.input_shape),
                'output_shape': str(model.output_shape),
                'parameters': model.count_params()
            }
        
        if model_info:
            info['training_info'] = model_info.get('training_histories', {})
        
        return jsonify(info)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    """Main prediction endpoint for 3-class classification"""
    try:
        # Get JSON data
        data = request.get_json()
        
        if not data or 'features' not in data:
            return jsonify({'error': 'Missing features in request body'}), 400
        
        # Get model preference (default to CNN)
        model_type = data.get('model', 'cnn_3class')
        
        if model_type not in models:
            available_models = list(models.keys())
            return jsonify({
                'error': f'Model {model_type} not available',
                'available_models': available_models
            }), 400
        
        # Preprocess input
        features = preprocess_input(data['features'])
        
        # Make prediction
        model = models[model_type]
        predictions = model.predict(features)
        predicted_class = np.argmax(predictions, axis=1)[0]
        confidence = float(np.max(predictions))
        
        # Map class to seizure type (adjust based on your class mapping)
        class_mapping = {
            0: 'Seizure Activity',
            1: 'Normal Activity'
        }
        
        result = {
            'prediction': {
                'class': int(predicted_class),
                'label': class_mapping.get(predicted_class, f'Class {predicted_class}'),
                'confidence': confidence
            },
            'model_used': model_type,
            'probabilities': predictions[0].tolist(),
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"❌ Prediction error: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/predict/binary', methods=['POST'])
def predict_binary():
    """Binary prediction endpoint (Epileptic vs Others)"""
    try:
        data = request.get_json()
        
        if not data or 'features' not in data:
            return jsonify({'error': 'Missing features in request body'}), 400
        
        # Use binary classification models
        model_type = data.get('model', 'cnn_binary')
        
        if model_type not in ['cnn_binary', 'bilstm_binary']:
            return jsonify({
                'error': f'Binary model {model_type} not available',
                'available_models': ['cnn_binary', 'bilstm_binary']
            }), 400
        
        if model_type not in models:
            return jsonify({'error': f'Model {model_type} not loaded'}), 500
        
        # Preprocess and predict
        features = preprocess_input(data['features'])
        model = models[model_type]
        predictions = model.predict(features)
        predicted_class = np.argmax(predictions, axis=1)[0]
        confidence = float(np.max(predictions))
        
        # Binary classification mapping
        binary_mapping = {
            0: 'Non-Epileptic',
            1: 'Epileptic Seizure'
        }
        
        result = {
            'prediction': {
                'class': int(predicted_class),
                'label': binary_mapping.get(predicted_class, f'Class {predicted_class}'),
                'confidence': confidence,
                'is_seizure': bool(predicted_class == 1)
            },
            'model_used': model_type,
            'probabilities': predictions[0].tolist(),
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"❌ Binary prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    """Batch prediction endpoint for multiple samples"""
    try:
        data = request.get_json()
        
        if not data or 'features' not in data:
            return jsonify({'error': 'Missing features in request body'}), 400
        
        model_type = data.get('model', 'cnn_3class')
        
        if model_type not in models:
            return jsonify({'error': f'Model {model_type} not available'}), 400
        
        # Preprocess batch input
        features = preprocess_input(data['features'])
        
        # Make batch predictions
        model = models[model_type]
        predictions = model.predict(features)
        predicted_classes = np.argmax(predictions, axis=1)
        confidences = np.max(predictions, axis=1)
        
        # Format results
        results = []
        for i in range(len(predicted_classes)):
            results.append({
                'sample_id': i,
                'class': int(predicted_classes[i]),
                'confidence': float(confidences[i]),
                'probabilities': predictions[i].tolist()
            })
        
        return jsonify({
            'predictions': results,
            'model_used': model_type,
            'batch_size': len(results),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"❌ Batch prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Load models at startup
    logger.info("🚀 Starting Epilepsy Detection API...")
    load_models()
    
    # Run the app
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)