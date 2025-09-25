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

def load_model_on_demand(model_name):
    """Load a specific model on demand to save memory"""
    global models, model_info
    
    # If model already loaded, return it
    if model_name in models:
        logger.info(f"🔄 Using cached model: {model_name}")
        return models[model_name]
    
    try:
        # Check if models directory exists
        models_dir = 'models'
        if not os.path.exists(models_dir):
            logger.error(f"❌ Models directory not found: {models_dir}")
            return None
        
        # Find the model file
        files_in_models = os.listdir(models_dir)
        h5_files = [f for f in files_in_models if f.endswith('.h5') and model_name in f]
        
        if not h5_files:
            logger.error(f"❌ Model file for '{model_name}' not found in {files_in_models}")
            return None
        
        model_file = h5_files[0]
        model_path = f"{models_dir}/{model_file}"
        
        logger.info(f"� Loading {model_name} on-demand from {model_path}")
        
        # Check file size
        file_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
        logger.info(f"📊 File size: {file_size:.2f} MB")
        
        # Load the model
        model = load_model(model_path)
        models[model_name] = model  # Cache for future use
        
        logger.info(f"✅ Successfully loaded {model_name} on-demand")
        logger.info(f"📋 Model input shape: {model.input_shape}")  
        logger.info(f"📋 Model output shape: {model.output_shape}")
        
        return model
        
    except Exception as e:
        logger.error(f"❌ Failed to load {model_name} on-demand: {str(e)}")
        logger.error(f"📍 Full traceback:")
        logger.error(traceback.format_exc())
        return None

def load_model_info():
    """Load model information file"""
    global model_info
    
    try:
        models_dir = 'models'
        if not os.path.exists(models_dir):
            logger.error(f"❌ Models directory not found: {models_dir}")
            return
        
        files_in_models = os.listdir(models_dir)
        pkl_files = [f for f in files_in_models if f.endswith('.pkl')]
        
        if pkl_files:
            info_path = f"{models_dir}/{pkl_files[0]}"
            try:
                with open(info_path, 'rb') as f:
                    model_info = pickle.load(f)
                logger.info(f"✅ Loaded model information from {info_path}")
            except Exception as e:
                logger.warning(f"⚠️ Could not load model info: {str(e)}")
        else:
            logger.warning("⚠️ No model info (.pkl) file found")
            
    except Exception as e:
        logger.error(f"❌ Error loading model info: {str(e)}")

def get_available_models():
    """Get list of available models based on files in models directory"""
    try:
        models_dir = 'models'
        if not os.path.exists(models_dir):
            return []
        
        files_in_models = os.listdir(models_dir)
        h5_files = [f for f in files_in_models if f.endswith('.h5')]
        
        available = []
        for file in h5_files:
            if 'cnn_3class' in file and 'cnn_3class' not in available:
                available.append('cnn_3class')
            elif 'bilstm_3class' in file and 'bilstm_3class' not in available:
                available.append('bilstm_3class')  
            elif 'cnn_binary' in file and 'cnn_binary' not in available:
                available.append('cnn_binary')
            elif 'bilstm_binary' in file and 'bilstm_binary' not in available:
                available.append('bilstm_binary')
        
        return available
        
    except Exception as e:
        logger.error(f"❌ Error getting available models: {str(e)}")
        return []

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
        'available_models': get_available_models(),
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
            'available_models': get_available_models(),
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

@app.route('/debug', methods=['GET'])
def debug_info():
    """Debug endpoint to check file system and model loading"""
    try:
        debug_data = {
            'current_directory': os.getcwd(),
            'models_directory_exists': os.path.exists('models'),
            'files_in_current_dir': os.listdir('.') if os.path.exists('.') else [],
            'loaded_models_count': len(models),
            'loaded_model_names': list(models.keys())
        }
        
        # Check models directory
        if os.path.exists('models'):
            debug_data['files_in_models_dir'] = os.listdir('models')
            
            # Check file sizes
            file_sizes = {}
            for file in debug_data['files_in_models_dir']:
                if file.endswith(('.h5', '.pkl')):
                    file_path = os.path.join('models', file)
                    size_mb = os.path.getsize(file_path) / (1024 * 1024)
                    file_sizes[file] = f"{size_mb:.2f} MB"
            debug_data['file_sizes'] = file_sizes
        else:
            debug_data['files_in_models_dir'] = 'models directory not found'
        
        return jsonify(debug_data)
        
    except Exception as e:
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500

@app.route('/test-load/<model_name>', methods=['GET'])
def test_load_single_model(model_name):
    """Test loading a single model for debugging"""
    try:
        models_dir = 'models'
        
        # Find the file
        model_file = None
        if os.path.exists(models_dir):
            files = os.listdir(models_dir)
            for file in files:
                if model_name in file and file.endswith('.h5'):
                    model_file = os.path.join(models_dir, file)
                    break
        
        if not model_file:
            return jsonify({'error': f'Model file for {model_name} not found'}), 404
        
        # Try to load
        logger.info(f"🧪 Testing load of {model_name} from {model_file}")
        
        file_size = os.path.getsize(model_file) / (1024 * 1024)
        logger.info(f"📊 File size: {file_size:.2f} MB")
        
        # Attempt to load
        test_model = load_model(model_file)
        
        result = {
            'status': 'success',
            'model_name': model_name,
            'file_path': model_file,
            'file_size_mb': f"{file_size:.2f}",
            'input_shape': str(test_model.input_shape),
            'output_shape': str(test_model.output_shape),
            'parameters': test_model.count_params()
        }
        
        # Don't keep the test model in memory
        del test_model
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"❌ Test load failed for {model_name}: {str(e)}")
        return jsonify({
            'status': 'error',
            'model_name': model_name,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

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
        
        # Load model on demand
        model = load_model_on_demand(model_type)
        if model is None:
            available_models = get_available_models()
            return jsonify({
                'error': f'Model {model_type} not available',
                'available_models': available_models
            }), 400
        
        # Preprocess input
        features = preprocess_input(data['features'])
        
        # Make prediction
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
        
        # Load model on demand
        model = load_model_on_demand(model_type)
        if model is None:
            return jsonify({'error': f'Model {model_type} not loaded'}), 500
        
        # Preprocess and predict
        features = preprocess_input(data['features'])
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
        
        # Load model on demand
        model = load_model_on_demand(model_type)
        if model is None:
            return jsonify({'error': f'Model {model_type} not available'}), 400
        
        # Preprocess batch input
        features = preprocess_input(data['features'])
        
        # Make batch predictions
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
    # Initialize API with on-demand model loading
    logger.info("🚀 Starting Epilepsy Detection API with On-Demand Model Loading...")
    logger.info(f"🐍 Python version: {os.sys.version}")
    logger.info(f"🧠 TensorFlow version: {tf.__version__}")
    
    try:
        # Load model info only (lightweight)
        load_model_info()
        available = get_available_models()
        logger.info(f"📋 Available models: {available}")
        logger.info("💡 Models will be loaded on-demand to save memory")
    except Exception as e:
        logger.error(f"❌ Critical error during initialization: {str(e)}")
        logger.error(traceback.format_exc())
    
    # Run the app
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)