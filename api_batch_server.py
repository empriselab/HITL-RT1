#!/usr/bin/env python3
"""
Flask API server for RT-1 model with batch processing and Monte Carlo dropout support.
With ngrok integration.
"""

import os
import json
import base64
import io
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS
import gin
from tensor2robot.utils import tensorspec_utils
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import time_step as ts
import sequence_agent
import transformer_network
from test_checkpoint_loading import custom_load_checkpoint

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Global variables
agent = None
config_loaded = False

def create_bbox_action_spec():
    action_spec = tensorspec_utils.TensorSpecStruct()
    action_spec.bbox = tensor_spec.BoundedTensorSpec((4,), dtype=tf.float32, minimum=0.0, maximum=1.0, name='bbox')
    return action_spec

def create_observation_spec(image_size=236):
    observation_spec = tensorspec_utils.TensorSpecStruct()
    observation_spec.image = tensor_spec.BoundedTensorSpec([
        image_size, image_size, 3], dtype=tf.float32, name='image', minimum=0.0, maximum=1.0)
    observation_spec.natural_language_embedding = tensor_spec.TensorSpec(
        shape=[512], dtype=tf.float32, name='natural_language_embedding')
    return observation_spec

# Action-Food mapping to embeddings
ACTION_FOOD_EMBEDDING_MAP = {
    ('skewering', 'banana'): 'embeddings/skewering_banana.npy',
    ('skewering', 'blueberry'): 'embeddings/skewering_blueberry.npy',
    ('skewering', 'broccoli'): 'embeddings/skewering_broccoli.npy',
    ('skewering', 'brownie'): 'embeddings/skewering_brownie.npy',
    ('skewering', 'cantaloupe'): 'embeddings/skewering_cantaloupe.npy',
    ('skewering', 'cherry_tomato'): 'embeddings/skewering_cherry_tomato.npy',
    ('skewering', 'chicken'): 'embeddings/skewering_chicken.npy',
    ('skewering', 'fettucine'): 'embeddings/skewering_fettucine.npy',
    ('skewering', 'grape'): 'embeddings/skewering_grape.npy',
    ('skewering', 'green_bean'): 'embeddings/skewering_green_bean.npy',
    ('skewering', 'honeydew'): 'embeddings/skewering_honeydew.npy',
    ('skewering', 'lettuce'): 'embeddings/skewering_lettuce.npy',
    ('skewering', 'mac_and_cheese'): 'embeddings/skewering_mac_and_cheese.npy',
    ('skewering', 'mashed_potato'): 'embeddings/skewering_mashed_potato.npy',
    ('skewering', 'oatmeal'): 'embeddings/skewering_oatmeal.npy',
    ('skewering', 'pineapple'): 'embeddings/skewering_pineapple.npy',
    ('skewering', 'rice'): 'embeddings/skewering_rice.npy',
    ('skewering', 'sausage'): 'embeddings/skewering_sausage.npy',
    ('skewering', 'spaghetti'): 'embeddings/skewering_spaghetti.npy',
    ('skewering', 'strawberry'): 'embeddings/skewering_strawberry.npy',
    ('skewering', 'watermelon'): 'embeddings/skewering_watermelon.npy',
    ('skewering', 'meatball'): 'embeddings/skewering_meatball.npy',
    
    ('scooping', 'banana'): 'embeddings/scoop_banana.npy',
    ('scooping', 'blueberry'): 'embeddings/scoop_blueberry.npy',
    ('scooping', 'broccoli'): 'embeddings/scoop_broccoli.npy',
    ('scooping', 'brownie'): 'embeddings/scoop_brownie.npy',
    ('scooping', 'cantaloupe'): 'embeddings/scoop_cantaloupe.npy',
    ('scooping', 'cherry_tomato'): 'embeddings/scoop_cherry_tomato.npy',
    ('scooping', 'chicken'): 'embeddings/scoop_chicken.npy',
    ('scooping', 'fettucine'): 'embeddings/scoop_fettucine.npy',
    ('scooping', 'grape'): 'embeddings/scoop_grape.npy',
    ('scooping', 'green_bean'): 'embeddings/scoop_green_bean.npy',
    ('scooping', 'honeydew'): 'embeddings/scoop_honeydew.npy',
    ('scooping', 'lettuce'): 'embeddings/scoop_lettuce.npy',
    ('scooping', 'mac_and_cheese'): 'embeddings/scoop_mac_and_cheese.npy',
    ('scooping', 'mashed_potato'): 'embeddings/scoop_mashed_potato.npy',
    ('scooping', 'oatmeal'): 'embeddings/scoop_oatmeal.npy',
    ('scooping', 'pineapple'): 'embeddings/scoop_pineapple.npy',
    ('scooping', 'rice'): 'embeddings/scoop_rice.npy',
    ('scooping', 'sausage'): 'embeddings/scoop_sausage.npy',
    ('scooping', 'spaghetti'): 'embeddings/scoop_spaghetti.npy',
    ('scooping', 'strawberry'): 'embeddings/scoop_strawberry.npy',
    ('scooping', 'watermelon'): 'embeddings/scoop_watermelon.npy',
    ('scooping', 'meatball'): 'embeddings/scoop_meatball.npy',
    
    ('twirling', 'banana'): 'embeddings/twirling_banana.npy',
    ('twirling', 'blueberry'): 'embeddings/twirling_blueberry.npy',
    ('twirling', 'broccoli'): 'embeddings/twirling_broccoli.npy',
    ('twirling', 'brownie'): 'embeddings/twirling_brownie.npy',
    ('twirling', 'cantaloupe'): 'embeddings/twirling_cantaloupe.npy',
    ('twirling', 'cherry_tomato'): 'embeddings/twirling_cherry_tomato.npy',
    ('twirling', 'chicken'): 'embeddings/twirling_chicken.npy',
    ('twirling', 'fettucine'): 'embeddings/twirling_fettucine.npy',
    ('twirling', 'grape'): 'embeddings/twirling_grape.npy',
    ('twirling', 'green_bean'): 'embeddings/twirling_green_bean.npy',
    ('twirling', 'honeydew'): 'embeddings/twirling_honeydew.npy',
    ('twirling', 'lettuce'): 'embeddings/twirling_lettuce.npy',
    ('twirling', 'mac_and_cheese'): 'embeddings/twirling_mac_and_cheese.npy',
    ('twirling', 'mashed_potato'): 'embeddings/twirling_mashed_potato.npy',
    ('twirling', 'oatmeal'): 'embeddings/twirling_oatmeal.npy',
    ('twirling', 'pineapple'): 'embeddings/twirling_pineapple.npy',
    ('twirling', 'rice'): 'embeddings/twirling_rice.npy',
    ('twirling', 'sausage'): 'embeddings/twirling_sausage.npy',
    ('twirling', 'spaghetti'): 'embeddings/twirling_spaghetti.npy',
    ('twirling', 'strawberry'): 'embeddings/twirling_strawberry.npy',
    ('twirling', 'watermelon'): 'embeddings/twirling_watermelon.npy',
    ('twirling', 'meatball'): 'embeddings/twirling_meatball.npy',
}

def get_embedding_for_action_food(action, food_type):
    """Get embedding for a specific action-food combination."""
    # Normalize inputs
    action = action.lower().strip()
    food_type = food_type.lower().strip()
    
    # Handle common variations
    if action in ['scoop', 'scooping']:
        action = 'scooping'
    elif action in ['skewer', 'skewering']:
        action = 'skewering'
    elif action in ['twirl', 'twirling']:
        action = 'twirling'
    
    # Handle food variations
    food_variations = {
        'green_bean': 'green_bean',
        'green-bean': 'green_bean',
        'green bean': 'green_bean',
        'green_beans': 'green_bean',  # For consistency, use singular form
        'pasta': 'fettucine',
        'noodles': 'fettucine',
        'spaghetti_noodles': 'spaghetti',
        'spaghetti noodles': 'spaghetti',
        'mashed_potatoes': 'mashed_potato',
        'mashed potatoes': 'mashed_potato',
        'cherry_tomatoes': 'cherry_tomato',
        'cherry tomato': 'cherry_tomato',
        'cherry tomatoes': 'cherry_tomato',
        'mac_and_cheese': 'mac_and_cheese',
        'mac and cheese': 'mac_and_cheese',
        'macaroni': 'mac_and_cheese',
        'blueberries': 'blueberry',
        'grapes': 'grape',
        'strawberries': 'strawberry',
        'brownies': 'brownie',
    }
    food_type = food_variations.get(food_type, food_type)
    
    # Look up embedding
    key = (action, food_type)
    embedding_path = ACTION_FOOD_EMBEDDING_MAP.get(key)
    
    if embedding_path and os.path.exists(embedding_path):
        print(f"[INFO] Loading embedding for {action} + {food_type}: {embedding_path}")
        try:
            embedding = np.load(embedding_path)
            if embedding.shape == (512,):
                print(f"[INFO] Successfully loaded embedding, shape: {embedding.shape}")
                return embedding
            else:
                print(f"[WARNING] Embedding shape mismatch: {embedding.shape}, expected (512,)")
        except Exception as e:
            print(f"[WARNING] Failed to load embedding {embedding_path}: {e}")
    else:
        print(f"[WARNING] No embedding found for {action} + {food_type}")
    
    # Fallback to deterministic dummy embedding
    return text_to_embedding(f"{action} {food_type}")

def text_to_embedding(text):
    """Create dummy text embedding (512-dimensional) as fallback."""
    print(f"[INFO] Creating dummy embedding for text: '{text}'")
    
    # Create a deterministic dummy embedding based on text hash
    np.random.seed(hash(text) % 2**32)  # Deterministic based on text
    embedding = np.random.normal(0, 1, 512).astype(np.float32)
    embedding = embedding / np.linalg.norm(embedding)  # Normalize to unit length
    
    print(f"[INFO] Dummy embedding created successfully, shape: {embedding.shape}")
    return embedding

def load_and_preprocess_image(image_data, image_size=236):
    """Load and preprocess image from base64 or bytes data."""
    try:
        # Try to decode as base64 first
        if isinstance(image_data, str):
            image_bytes = base64.b64decode(image_data)
        else:
            image_bytes = image_data
            
        # Load image using PIL
        pil_image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB if needed
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        # Resize to target size
        pil_image = pil_image.resize((image_size, image_size), Image.Resampling.LANCZOS)
        
        # Convert to numpy array
        image = np.array(pil_image)
        
        # Convert to float32 and normalize to [0, 1] for model input
        image = image.astype(np.float32) / 255.0
        
        return image
    except Exception as e:
        raise ValueError(f"Failed to load image: {e}")

def calculate_batch_statistics(predictions):
    """Calculate statistics across batch predictions."""
    predictions = np.array(predictions)  # Shape: [batch_size, 4]
    
    # Basic statistics
    mean_pred = np.mean(predictions, axis=0)
    std_pred = np.std(predictions, axis=0)
    
    # Calculate total variance (sum of variances across all coordinates)
    total_variance = np.sum(np.var(predictions, axis=0))
    
    # Calculate centroid variance
    centroids = (predictions[:, :2] + predictions[:, 2:]) / 2  # [batch_size, 2]
    centroid_variance = np.sum(np.var(centroids, axis=0))
    
    # Calculate size variance
    sizes = predictions[:, 2:] - predictions[:, :2]  # [batch_size, 2]
    size_variance = np.sum(np.var(sizes, axis=0))
    
    # Calculate distance variance (Euclidean distance from origin)
    distances = np.sqrt(np.sum(predictions**2, axis=1))  # [batch_size]
    distance_variance = np.var(distances)
    
    return {
        'mean': mean_pred.tolist(),
        'std': std_pred.tolist(),
        'total_variance': float(total_variance),
        'centroid_variance': float(centroid_variance),
        'size_variance': float(size_variance),
        'distance_variance': float(distance_variance),
        'num_samples': len(predictions)
    }

def initialize_model(checkpoint_path, config_file, loss_type='smooth_l1'):
    """Initialize the model and load checkpoint."""
    global agent, config_loaded
    
    if agent is not None:
        return agent
    
    print("[INFO] Initializing model...")
    
    # Parse gin config
    gin.parse_config_file(config_file)
    
    # Create specs and agent
    observation_spec = create_observation_spec()
    action_spec = create_bbox_action_spec()
    
    print("[INFO] Building agent model...")
    with tf.device('/GPU:0'):
        time_step_spec = ts.time_step_spec(observation_spec=observation_spec)
        agent = sequence_agent.SequenceAgent(
            time_step_spec=time_step_spec,
            action_spec=action_spec,
            actor_network=transformer_network.TransformerNetwork,
            actor_optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            train_step_counter=tf.compat.v1.train.get_or_create_global_step(),
            time_sequence_length=1,
            debug_summaries=False,
            loss_type=loss_type
        )
    
    # Build model variables
    print("[INFO] Building model variables...")
    with tf.device('/GPU:0'):
        dummy_observations = {}
        for key, spec in observation_spec.items():
            if hasattr(spec, 'shape') and hasattr(spec, 'dtype'):
                dummy_observations[key] = tf.zeros([1] + list(spec.shape), dtype=spec.dtype)
            else:
                dummy_observations[key] = tf.zeros([1, 512], dtype=tf.float32)
        initial_state = agent._actor_network.get_initial_state(batch_size=1)
        _ = agent._actor_network(dummy_observations, initial_state, training=False)
    
    # Load checkpoint
    if checkpoint_path:
        print(f"[INFO] Loading checkpoint: {checkpoint_path}")
        custom_load_checkpoint(agent, checkpoint_path)
        print("[INFO] Checkpoint loaded successfully.")
    
    config_loaded = True
    return agent

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': agent is not None,
        'config_loaded': config_loaded
    })

@app.route('/available_combinations', methods=['GET'])
def available_combinations():
    """Get available action-food combinations."""
    combinations = []
    actions = set()
    food_types = set()
    
    for action, food_type in ACTION_FOOD_EMBEDDING_MAP.keys():
        combinations.append({'action': action, 'food_type': food_type})
        actions.add(action)
        food_types.add(food_type)
    
    return jsonify({
        'combinations': combinations,
        'available_actions': sorted(list(actions)),
        'available_food_types': sorted(list(food_types)),
        'total_combinations': len(combinations)
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Single prediction endpoint (backward compatibility)."""
    try:
        # Check if model is loaded
        if agent is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        # Get request data
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        # Extract parameters
        image_data = data.get('image')
        action = data.get('action', 'skewering')
        food_type = data.get('food_type', 'food_item')
        text_instruction = data.get('text', f'{action} {food_type}')  # Fallback text
        
        if not image_data:
            return jsonify({'error': 'No image data provided'}), 400
        
        # Load and preprocess image
        image = load_and_preprocess_image(image_data)
        
        # Get embedding based on action and food type
        if action and food_type:
            text_embedding = get_embedding_for_action_food(action, food_type)
            print(f"[INFO] Using action-food embedding: {action} + {food_type}")
        else:
            # Fallback to text-based embedding
            text_embedding = text_to_embedding(text_instruction)
            print(f"[INFO] Using text-based embedding: {text_instruction}")
        
        # Prepare input for inference
        image_batch = np.expand_dims(np.expand_dims(image, axis=0), axis=0)  # [1, 1, H, W, C]
        text_embedding_batch = np.expand_dims(np.expand_dims(text_embedding, axis=0), axis=0)  # [1, 1, 512]
        
        # Convert to tensors
        image_tensor = tf.convert_to_tensor(image_batch, dtype=tf.float32)
        text_embedding_tensor = tf.convert_to_tensor(text_embedding_batch, dtype=tf.float32)
        
        # Create observations
        observations = {
            'image': image_tensor,
            'natural_language_embedding': text_embedding_tensor
        }
        
        # Run inference
        with tf.device('/GPU:0'):
            # Create TimeStep
            time_steps = ts.TimeStep(
                step_type=tf.expand_dims(tf.constant([ts.StepType.FIRST], dtype=tf.int32), axis=1),
                reward=tf.zeros([1, 1], dtype=tf.float32),
                discount=tf.ones([1, 1], dtype=tf.float32),
                observation=observations
            )
            
            # Get policy action
            policy_state = agent.policy.get_initial_state(batch_size=1)
            action_step = agent.policy.action(time_steps, policy_state=policy_state)
            
            # Get predicted bbox
            pred_bbox = action_step.action.bbox.numpy() if hasattr(action_step.action, 'bbox') else action_step.action.numpy()
            pred_bbox = pred_bbox[0] if pred_bbox.ndim == 2 else pred_bbox
        
        # Calculate additional metrics
        skewer_x = (pred_bbox[0] + pred_bbox[2]) / 2
        skewer_y = (pred_bbox[1] + pred_bbox[3]) / 2
        direction = np.degrees(np.arctan2(pred_bbox[3] - pred_bbox[1], 
                                        pred_bbox[2] - pred_bbox[0]))
        
        # Return results
        result = {
            'success': True,
            'prediction': {
                'start_x': float(pred_bbox[0]),
                'start_y': float(pred_bbox[1]),
                'end_x': float(pred_bbox[2]),
                'end_y': float(pred_bbox[3]),
                'skewer_x': float(skewer_x),
                'skewer_y': float(skewer_y),
                'direction_degrees': float(direction)
            },
            'input_action': action,
            'input_food_type': food_type,
            'input_text': text_instruction
        }
        
        return jsonify(result)
        
    except Exception as e:
        print(f"[ERROR] Prediction failed: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    """Batch prediction endpoint with Monte Carlo dropout."""
    try:
        # Check if model is loaded
        if agent is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        # Get request data
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        # Extract parameters
        image_data = data.get('image')
        action = data.get('action', 'skewering')
        food_type = data.get('food_type', 'food_item')
        text_instruction = data.get('text', f'{action} {food_type}')
        batch_size = data.get('batch_size', 16)  # Default to 16 for Monte Carlo
        enable_dropout = data.get('enable_dropout', True)  # Enable dropout by default for uncertainty
        
        if not image_data:
            return jsonify({'error': 'No image data provided'}), 400
        
        if batch_size < 1 or batch_size > 64:  # Reasonable limits
            return jsonify({'error': 'batch_size must be between 1 and 64'}), 400
        
        print(f"[INFO] Running batch prediction with batch_size={batch_size}, enable_dropout={enable_dropout}")
        
        # Load and preprocess image
        image = load_and_preprocess_image(image_data)
        
        # Get embedding based on action and food type
        if action and food_type:
            text_embedding = get_embedding_for_action_food(action, food_type)
            print(f"[INFO] Using action-food embedding: {action} + {food_type}")
        else:
            # Fallback to text-based embedding
            text_embedding = text_to_embedding(text_instruction)
            print(f"[INFO] Using text-based embedding: {text_instruction}")
        
        # Prepare batch input - replicate the same image and embedding batch_size times
        # Shape: [batch_size, 1, H, W, C] for image, [batch_size, 1, 512] for embedding
        image_batch = np.tile(np.expand_dims(np.expand_dims(image, axis=0), axis=0), [batch_size, 1, 1, 1, 1])
        text_embedding_batch = np.tile(np.expand_dims(np.expand_dims(text_embedding, axis=0), axis=0), [batch_size, 1, 1])
        
        # Convert to tensors
        image_tensor = tf.convert_to_tensor(image_batch, dtype=tf.float32)
        text_embedding_tensor = tf.convert_to_tensor(text_embedding_batch, dtype=tf.float32)
        
        # Create observations
        observations = {
            'image': image_tensor,
            'natural_language_embedding': text_embedding_tensor
        }
        
        # Run batch inference
        with tf.device('/GPU:0'):
            # Create TimeStep for batch
            time_steps = ts.TimeStep(
                step_type=tf.constant([[ts.StepType.FIRST]] * batch_size, dtype=tf.int32),
                reward=tf.zeros([batch_size, 1], dtype=tf.float32),
                discount=tf.ones([batch_size, 1], dtype=tf.float32),
                observation=observations
            )
            
            # Set training mode for dropout if enabled
            if enable_dropout:
                agent.policy.set_training(True)
                print("[INFO] Dropout enabled for uncertainty quantification")
            else:
                agent.policy.set_training(False)
                print("[INFO] Dropout disabled for deterministic inference")
            
            # Get policy action for entire batch
            policy_state = agent.policy.get_initial_state(batch_size=batch_size)
            action_step = agent.policy.action(time_steps, policy_state=policy_state)
            
            # Get predicted bboxes for entire batch
            pred_bboxes = action_step.action.bbox.numpy() if hasattr(action_step.action, 'bbox') else action_step.action.numpy()
            # Shape: [batch_size, 4]
            
            # Reset training mode
            agent.policy.set_training(False)
        
        # Calculate statistics across the batch
        stats = calculate_batch_statistics(pred_bboxes)
        
        # Calculate additional metrics for mean prediction
        mean_bbox = stats['mean']
        skewer_x = (mean_bbox[0] + mean_bbox[2]) / 2
        skewer_y = (mean_bbox[1] + mean_bbox[3]) / 2
        direction = np.degrees(np.arctan2(mean_bbox[3] - mean_bbox[1], 
                                        mean_bbox[2] - mean_bbox[0]))
        
        # Return results
        result = {
            'success': True,
            'batch_size': batch_size,
            'enable_dropout': enable_dropout,
            'predictions': pred_bboxes.tolist(),  # All individual predictions
            'statistics': stats,
            'mean_prediction': {
                'start_x': float(mean_bbox[0]),
                'start_y': float(mean_bbox[1]),
                'end_x': float(mean_bbox[2]),
                'end_y': float(mean_bbox[3]),
                'skewer_x': float(skewer_x),
                'skewer_y': float(skewer_y),
                'direction_degrees': float(direction)
            },
            'input_action': action,
            'input_food_type': food_type,
            'input_text': text_instruction
        }
        
        return jsonify(result)
        
    except Exception as e:
        print(f"[ERROR] Batch prediction failed: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/load_model', methods=['POST'])
def load_model_endpoint():
    """Endpoint to load/reload the model."""
    try:
        data = request.get_json()
        checkpoint_path = data.get('checkpoint_path', 'output_checkpoints25-smooth_l1')
        config_file = data.get('config_file', 'configs/transformer_mixin.gin')
        loss_type = data.get('loss_type', 'smooth_l1')
        
        initialize_model(checkpoint_path, config_file, loss_type)
        
        return jsonify({
            'success': True,
            'message': f'Model loaded from {checkpoint_path}'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Initialize model on startup
    checkpoint_path = 'output_checkpoints25-smooth_l1'
    config_file = os.environ.get('CONFIG_FILE', 'configs/transformer_mixin.gin')
    loss_type = os.environ.get('LOSS_TYPE', 'smooth_l1')
    
    try:
        initialize_model(checkpoint_path, config_file, loss_type)
        print("[INFO] Model initialized successfully!")
    except Exception as e:
        print(f"[ERROR] Failed to initialize model: {e}")
    
    # Run the Flask app
    port = int(os.environ.get('PORT', 5000))
    
    # Check if ngrok should be used
    use_ngrok = os.environ.get('USE_NGROK', 'false').lower() == 'true'
    
    if use_ngrok:
        try:
            from pyngrok import ngrok
            print("[INFO] Starting ngrok tunnel...")
            
            # Check if authtoken is set
            authtoken = os.environ.get('NGROK_AUTHTOKEN')
            if authtoken:
                ngrok.set_auth_token(authtoken)
                print("[INFO] ngrok authtoken configured")
            else:
                print("[WARNING] NGROK_AUTHTOKEN not set. Please set it or sign up at https://dashboard.ngrok.com/signup")
                print("[INFO] Running without ngrok tunnel...")
                use_ngrok = False
            
            if use_ngrok:
                public_url = ngrok.connect(port)
                print(f"[INFO] Public URL: {public_url}")
                print(f"[INFO] You can now access your API at: {public_url}")
                
        except ImportError:
            print("[WARNING] pyngrok not installed. Install with: pip install pyngrok")
            use_ngrok = False
        except Exception as e:
            print(f"[WARNING] ngrok failed to start: {e}")
            print("[INFO] Running without ngrok tunnel...")
            use_ngrok = False
    
    app.run(host='0.0.0.0', port=port, debug=False)