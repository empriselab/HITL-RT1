#!/usr/bin/env python3
"""Utility functions for RT-1 training and model management."""

import os
import gc
import logging
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from absl import logging as absl_logging
from PIL import Image, ImageEnhance
import random

import json
import glob
import copy
import math

import sequence_agent
import transformer_network
from tokenizers import action_tokenizer
from tensor2robot.utils import tensorspec_utils
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.trajectories import policy_step
from tf_agents.trajectories import trajectory
from test_checkpoint_loading import custom_load_checkpoint, map_model_name_to_checkpoint_name

# =============================================================================
# DATA PROCESSING & DATASET UTILS
# =============================================================================

def create_bbox_action_spec():
    """Create action specification for 4D bbox prediction task."""
    action_spec = tensorspec_utils.TensorSpecStruct()
    
    # 4D bounding box: (x1, y1, x2, y2) - normalized to [0, 1]
    action_spec.bbox = tensor_spec.BoundedTensorSpec(
        (4,), dtype=tf.float32, minimum=0.0, maximum=1.0, name='bbox')
    
    return action_spec

def create_observation_spec(image_size=236):
    """Create observation specification for single image input."""
    observation_spec = tensorspec_utils.TensorSpecStruct()
    
    # Single image input - uint8 format for RT-1
    observation_spec.image = tensor_spec.BoundedTensorSpec(
        [image_size, image_size, 3],
        dtype=tf.uint8,
        name='image',
        minimum=0,
        maximum=255)
    
    # Natural language embedding
    observation_spec.natural_language_embedding = tensor_spec.TensorSpec(
        shape=[512],
        dtype=tf.float32,
        name='natural_language_embedding')
    
    return observation_spec

def create_npz_dataset(npz_path, batch_size, split_ratio=0.8, shuffle_buffer=1000, split_file=None, 
                      apply_rotation_augmentation=False, apply_lighting_augmentation=False, image_size=236, loss_type='cross_entropy'):
    """Create train and validation datasets from npz file with optional rotation augmentation."""
    logging.info(f"Loading dataset from {npz_path}...")
    data = np.load(npz_path)
    num_samples = data['images'].shape[0]
    logging.info(f"Dataset loaded. Number of samples: {num_samples}")

    # Check if split file exists
    if split_file and os.path.exists(split_file):
        logging.info(f"Loading existing split from {split_file}")
        split_data = np.load(split_file)
        train_idx = split_data['train_idx']
        val_idx = split_data['val_idx']
        logging.info(f"Loaded split: {len(train_idx)} train, {len(val_idx)} validation samples")
    else:
        # Create new split
        logging.info("No split file found, creating new split...")
        indices = np.arange(num_samples)
        np.random.shuffle(indices)
        split_idx = int(num_samples * split_ratio)
        train_idx, val_idx = indices[:split_idx], indices[split_idx:]
        
        # Save split for future use
        if split_file:
            logging.info(f"Saving split to {split_file}")
            os.makedirs(os.path.dirname(split_file), exist_ok=True)
            np.savez(split_file, train_idx=train_idx, val_idx=val_idx)
    
    logging.info(f"Train samples: {len(train_idx)}, Validation samples: {len(val_idx)}")

    def rotate_bounding_box(x1, y1, x2, y2, rotation_deg, img_size=236):
        """Rotate bounding box coordinates for 90-degree rotations."""
        # Convert to pixel coordinates
        x1_px = x1 * img_size
        y1_px = y1 * img_size
        x2_px = x2 * img_size
        y2_px = y2 * img_size
        
        if rotation_deg == 0:
            return x1, y1, x2, y2
        elif rotation_deg == 90:
            # 90° clockwise: (x, y) -> (y, 1-x)
            new_x1 = y1_px
            new_y1 = img_size - x2_px
            new_x2 = y2_px
            new_y2 = img_size - x1_px
        elif rotation_deg == 180:
            # 180°: (x, y) -> (1-x, 1-y)
            new_x1 = img_size - x2_px
            new_y1 = img_size - y2_px
            new_x2 = img_size - x1_px
            new_y2 = img_size - y1_px
        elif rotation_deg == 270:
            # 270° clockwise: (x, y) -> (1-y, x)
            new_x1 = img_size - y2_px
            new_y1 = x1_px
            new_x2 = img_size - y1_px
            new_y2 = x2_px
        else:
            raise ValueError(f"Unsupported rotation angle: {rotation_deg}")
        
        # Convert back to normalized coordinates
        return (new_x1 / img_size, new_y1 / img_size, 
                new_x2 / img_size, new_y2 / img_size)



    def make_dataset(idx, apply_lighting_augmentation=False):
        # Get the data
        images = data['images'][idx]
        bboxes = data['actions'][idx]
        embeddings = data['embeddings'][idx]
        
        # Convert to uint8 if needed
        if images.dtype == np.float32 and images.max() <= 1.0:
            images = (images * 255).astype(np.uint8)
            logging.info(f"Converted images from float32 (0-1) to uint8 (0-255)")
        
        # Handle bbox data based on loss type
        if loss_type in ('l1', 'smooth_l1'):
            # For L1 loss, ensure bboxes are continuous coordinates [0,1]
            if bboxes.dtype != np.float32:
                bboxes = bboxes.astype(np.float32)
            # Ensure coordinates are in [0,1] range
            bboxes = np.clip(bboxes, 0.0, 1.0)
            logging.info(f"L1 loss mode: bboxes shape {bboxes.shape}, dtype {bboxes.dtype}, range [{bboxes.min():.3f}, {bboxes.max():.3f}]")
        elif loss_type == 'cross_entropy':
            # For cross-entropy, bboxes should be discrete tokens
            # This assumes the data is already tokenized
            logging.info(f"Cross-entropy loss mode: bboxes shape {bboxes.shape}, dtype {bboxes.dtype}")
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
        
        # Handle rotation augmentation
        if apply_rotation_augmentation:
            # Rotation requires generator approach (4x dataset multiplication)
            def data_generator():
                rotations = [0, 90, 180, 270]
                
                for sample_idx in range(len(idx)):
                    image = images[sample_idx]
                    bbox = bboxes[sample_idx]
                    embedding = embeddings[sample_idx]
                    
                    for rotation_deg in rotations:
                        # Rotate image
                        if rotation_deg == 0:
                            processed_image = image
                        else:
                            pil_image = Image.fromarray(image.astype(np.uint8))
                            rotated_pil = pil_image.rotate(-rotation_deg)
                            processed_image = np.array(rotated_pil)
                        
                        # Apply lighting augmentation if enabled
                        if apply_lighting_augmentation:
                            try:
                                processed_image = apply_fast_lighting_augmentation(processed_image)
                            except Exception as e:
                                print(f"ERROR in rotation generator lighting augmentation: {e}")
                                processed_image = processed_image
                        
                        # Rotate bounding box coordinates if rotation is applied
                        if rotation_deg == 0:
                            processed_bbox = bbox
                        else:
                            if loss_type == 'l1':
                                # For L1 loss, rotate continuous coordinates
                                x1, y1, x2, y2 = bbox
                                rotated_x1, rotated_y1, rotated_x2, rotated_y2 = rotate_bounding_box(
                                    x1, y1, x2, y2, rotation_deg, image_size
                                )
                                processed_bbox = np.array([rotated_x1, rotated_y1, rotated_x2, rotated_y2], dtype=np.float32)
                            else:
                                # For cross-entropy, assume bbox contains discrete tokens
                                # Rotation might need special handling for discrete tokens
                                processed_bbox = bbox
                        
                        yield {
                            'image': processed_image,
                            'bbox': processed_bbox,
                            'natural_language_embedding': embedding
                        }
            
            # Create dataset from generator for rotation
            bbox_dtype = tf.float32 if loss_type in ('l1', 'smooth_l1') else tf.int32
            dataset = tf.data.Dataset.from_generator(
                data_generator,
                output_signature={
                    'image': tf.TensorSpec(shape=(image_size, image_size, 3), dtype=tf.uint8),
                    'bbox': tf.TensorSpec(shape=(4,), dtype=bbox_dtype),
                    'natural_language_embedding': tf.TensorSpec(shape=(512,), dtype=tf.float32)
                }
            )
        else:
            # No rotation - use fast approach for lighting augmentation
            if apply_lighting_augmentation:
                # Apply lighting augmentation to all images at once (fast)
                logging.info("Applying fast lighting augmentation...")
                augmented_images = []
                for i, image in enumerate(images):
                    if i % 1000 == 0:
                        logging.info(f"Processing image {i}/{len(images)}")
                    augmented_image = apply_fast_lighting_augmentation(image)
                    augmented_images.append(augmented_image)
                images = np.array(augmented_images)
                logging.info(f"Fast augmentation complete. Shape: {images.shape}")
            
            # Create fast tensor-based dataset
            bbox_dtype = tf.float32 if loss_type in ('l1', 'smooth_l1') else tf.int32
            dataset = tf.data.Dataset.from_tensor_slices({
                'image': images,
                'bbox': bboxes,
                'natural_language_embedding': embeddings
            })
        
        return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    # Apply lighting augmentation only to training data, not validation data
    train_ds = make_dataset(train_idx, apply_lighting_augmentation=apply_lighting_augmentation)
    val_ds = make_dataset(val_idx, apply_lighting_augmentation=False)
    
    # Get final dataset sizes after augmentation
    augmentation_multiplier = 1
    if apply_rotation_augmentation:
        augmentation_multiplier *= 4
    if apply_lighting_augmentation:
        # Lighting augmentation doesn't multiply dataset size, it just adds variety
        logging.info("Lighting augmentation enabled - adds variety without multiplying dataset size")
    
    if augmentation_multiplier > 1:
        final_train_size = len(train_idx) * augmentation_multiplier
        final_val_size = len(val_idx) * augmentation_multiplier
        logging.info(f"After augmentation - Train: {final_train_size}, Val: {final_val_size}")
        return train_ds, val_ds, final_train_size, final_val_size
    else:
        return train_ds, val_ds, len(train_idx), len(val_idx)

# =============================================================================
# DATA AUGMENTATION UTILS
# =============================================================================

def apply_proper_lighting_augmentation(image):
    """
    Apply proper, subtle data augmentation using PIL ImageEnhance.
    This preserves color relationships while adding realistic lighting variations.
    
    Args:
        image: Input image as numpy array (uint8, 0-255)
        
    Returns:
        Augmented image as numpy array
    """
    try:
        
        # Convert numpy to PIL
        pil_image = Image.fromarray(image)
        
        # Apply subtle variations (50% chance each, like the example)
        if random.random() > 0.5:
            pil_image = ImageEnhance.Color(pil_image).enhance(random.uniform(0.8, 1.2))      # 80%-120% saturation
        if random.random() > 0.5:
            pil_image = ImageEnhance.Brightness(pil_image).enhance(random.uniform(0.8, 1.2)) # 80%-120% brightness
        if random.random() > 0.5:
            pil_image = ImageEnhance.Contrast(pil_image).enhance(random.uniform(0.8, 1.2))   # 80%-120% contrast
        if random.random() > 0.5:
            pil_image = ImageEnhance.Sharpness(pil_image).enhance(random.uniform(0.9, 1.1))  # 90%-110% sharpness
        
        return np.array(pil_image)
    except Exception as e:
        print(f"ERROR in apply_proper_lighting_augmentation: {e}")
        # Return original image if augmentation fails
        return image

def apply_fast_lighting_augmentation(image):
    """
    Apply fast, subtle data augmentation similar to SPANet approach.
    This is applied during data loading, not in a generator.
    """
    try:
        import random
        from PIL import Image, ImageEnhance
        
        # Convert to PIL
        pil_image = Image.fromarray(image)
        
        # Apply augmentation with 50% probability (like SPANet)
        if random.random() > 0.5:
            # Color (saturation) - similar to SPANet's range
            pil_image = ImageEnhance.Color(pil_image).enhance(random.uniform(0.8, 1.2))
        if random.random() > 0.5:
            # Brightness - similar to SPANet's range but more conservative
            pil_image = ImageEnhance.Brightness(pil_image).enhance(random.uniform(0.8, 1.2))
        if random.random() > 0.5:
            # Contrast - similar to SPANet's range but more conservative
            pil_image = ImageEnhance.Contrast(pil_image).enhance(random.uniform(0.8, 1.2))
        if random.random() > 0.5:
            # Sharpness - conservative range
            pil_image = ImageEnhance.Sharpness(pil_image).enhance(random.uniform(0.9, 1.1))
        
        return np.array(pil_image)
    except Exception as e:
        print(f"ERROR in fast lighting augmentation: {e}")
        return image

def create_fast_augmented_dataset(images, bboxes, embeddings, apply_lighting_augmentation=False):
    """
    Create a fast augmented dataset using vectorized operations.
    This avoids the slow generator approach.
    """
    if not apply_lighting_augmentation:
        # No augmentation - return original data
        return tf.data.Dataset.from_tensor_slices({
            'image': images,
            'bbox': bboxes,
            'natural_language_embedding': embeddings
        })
    
    # With augmentation - apply to all images at once
    print("Applying fast lighting augmentation to all images...")
    augmented_images = []
    
    for i, image in enumerate(images):
        if i % 1000 == 0:
            print(f"Processing image {i}/{len(images)}")
        augmented_image = apply_fast_lighting_augmentation(image)
        augmented_images.append(augmented_image)
    
    augmented_images = np.array(augmented_images)
    print(f"Augmentation complete. Shape: {augmented_images.shape}")
    
    return tf.data.Dataset.from_tensor_slices({
        'image': augmented_images,
        'bbox': bboxes,
        'natural_language_embedding': embeddings
    })

# =============================================================================
# MODEL & AGENT UTILS
# =============================================================================

def create_agent(learning_rate=0.0001, use_gaussian_smoothing=False, gaussian_std=2.0, gaussian_truncate=4.0, loss_type='cross_entropy'):
    """Create the sequence agent for bbox prediction with configurable loss type."""
    # Create specs
    observation_spec = create_observation_spec()
    action_spec = create_bbox_action_spec()
    
    # Create time step spec
    time_step_spec = ts.time_step_spec(observation_spec=observation_spec)
    
    # Create agent
    agent = sequence_agent.SequenceAgent(
        time_step_spec=time_step_spec,
        action_spec=action_spec,
        actor_network=transformer_network.TransformerNetwork,
        actor_optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        train_step_counter=tf.compat.v1.train.get_or_create_global_step(),
        time_sequence_length=1,  # Single image, so sequence length is 1
        debug_summaries=False,
        loss_type=loss_type)  # Pass loss_type to the network
    
    # Set loss type in the network
    network = agent._actor_network
    if loss_type == 'cross_entropy':
        logging.info("[LOSS MODE] Using SparseCategoricalCrossentropy: targets are integer class indices (no one-hot, no smoothing).")
    elif loss_type == 'l1':
        logging.info("[LOSS MODE] Using L1 loss: targets are continuous coordinates.")
    elif loss_type == 'smooth_l1':
        logging.info("[LOSS MODE] Using Smooth L1 (Huber) loss: targets are continuous coordinates with a quadratic region near zero.")
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
    
    # Replace the loss function with Gaussian smoothed loss if enabled (only for cross-entropy)
    if use_gaussian_smoothing and loss_type == 'cross_entropy':
        custom_loss = create_gaussian_smoothed_loss(
            vocab_size=256, 
            std=gaussian_std, 
            truncate=gaussian_truncate
        )
        # We need to monkey patch the loss computation since it's hardcoded in the network
        # This is a bit hacky but necessary to apply Gaussian smoothing
        original_loss_object = network._loss_object
        
        def custom_loss_wrapper(y_true, y_pred):
            return custom_loss(y_true, y_pred)
        
        network._loss_object = custom_loss_wrapper
        logging.info(f"Applied Gaussian smoothed loss with std={gaussian_std}, truncate={gaussian_truncate}")
        logging.info("[LOSS MODE] Using Gaussian-smoothed loss: targets are smoothed distributions, not class indices.")
    
    return agent

def check_trainable_parameters(agent):
    """Check and print which parameters are trainable vs frozen."""
    logging.info("="*60)
    logging.info("PARAMETER TRAINABILITY CHECK")
    logging.info("="*60)
    
    network = agent._actor_network
    image_tokenizer = network._image_tokenizer

    # EfficientNet
    logging.info(f"EfficientNet Encoder trainable: {image_tokenizer._tokenizer.trainable}")

    # TokenLearner
    if hasattr(image_tokenizer, '_token_learner') and image_tokenizer._token_learner:
        logging.info(f"TokenLearner trainable: {image_tokenizer._token_learner.trainable}")
        if image_tokenizer._token_learner.trainable:
            logging.info("  ✅ TokenLearner will be updated during training!")
        else:
            logging.info("  ❌ TokenLearner is frozen and will NOT be updated")
    else:
        logging.info("TokenLearner: not present")
        
    # Transformer
    transformer = network._transformer
    logging.info(f"Transformer trainable: {transformer.trainable}")

    # Action-related layers (different for cross-entropy vs L1)
    if hasattr(network, '_action_token_emb'):
        # Cross-entropy loss: action token embedding
        action_token_emb = network._action_token_emb
        logging.info(f"Action Token Embedding trainable: {action_token_emb.trainable}")
    elif hasattr(network, '_regression_output'):
        # L1 loss: regression output layer
        regression_output = network._regression_output
        logging.info(f"Regression Output Layer trainable: {regression_output.trainable}")
    else:
        logging.warning("No action-related layer found for parameter check")
    
    logging.info("="*60)

def apply_freezing_logic(agent, freeze_efficientnet=False, freeze_tokenlearner=False, freeze_transformer=False):
    """Apply freezing logic to different components of the model."""
    logging.info("Applying freezing logic...")
    
    network = agent._actor_network
    image_tokenizer = network._image_tokenizer
    
    # Freeze EfficientNet encoder
    if freeze_efficientnet:
        logging.info("Freezing EfficientNet encoder...")
        image_tokenizer._tokenizer.trainable = False
    else:
        logging.info("EfficientNet encoder will be trainable")
        image_tokenizer._tokenizer.trainable = True
    
    # Freeze TokenLearner
    if freeze_tokenlearner and hasattr(image_tokenizer, '_token_learner') and image_tokenizer._token_learner:
        logging.info("Freezing TokenLearner...")
        image_tokenizer._token_learner.trainable = False
    elif hasattr(image_tokenizer, '_token_learner') and image_tokenizer._token_learner:
        logging.info("TokenLearner will be trainable")
        image_tokenizer._token_learner.trainable = True
    
    # Freeze Transformer
    if freeze_transformer:
        logging.info("Freezing Transformer layers...")
        network._transformer.trainable = False
    else:
        logging.info("Transformer layers will be trainable")
        network._transformer.trainable = True
    
    # Action-related layers are always trainable (different for cross-entropy vs L1)
    if hasattr(network, '_action_token_emb'):
        # Cross-entropy loss: action token embedding
        logging.info("Action token embedding will be trainable")
        network._action_token_emb.trainable = True
    elif hasattr(network, '_regression_output'):
        # L1 loss: regression output layer
        logging.info("Regression output layer will be trainable")
        network._regression_output.trainable = True
    else:
        logging.warning("No action-related layer found for freezing logic")
    
    logging.info("Freezing logic applied!")

# =============================================================================
# CHECKPOINT MANAGEMENT UTILS
# =============================================================================

def load_pretrained_checkpoint(agent, checkpoint_path):
    """Load pretrained checkpoint using custom loading logic."""
    logging.info(f"Loading pretrained checkpoint from: {checkpoint_path}")
    
    try:
        # Use custom loading function for pretrained checkpoints
        custom_load_checkpoint(agent._actor_network, checkpoint_path)
        logging.info("✅ Pretrained checkpoint loaded successfully!")
        return True
    except Exception as e:
        logging.error(f"❌ Failed to load pretrained checkpoint: {e}")
        return False

def load_resume_checkpoint(agent, checkpoint_path):
    """Load resume checkpoint using custom loading logic."""
    logging.info(f"Loading resume checkpoint from: {checkpoint_path}")
    
    try:
        # Use custom loading function for resume checkpoints
        custom_load_checkpoint(agent._actor_network, checkpoint_path)
        logging.info("✅ Resume checkpoint loaded successfully!")
        return True
    except Exception as e:
        logging.error(f"❌ Failed to load resume checkpoint: {e}")
        return False

def _log_checkpoint_loading_details(checkpoint_path, model_weights):
    """Log detailed information about checkpoint loading."""
    logging.info(f"Checkpoint path: {checkpoint_path}")
    logging.info(f"Number of model weights: {len(model_weights)}")
    
    # Log some weight statistics
    total_params = 0
    for weight in model_weights:
        if hasattr(weight, 'shape'):
            params = np.prod(weight.shape)
            total_params += params
            logging.info(f"  {weight.name}: {weight.shape} ({params:,} parameters)")
    
    logging.info(f"Total parameters: {total_params:,}")

# =============================================================================
# TRAINING & LOSS UTILS
# =============================================================================

def create_gaussian_smoothed_loss(vocab_size, std=1.0, truncate=3.0):
    """Create a loss function that applies Gaussian smoothing to token targets."""
    def gaussian_smoothed_loss(y_true, y_pred):
        # y_true: [batch, timesteps, tokens_per_action] - integer tokens
        # y_pred: [batch, timesteps, tokens_per_action, vocab_size] - logits
        
        # Convert integer tokens to one-hot
        y_true_onehot = tf.one_hot(y_true, depth=vocab_size, dtype=tf.float32)
        
        # Apply Gaussian smoothing
        # Create coordinate grid for the vocabulary
        vocab_coords = tf.range(vocab_size, dtype=tf.float32)  # [0, 1, ..., vocab_size-1]
        vocab_coords = tf.reshape(vocab_coords, [1, 1, 1, vocab_size])  # [1, 1, 1, vocab_size]
        
        # Expand target tokens for broadcasting
        y_true_float = tf.cast(y_true, tf.float32)  # [batch, timesteps, tokens_per_action]
        y_true_expanded = tf.expand_dims(y_true_float, axis=-1)  # [batch, timesteps, tokens_per_action, 1]
        
        # Calculate Gaussian probabilities
        gaussian_probs = tf.exp(-0.5 * ((vocab_coords - y_true_expanded) / std) ** 2)
        gaussian_probs = gaussian_probs / tf.reduce_sum(gaussian_probs, axis=-1, keepdims=True)
        
        # Truncate Gaussian
        if truncate > 0:
            distance = tf.abs(vocab_coords - y_true_expanded)
            mask = distance <= (std * truncate)
            gaussian_probs = gaussian_probs * tf.cast(mask, tf.float32)
            # Renormalize
            gaussian_probs = gaussian_probs / (tf.reduce_sum(gaussian_probs, axis=-1, keepdims=True) + 1e-8)
        
        # Calculate cross-entropy loss
        loss = tf.keras.losses.categorical_crossentropy(gaussian_probs, y_pred, from_logits=True)
        return tf.reduce_mean(loss)
    
    return gaussian_smoothed_loss

def train_step(agent, batch_data):
    """Single training step."""
    # Prepare observations
    observations = {
        'image': batch_data['image'],
        'natural_language_embedding': batch_data['natural_language_embedding']
    }
    
    # Prepare actions
    actions = tensorspec_utils.TensorSpecStruct(
        bbox=batch_data['bbox']
    )
    
    # Get batch size from data
    batch_size = tf.shape(observations['image'])[0]
    
    # Expand observations to [B, T, ...]
    observations = {
        'image': tf.expand_dims(observations['image'], axis=1),  # [B, 1, H, W, C]
        'natural_language_embedding': tf.expand_dims(observations['natural_language_embedding'], axis=1)  # [B, 1, 512]
    }
    
    # Expand actions to [B, T, ...]
    actions = tensorspec_utils.TensorSpecStruct(
        bbox=tf.expand_dims(actions.bbox, axis=1)  # [B, 1, 4]
    )
    
    # Create time steps with time dimension
    time_steps = ts.TimeStep(
        step_type=tf.expand_dims(tf.constant([ts.StepType.FIRST] * batch_size), axis=1),  # [B, 1]
        reward=tf.zeros([batch_size, 1], dtype=tf.float32),  # [B, 1]
        discount=tf.ones([batch_size, 1], dtype=tf.float32),  # [B, 1]
        observation=observations)
    
    # Create policy steps
    policy_steps = policy_step.PolicyStep(action=actions)
    
    # Create experience
    experience = trajectory.from_transition(time_steps, policy_steps, time_steps)
    
    # Train the agent
    loss_info = agent.train(experience)

    return loss_info

def compute_bbox_metrics(pred_bbox, gt_bbox):
    """Compute MSE and MAE metrics for bbox predictions.
    
    Args:
        pred_bbox: Predicted bbox [batch, 4] (x1, y1, x2, y2)
        gt_bbox: Ground truth bbox [batch, 4] (x1, y1, x2, y2)
    
    Returns:
        Dictionary containing various metrics
    """
    try:
        # Convert to numpy for easier computation
        pred_bbox = pred_bbox.numpy() if hasattr(pred_bbox, 'numpy') else pred_bbox
        gt_bbox = gt_bbox.numpy() if hasattr(gt_bbox, 'numpy') else gt_bbox
        
        # Ensure both are numpy arrays
        pred_bbox = np.array(pred_bbox)
        gt_bbox = np.array(gt_bbox)
        
        # Check shapes
        if pred_bbox.shape != gt_bbox.shape:
            raise ValueError(f"Shape mismatch: pred_bbox {pred_bbox.shape} vs gt_bbox {gt_bbox.shape}")
        
        if len(pred_bbox.shape) != 2 or pred_bbox.shape[1] != 4:
            raise ValueError(f"Expected pred_bbox shape [batch, 4], got {pred_bbox.shape}")
        
        # Check for NaN or inf values
        if np.any(np.isnan(pred_bbox)) or np.any(np.isinf(pred_bbox)):
            logging.warning("NaN or inf values detected in predictions")
            pred_bbox = np.nan_to_num(pred_bbox, nan=0.0, posinf=1.0, neginf=0.0)
        
        if np.any(np.isnan(gt_bbox)) or np.any(np.isinf(gt_bbox)):
            logging.warning("NaN or inf values detected in ground truth")
            gt_bbox = np.nan_to_num(gt_bbox, nan=0.0, posinf=1.0, neginf=0.0)
        
        # Overall bbox MSE and MAE
        bbox_mse = np.mean((pred_bbox - gt_bbox) ** 2)
        bbox_mae = np.mean(np.abs(pred_bbox - gt_bbox))
        
        # Centroid metrics (center point of bbox)
        pred_centroid_x = (pred_bbox[:, 0] + pred_bbox[:, 2]) / 2
        pred_centroid_y = (pred_bbox[:, 1] + pred_bbox[:, 3]) / 2
        gt_centroid_x = (gt_bbox[:, 0] + gt_bbox[:, 2]) / 2
        gt_centroid_y = (gt_bbox[:, 1] + gt_bbox[:, 3]) / 2
        
        pred_centroids = np.stack([pred_centroid_x, pred_centroid_y], axis=1)
        gt_centroids = np.stack([gt_centroid_x, gt_centroid_y], axis=1)
        
        centroid_mse = np.mean((pred_centroids - gt_centroids) ** 2)
        centroid_mae = np.mean(np.abs(pred_centroids - gt_centroids))
        
        # Skewering point metrics (midpoint of the line segment)
        # For bbox, skewering point is the center of the diagonal from (x1,y1) to (x2,y2)
        pred_skewer_x = (pred_bbox[:, 0] + pred_bbox[:, 2]) / 2
        pred_skewer_y = (pred_bbox[:, 1] + pred_bbox[:, 3]) / 2
        gt_skewer_x = (gt_bbox[:, 0] + gt_bbox[:, 2]) / 2
        gt_skewer_y = (gt_bbox[:, 1] + gt_bbox[:, 3]) / 2
        
        pred_skewer = np.stack([pred_skewer_x, pred_skewer_y], axis=1)
        gt_skewer = np.stack([gt_skewer_x, gt_skewer_y], axis=1)
        
        skewer_mse = np.mean((pred_skewer - gt_skewer) ** 2)
        skewer_mae = np.mean(np.abs(pred_skewer - gt_skewer))
        
        # Orientation metrics (angle of the bbox diagonal)
        pred_dx = pred_bbox[:, 2] - pred_bbox[:, 0]
        pred_dy = pred_bbox[:, 3] - pred_bbox[:, 1]
        gt_dx = gt_bbox[:, 2] - gt_bbox[:, 0]
        gt_dy = gt_bbox[:, 3] - gt_bbox[:, 1]
        
        pred_angle = np.arctan2(pred_dy, pred_dx)
        gt_angle = np.arctan2(gt_dy, gt_dx)
        
        # Handle angle wrapping
        angle_diff = (pred_angle - gt_angle + np.pi) % (2 * np.pi) - np.pi
        angle_mse_rad = np.mean(angle_diff ** 2)
        angle_mae_rad = np.mean(np.abs(angle_diff))
        angle_mse_deg = np.mean((angle_diff * 180 / np.pi) ** 2)
        angle_mae_deg = np.mean(np.abs(angle_diff * 180 / np.pi))
        
        return {
            'bbox_mse': bbox_mse,
            'bbox_mae': bbox_mae,
            'centroid_mse': centroid_mse,
            'centroid_mae': centroid_mae,
            'skewer_mse': skewer_mse,
            'skewer_mae': skewer_mae,
            'angle_mse_rad': angle_mse_rad,
            'angle_mae_rad': angle_mae_rad,
            'angle_mse_deg': angle_mse_deg,
            'angle_mae_deg': angle_mae_deg
        }
    except Exception as e:
        logging.error(f"compute_bbox_metrics failed: {e}")
        logging.error(f"pred_bbox shape: {pred_bbox.shape if hasattr(pred_bbox, 'shape') else 'no shape'}")
        logging.error(f"gt_bbox shape: {gt_bbox.shape if hasattr(gt_bbox, 'shape') else 'no shape'}")
        # Return default metrics on error
        return {
            'bbox_mse': 1.0,
            'bbox_mae': 1.0,
            'centroid_mse': 1.0,
            'centroid_mae': 1.0,
            'skewer_mse': 1.0,
            'skewer_mae': 1.0,
            'angle_mse_rad': 1.0,
            'angle_mae_rad': 1.0,
            'angle_mse_deg': 1.0,
            'angle_mae_deg': 1.0
        }

def validate_step_with_metrics(agent, batch_data):
    """Run validation step and compute bbox metrics."""
    try:
        # Prepare observations
        observations = {
            'image': batch_data['image'],
            'natural_language_embedding': batch_data['natural_language_embedding']
        }
        actions = tensorspec_utils.TensorSpecStruct(
            bbox=batch_data['bbox']
        )
        batch_size = tf.shape(observations['image'])[0]
        
        # Expand dims for time axis
        observations = {
            'image': tf.expand_dims(observations['image'], axis=1),
            'natural_language_embedding': tf.expand_dims(observations['natural_language_embedding'], axis=1)
        }
        actions = tensorspec_utils.TensorSpecStruct(
            bbox=tf.expand_dims(actions.bbox, axis=1)
        )
        
        # Create TimeStep and PolicyStep
        time_steps = ts.TimeStep(
            step_type=tf.expand_dims(tf.constant([ts.StepType.FIRST] * batch_size, dtype=tf.int32), axis=1),
            reward=tf.zeros([batch_size, 1], dtype=tf.float32),
            discount=tf.ones([batch_size, 1], dtype=tf.float32),
            observation=observations
        )
        policy_steps = policy_step.PolicyStep(action=actions)
        
        # Create experience
        experience = trajectory.from_transition(time_steps, policy_steps, time_steps)
        
        # Compute loss (no gradient update)
        loss_info = agent._loss(experience, weights=None, training=False)
        
        # Get predicted bbox from the model
        network = agent._actor_network
        
        # Handle different loss types
        if hasattr(network, '_action_tokenizer') and network._action_tokenizer is not None:
            # Cross-entropy loss: get logits, argmax, then detokenize
            predicted_tokens = network.get_action_logits()  # [batch, tokens_per_action, vocab_size]
            predicted_tokens = tf.argmax(predicted_tokens, axis=-1, output_type=tf.int32)  # [batch, tokens_per_action]
            
            # Detokenize to get predicted bbox
            predicted_actions = network._action_tokenizer.detokenize(predicted_tokens)
            pred_bbox = predicted_actions.bbox  # [batch, 4]
        else:
            # L1 loss: get predictions directly from aux_info
            aux_info = network.get_aux_info()
            pred_bbox = aux_info['action_predictions']  # [batch, 4]
        
        # Get ground truth bbox
        gt_bbox = batch_data['bbox']  # [batch, 4]
        
        # Debug shapes and values
        logging.debug(f"Validation - pred_bbox shape: {pred_bbox.shape}, dtype: {pred_bbox.dtype}")
        logging.debug(f"Validation - gt_bbox shape: {gt_bbox.shape}, dtype: {gt_bbox.dtype}")
        logging.debug(f"Validation - pred_bbox range: [{tf.reduce_min(pred_bbox):.4f}, {tf.reduce_max(pred_bbox):.4f}]")
        logging.debug(f"Validation - gt_bbox range: [{tf.reduce_min(gt_bbox):.4f}, {tf.reduce_max(gt_bbox):.4f}]")
        
        # Compute metrics
        metrics = compute_bbox_metrics(pred_bbox, gt_bbox)
        
        return loss_info, metrics
        
    except Exception as e:
        logging.error(f"Validation step failed: {e}")
        logging.error(f"Batch data keys: {list(batch_data.keys())}")
        logging.error(f"Batch data shapes: {[(k, v.shape if hasattr(v, 'shape') else 'no shape') for k, v in batch_data.items()]}")
        raise

def validate_step(agent, batch_data):
    """Run validation step without updating weights, returning agent loss."""
    # Prepare observations
    observations = {
        'image': batch_data['image'],
        'natural_language_embedding': batch_data['natural_language_embedding']
    }
    actions = tensorspec_utils.TensorSpecStruct(
        bbox=batch_data['bbox']
    )
    batch_size = tf.shape(observations['image'])[0]
    # Expand dims for time axis
    observations = {
        'image': tf.expand_dims(observations['image'], axis=1),
        'natural_language_embedding': tf.expand_dims(observations['natural_language_embedding'], axis=1)
    }
    actions = tensorspec_utils.TensorSpecStruct(
        bbox=tf.expand_dims(actions.bbox, axis=1)
    )
    # Create TimeStep and PolicyStep
    time_steps = ts.TimeStep(
        step_type=tf.expand_dims(tf.constant([ts.StepType.FIRST] * batch_size, dtype=tf.int32), axis=1),
        reward=tf.zeros([batch_size, 1], dtype=tf.float32),
        discount=tf.ones([batch_size, 1], dtype=tf.float32),
        observation=observations
    )
    policy_steps = policy_step.PolicyStep(action=actions)
    # Create experience
    experience = trajectory.from_transition(time_steps, policy_steps, time_steps)
    # Compute loss (no gradient update)
    loss_info = agent._loss(experience, weights=None, training=False)
    return loss_info

# =============================================================================
# MONITORING & VISUALIZATION UTILS
# =============================================================================

def plot_training_curves(train_losses, val_losses, checkpoint_dir):
    """Plot and save training curves."""
    plt.figure(figsize=(12, 4))
    
    # Plot losses
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss', color='blue')
    plt.plot(val_losses, label='Validation Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot loss difference
    plt.subplot(1, 2, 2)
    loss_diff = [abs(t - v) for t, v in zip(train_losses, val_losses)]
    plt.plot(loss_diff, label='|Train - Val| Loss', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Loss Difference')
    plt.title('Training vs Validation Loss Difference')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(checkpoint_dir, 'training_curves.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info(f"Training curves saved to: {plot_path}")

def log_memory_usage():
    """Log current memory usage."""
    try:
        if tf.config.list_physical_devices('GPU'):
            gpu_devices = tf.config.list_physical_devices('GPU')
            for i, device in enumerate(gpu_devices):
                memory_info = tf.config.experimental.get_memory_info(device.name)
                logging.info(f"GPU {i} memory: {memory_info['current'] / 1024**3:.2f}GB / {memory_info['peak'] / 1024**3:.2f}GB")
    except:
        logging.info("Could not get detailed memory info")

def clear_memory():
    """Clear memory and garbage collect."""
    gc.collect()
    tf.keras.backend.clear_session()

def get_weight_stats(weights):
    """Get statistics about model weights."""
    stats = {}
    for weight in weights:
        if hasattr(weight, 'numpy'):
            weight_np = weight.numpy()
            stats[weight.name] = {
                'shape': weight_np.shape,
                'mean': float(np.mean(weight_np)),
                'std': float(np.std(weight_np)),
                'min': float(np.min(weight_np)),
                'max': float(np.max(weight_np)),
                'norm': float(np.linalg.norm(weight_np))
            }
    return stats

def compare_weight_stats(before_stats, after_stats, logger=logging.info):
    """Compare weight statistics before and after some operation."""
    logger("="*60)
    logger("WEIGHT STATISTICS COMPARISON")
    logger("="*60)
    
    for name in before_stats.keys():
        if name in after_stats:
            before = before_stats[name]
            after = after_stats[name]
            
            logger(f"Layer: {name}")
            logger(f"  Mean: {before['mean']:.6f} -> {after['mean']:.6f} (diff: {after['mean'] - before['mean']:.6f})")
            logger(f"  Std:  {before['std']:.6f} -> {after['std']:.6f} (diff: {after['std'] - before['std']:.6f})")
            logger(f"  Norm: {before['norm']:.6f} -> {after['norm']:.6f} (diff: {after['norm'] - before['norm']:.6f})")
            logger("")
    
    logger("="*60)


