#!/usr/bin/env python3
"""Training script for RT-1 fine-tuning."""

import os
import gin
import tensorflow as tf
import numpy as np
from absl import app, flags, logging
from PIL import Image
import json
import glob
import copy
import gc
import matplotlib.pyplot as plt
import math

# Import utility functions
import utils

learning_rate = 0.0001

# Gaussian smoothing parameters (can be tuned)
GAUSSIAN_SMOOTHING_STD = 2.0  # Standard deviation for Gaussian smoothing
GAUSSIAN_SMOOTHING_TRUNCATE = 4.0  # Truncate Gaussian at ±4σ
USE_GAUSSIAN_SMOOTHING = False  # Whether to use Gaussian smoothing instead of one-hot


apply_rotation_augmentation = False
apply_lighting_augmentation = True   # Enable fast lighting augmentation

# Suppress TensorFlow logging to stop parameter value printing
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF logging
tf.get_logger().setLevel('ERROR')  # Only show errors

FLAGS = flags.FLAGS
#trained_checkpoints/rt1main/ckpt-424760
flags.DEFINE_string('config_file', 'configs/transformer_mixin.gin',
                    'Path to gin config file.')
flags.DEFINE_string('checkpoint_dir', './output_checkpoints27-smooth_l1',
                    'Directory to save checkpoints.')
flags.DEFINE_string('pretrained_checkpoint', './output_checkpoints26-smooth_l1/best_checkpoint',
                    'Path to pretrained RT-1 checkpoint to load.')
flags.DEFINE_string('resume_checkpoint', "",
                    'Path to checkpoint to resume training from.')
flags.DEFINE_string('data_dir', '/share/bhattacharjee/frank_data',
                    'Directory containing prepared training data.')
flags.DEFINE_string('dataset_file', '/share/bhattacharjee/frank_data/dataset_sk_sc.npz',
                    'Path to the dataset .npz file.')
flags.DEFINE_integer('batch_size', 16, 'Training batch size.')
flags.DEFINE_integer('num_epochs', 30, 'Number of training epochs.')
flags.DEFINE_integer('image_size', 236, 'Input image size (assumes square).')
flags.DEFINE_boolean('freeze_efficientnet', False, 'Whether to freeze EfficientNet encoder.')
flags.DEFINE_boolean('freeze_tokenlearner', False, 'Whether to freeze TokenLearner.')
flags.DEFINE_boolean('freeze_transformer', False, 'Whether to freeze Transformer layers.')
flags.DEFINE_enum('loss_type', 'smooth_l1', ['cross_entropy', 'l1', 'smooth_l1'], 'Loss function type: cross_entropy, l1, or smooth_l1')

print("TensorFlow version:", tf.__version__)

def main(argv):
  """Main training function."""
  del argv
  
  # Check GPU availability
  print("="*60)
  print("DEVICE CONFIGURATION")
  print("="*60)
  print(f"TensorFlow version: {tf.__version__}")
  print(f"CUDA available: {tf.test.is_built_with_cuda()}")
  print(f"GPU devices: {tf.config.list_physical_devices('GPU')}")
  print(f"CPU devices: {tf.config.list_physical_devices('CPU')}")
  
  # Test device placement
  with tf.device('/GPU:0'):
    test_tensor = tf.constant([1.0, 2.0, 3.0])
    print(f"Test tensor device: {test_tensor.device}")
  
  print("="*60)
  
  # Load gin configuration
  gin.parse_config_file(FLAGS.config_file)
  
  # Create checkpoint directory
  os.makedirs(FLAGS.checkpoint_dir, exist_ok=True)
  
  # Create agent using utility function
  agent = utils.create_agent(
      learning_rate=learning_rate,
      use_gaussian_smoothing=USE_GAUSSIAN_SMOOTHING,
      gaussian_std=GAUSSIAN_SMOOTHING_STD,
      gaussian_truncate=GAUSSIAN_SMOOTHING_TRUNCATE,
      loss_type=FLAGS.loss_type
  )

  # Print all variables in the model after creation
  print("\nVariables in the model after creation:", flush=True)
  for var in agent.variables:
      print(f"{var.name}: {var.shape}", flush=True)

  # Load pretrained checkpoint if available
  if FLAGS.pretrained_checkpoint:
    utils.load_pretrained_checkpoint(agent, FLAGS.pretrained_checkpoint)
    # Print all variables in the pretrained checkpoint
    try:
        checkpoint_vars = tf.train.list_variables(FLAGS.pretrained_checkpoint)
        print("\nVariables in the pretrained checkpoint:", flush=True)
        for var_name, var_shape in checkpoint_vars:
            print(f"{var_name}: {var_shape}", flush=True)
    except Exception as e:
        print(f"Could not list checkpoint variables: {e}", flush=True)
    # Print all variables in the model after loading
    print("\nVariables in the model after loading pretrained checkpoint:", flush=True)
    for var in agent.variables:
        print(f"{var.name}: {var.shape}", flush=True)
  
  # Load resume checkpoint if available (for continuing training)
  resumed = False
  if FLAGS.resume_checkpoint:
    resumed = utils.load_resume_checkpoint(agent, FLAGS.resume_checkpoint)
    # Print all variables in the checkpoint
    try:
        checkpoint_vars = tf.train.list_variables(FLAGS.resume_checkpoint)
        print("\nVariables in the resume checkpoint:", flush=True)
        for var_name, var_shape in checkpoint_vars:
            print(f"{var_name}: {var_shape}", flush=True)
    except Exception as e:
        print(f"Could not list checkpoint variables: {e}")
    # Print all variables in the model after loading
    print("\nVariables in the model after loading resume checkpoint:", flush=True)
    for var in agent.variables:
        print(f"{var.name}: {var.shape}", flush=True)
  
  # Apply freezing logic based on FLAGS
  utils.apply_freezing_logic(
      agent, 
      freeze_efficientnet=FLAGS.freeze_efficientnet,
      freeze_tokenlearner=FLAGS.freeze_tokenlearner,
      freeze_transformer=FLAGS.freeze_transformer
  )
  
  # Check which parameters are trainable vs frozen
  utils.check_trainable_parameters(agent)
  
  # Create train/val datasets
  split_file = os.path.join(FLAGS.data_dir, 'train_val_split_sk_sc.npz')
  train_ds, val_ds, num_train, num_val = utils.create_npz_dataset(
      FLAGS.dataset_file, 
      FLAGS.batch_size, 
      split_file=split_file,
      apply_rotation_augmentation=apply_rotation_augmentation,
      apply_lighting_augmentation=apply_lighting_augmentation,
      loss_type=FLAGS.loss_type
  )
  
  # Calculate steps per epoch automatically
  steps_per_epoch = int(np.ceil(num_train / FLAGS.batch_size))
  steps_per_val = int(np.ceil(num_val / FLAGS.batch_size))
  
  logging.info(f"Training samples: {num_train}, Validation samples: {num_val}")
  logging.info(f"Batch size: {FLAGS.batch_size}")
  logging.info(f"Steps per epoch: {steps_per_epoch}")
  logging.info(f"Steps per validation: {steps_per_val}")
  
  # Training loop
  logging.info("Starting training for bbox prediction...")
  
  # Initialize loss tracking
  train_losses = []
  val_losses = []
  step_losses = []
  best_centroid_mae = float('inf')  # Changed from best_val_loss to best_centroid_mae
  best_val_epoch = -1
  
  for epoch in range(FLAGS.num_epochs):
    epoch_losses = []
    # Log memory usage at start of epoch
    logging.info(f"Epoch {epoch} starting - Memory usage:")
    utils.log_memory_usage()
    # Create fresh training iterator for each epoch
    train_iter = iter(train_ds)
    for step in range(steps_per_epoch):
        batch_data = next(train_iter)
        loss_info = utils.train_step(agent, batch_data)
        step_loss = loss_info.loss.numpy()
        epoch_losses.append(step_loss)
        step_losses.append(step_loss)
        if step % 10 == 0:
            logging.info(f"Epoch {epoch}, Step {step}, Loss: {step_loss:.4f}")

        # --- Distribution comparison plot for epoch 0, step 0 (only for cross-entropy loss) ---
        if epoch == 0 and step == 0 and FLAGS.loss_type == 'cross_entropy':
            # Get logits and targets from the agent/network
            network = agent._actor_network
            logits = network.get_action_logits().numpy()  # [B, tokens_per_action, vocab_size]
            labels = network.get_aux_info()['action_labels'].numpy()  # [B, T, tokens_per_action]
            if labels.ndim == 3:
                labels = labels[:, -1, :]  # [B, tokens_per_action]

            batch_size = logits.shape[0]
            tokens_per_action = logits.shape[1]
            vocab_size = logits.shape[2]
            # Plotting code (unchanged)
            fig, axes = plt.subplots(batch_size, tokens_per_action, figsize=(tokens_per_action*4, batch_size*2))
            if batch_size == 1:
                axes = np.expand_dims(axes, 0)
            for i in range(batch_size):
                for j in range(tokens_per_action):
                    ax = axes[i, j] if batch_size > 1 else axes[0, j]
                    pred_dist = np.exp(logits[i, j] - np.max(logits[i, j]))
                    pred_dist /= np.sum(pred_dist)
                    gt_class = labels[i, j]
                    gt_dist = np.zeros_like(pred_dist)
                    gt_dist[gt_class] = 1.0
                    ax.plot(pred_dist, label='Predicted', color='blue')
                    ax.plot(gt_dist, label='Ground Truth', color='red', linestyle='dashed')
                    ax.set_title(f'Sample {i}, Token {j}')
                    ax.set_xlabel('Class')
                    ax.set_ylabel('Probability')
                    ax.legend()
            plt.tight_layout()
            plot_path = os.path.join(FLAGS.checkpoint_dir, 'distribution_comparison_epoch0_step0.png')
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            logging.info(f"Saved distribution comparison plot to {plot_path}")

            # --- Debug loss calculation ---
            # Calculate per-token cross-entropy loss manually
            ce_losses = []
            gt_probs = []
            for i in range(batch_size):
                for j in range(tokens_per_action):
                    pred_dist = np.exp(logits[i, j] - np.max(logits[i, j]))
                    pred_dist /= np.sum(pred_dist)
                    gt_class = labels[i, j]
                    gt_prob = pred_dist[gt_class]
                    gt_probs.append(gt_prob)
                    ce_loss = -np.log(max(gt_prob, 1e-12))
                    ce_losses.append(ce_loss)
                    logging.info(f"Sample {i}, Token {j}: GT class={gt_class}, Pred prob={gt_prob:.6f}, CE loss={ce_loss:.6f}")
            ce_losses = np.array(ce_losses)
            gt_probs = np.array(gt_probs)
            mean_ce_loss = np.mean(ce_losses)
            logging.info(f"[DEBUG] Mean per-token cross-entropy loss (manual): {mean_ce_loss:.6f}")
            logging.info(f"[DEBUG] Min/Max GT prob: {gt_probs.min():.6f} / {gt_probs.max():.6f}")
            # Print correct denominator used in loss calculation
            num_action_tokens = float(batch_size * tokens_per_action)
            logging.info(f"[DEBUG] Denominator used in loss calculation: num_action_tokens={num_action_tokens}")
        elif epoch == 0 and step == 0 and FLAGS.loss_type == 'l1':
            # For L1 loss, log some coordinate statistics instead
            network = agent._actor_network
            aux_info = network.get_aux_info()
            predictions = aux_info['action_predictions'].numpy()  # [B, 4]
            labels = aux_info['action_labels'].numpy()  # [B, 4]
            
            logging.info(f"[L1 DEBUG] Predictions shape: {predictions.shape}, Labels shape: {labels.shape}")
            logging.info(f"[L1 DEBUG] Predictions range: [{predictions.min():.4f}, {predictions.max():.4f}]")
            logging.info(f"[L1 DEBUG] Labels range: [{labels.min():.4f}, {labels.max():.4f}]")
            
            # Calculate per-coordinate L1 errors
            l1_errors = np.abs(predictions - labels)
            for i in range(4):
                coord_name = ['x1', 'y1', 'x2', 'y2'][i]
                mean_error = np.mean(l1_errors[:, i])
                logging.info(f"[L1 DEBUG] {coord_name} mean L1 error: {mean_error:.6f}")
            
            logging.info(f"[L1 DEBUG] Overall mean L1 error: {np.mean(l1_errors):.6f}")
    # Calculate average loss for epoch
    avg_loss = np.mean(epoch_losses)
    logging.info(f"Epoch {epoch} completed. Average loss: {avg_loss:.4f}")
    train_losses.append(avg_loss)
    # Validation
    logging.info(f"Starting validation for epoch {epoch}...")
    epoch_val_losses = []
    epoch_metrics = []
    try:
        val_iter = iter(val_ds)
        for val_step in range(steps_per_val):
        # for val_step in range(10):  # Only do 10 validation steps per epoch for testing
            val_batch = next(val_iter)
            val_loss_info, val_metrics = utils.validate_step_with_metrics(agent, val_batch)
            val_loss = val_loss_info.loss.numpy()
            epoch_val_losses.append(val_loss)
            epoch_metrics.append(val_metrics)
        
        avg_val_loss = np.mean(epoch_val_losses)
        
        # Compute average metrics across all validation steps
        avg_metrics = {}
        for key in epoch_metrics[0].keys():
            avg_metrics[key] = np.mean([m[key] for m in epoch_metrics])
        
        logging.info(f"Epoch {epoch} validation loss: {avg_val_loss:.4f}")
        logging.info(f"Epoch {epoch} bbox metrics:")
        logging.info(f"  Bbox MSE: {avg_metrics['bbox_mse']:.6f}, MAE: {avg_metrics['bbox_mae']:.6f}")
        logging.info(f"  Centroid MSE: {avg_metrics['centroid_mse']:.6f}, MAE: {avg_metrics['centroid_mae']:.6f}")
        logging.info(f"  Skewer MSE: {avg_metrics['skewer_mse']:.6f}, MAE: {avg_metrics['skewer_mae']:.6f}")
        logging.info(f"  Angle MSE: {avg_metrics['angle_mse_deg']:.6f} deg², MAE: {avg_metrics['angle_mae_deg']:.6f} deg")
        
        val_losses.append(avg_val_loss)
    except Exception as e:
        logging.error(f"Validation failed: {e}")
        avg_val_loss = 0.0
        logging.info(f"Epoch {epoch} validation loss: {avg_val_loss:.4f} (failed)")
        val_losses.append(avg_val_loss)

    # Append to loss log file
    loss_log_path = os.path.join(FLAGS.checkpoint_dir, 'loss_log.txt')
    if epoch == 0 and not os.path.exists(loss_log_path):
        with open(loss_log_path, 'w') as f:
            f.write('epoch,train_loss,val_loss,bbox_mse,bbox_mae,centroid_mse,centroid_mae,skewer_mse,skewer_mae,angle_mse_deg,angle_mae_deg\n')
    
    # Prepare metrics for logging
    if 'avg_metrics' in locals():
        log_line = f'{epoch},{avg_loss},{avg_val_loss},{avg_metrics["bbox_mse"]:.6f},{avg_metrics["bbox_mae"]:.6f},{avg_metrics["centroid_mse"]:.6f},{avg_metrics["centroid_mae"]:.6f},{avg_metrics["skewer_mse"]:.6f},{avg_metrics["skewer_mae"]:.6f},{avg_metrics["angle_mse_deg"]:.6f},{avg_metrics["angle_mae_deg"]:.6f}\n'
    else:
        log_line = f'{epoch},{avg_loss},{avg_val_loss},0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0\n'
    
    with open(loss_log_path, 'a') as f:
        f.write(log_line)

    # Save best checkpoint based on centroid_mae (lower is better)
    if 'avg_metrics' in locals() and avg_metrics['centroid_mae'] < best_centroid_mae:
        best_centroid_mae = avg_metrics['centroid_mae']
        best_val_epoch = epoch
        
        best_checkpoint_path = os.path.join(FLAGS.checkpoint_dir, 'best_checkpoint')
        checkpoint = tf.train.Checkpoint(agent=agent, optimizer=agent._actor_optimizer)
        checkpoint.save(best_checkpoint_path)
        logging.info(f"New best centroid MAE! Checkpoint saved to {best_checkpoint_path} (epoch {epoch}, centroid_mae: {avg_metrics['centroid_mae']:.6f})")
    
    # Clear memory after each epoch
    utils.clear_memory()
    logging.info(f"Memory cleared after epoch {epoch}")
  
  # Save final checkpoint
  final_checkpoint_path = os.path.join(FLAGS.checkpoint_dir, 'final_checkpoint')
  checkpoint = tf.train.Checkpoint(agent=agent, optimizer=agent._actor_optimizer)
  checkpoint.save(final_checkpoint_path)
  logging.info(f"Final checkpoint saved to {final_checkpoint_path}")
  
  # Final memory cleanup
  utils.clear_memory()
  logging.info("Training completed!")
  
  # Plot training curves and save data
  plot_path = utils.plot_training_curves(train_losses, val_losses, FLAGS.checkpoint_dir)
  
  # Print training summary
  print("\n" + "="*60)
  print("TRAINING SUMMARY")
  print("="*60)
  print(f"Total epochs: {len(train_losses)}")
  print(f"Final training loss: {train_losses[-1]:.4f}")
  print(f"Final validation loss: {val_losses[-1]:.4f}")
  print(f"Best training loss: {min(train_losses):.4f} (epoch {train_losses.index(min(train_losses)) + 1})")
  print(f"Best validation loss: {min(val_losses):.4f} (epoch {val_losses.index(min(val_losses)) + 1})")
  print(f"Best centroid MAE: {best_centroid_mae:.6f} (epoch {best_val_epoch + 1})")
  print(f"Training curves saved to: {plot_path}")

if __name__ == '__main__':
  app.run(main) 