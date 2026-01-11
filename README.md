# RT-1 Fine-Tuning for Food Manipulation Tasks

This repository contains a fine-tuned RT-1 (Robotic Transformer) model that predicts low-level execution parameters for food acquisition tasks. Given an image of a food item and a natural language instruction, the model outputs execution parameters [x_1, y_1, x_2, y_2] that specify how to perform the manipulation action. The model supports three manipulation actions: **skewering**, **scooping**, and **twirling**.

## Quick Start

There are two main scripts to run:

1. **Training**: `slurm_scripts/run_training.slurm` - Starts the fine-tuning process
2. **Inference Server**: `slurm_scripts/api_server.slurm` - Runs the API server for predictions

### Running Training

```bash
sbatch slurm_scripts/run_training.slurm
```

This script calls `train.py`.

### Running the API Server

```bash
sbatch slurm_scripts/api_server.slurm
```

This starts a Flask API server that loads a trained checkpoint and serves predictions via HTTP endpoints.

## Environment Setup

The project uses a conda environment. Package versions are specified in `package_verisons.txt`. To recreate the environment:

Key dependencies:
- TensorFlow 2.11.0
- TF-Agents 0.15.0
- Gin-Config 0.5.0
- Flask 3.1.2
- NumPy 1.21.6
- OpenCV 4.7.0.72

## Repository Structure and Logic Flow

### Core Architecture Files

The model architecture follows this chain:

1. **`agent_wrapper.py`** - Top-level wrapper that creates the Agent model
   - Wraps `ActorNetwork` to match checkpoint variable naming hierarchy
   - Ensures variable names follow pattern: `agent/_actor_network/...`

2. **`transformer_network.py`** - Main transformer-based actor network
   - Implements `TransformerNetwork` class that combines image and action tokens
   - Uses `RT1ImageTokenizer` and `RT1ActionTokenizer` for tokenization
   - Processes tokens through transformer layers with causal attention masks

3. **`sequence_agent.py`** - TF-Agents wrapper
   - `SequenceAgent`: Wraps the transformer network for training/inference
   - `SequencePolicy`: Policy that outputs actions via actor network
   - Handles loss computation and gradient updates

4. **`transformer.py`** - Transformer implementation
   - `Transformer`: Multi-layer transformer decoder
   - `_TransformerLayer`: Single transformer block with multi-head attention and feed-forward layers

### Tokenization Components

5. **`tokenizers/image_tokenizer.py`** - Image tokenization
   - `RT1ImageTokenizer`: Converts images to tokens using EfficientNet + TokenLearner
   - Uses FiLM (Feature-wise Linear Modulation) for natural language conditioning

6. **`tokenizers/action_tokenizer.py`** - Action tokenization
   - `RT1ActionTokenizer`: Converts execution parameters (4D coordinates) to discrete tokens
   - Supports both tokenization (action → tokens) and detokenization (tokens → action)

7. **`tokenizers/token_learner.py`** - Token learning module
   - Learns to extract compact token representations from image features

### Image Encoding

8. **`film_efficientnet/`** - EfficientNet encoder with FiLM conditioning
   - `pretrained_efficientnet_encoder.py`: Loads pretrained EfficientNet weights
   - `film_conditioning_layer.py`: FiLM layer for language conditioning
   - `film_efficientnet_encoder.py`: Combines EfficientNet with FiLM

### Training Pipeline

9. **`train.py`** - Main training script
   - Loads dataset from `.npz` file
   - Creates train/validation splits
   - Initializes agent with pretrained checkpoint
   - Runs training loop with validation
   - Saves best checkpoints based on validation metrics (centroid MAE for execution parameters)

10. **`utils.py`** - Utility functions for training
    - `create_agent()`: Creates and configures the agent
    - `create_npz_dataset()`: Loads and preprocesses dataset
    - `train_step()`: Single training step
    - `validate_step_with_metrics()`: Validation with execution parameter metrics
    - `load_pretrained_checkpoint()` / `load_resume_checkpoint()`: Checkpoint loading
    - `apply_freezing_logic()`: Freeze/unfreeze model components
    - Data augmentation functions (rotation, lighting)

11. **`test_checkpoint_loading.py`** - Checkpoint loading utilities
    - `custom_load_checkpoint()`: Custom checkpoint loading with variable name mapping
    - Handles mapping between checkpoint variable names and model variable names
    - Supports loading pretrained RT-1 checkpoints

### Inference Pipeline

12. **`api_batch_server.py`** - Flask API server for inference
    - `/predict`: Single prediction endpoint
    - `/predict_batch`: Batch prediction with Monte Carlo Dropout for uncertainty
    - `/available_combinations`: List available action-food combinations
    - `/load_model`: Load/reload model checkpoint
    - Uses pre-computed embeddings from `embeddings/` directory

13. **`api_server.py`** - Alternative API server (simpler version)
    - Similar to `api_batch_server.py` but without batch processing

14. **`api_client.py`** - Client for sending API requests
    - Example client code for interacting with the API server

15. **`batch_api_inference.py`** - Batch inference script
    - Processes multiple images from a directory
    - Saves results to JSON file
    - Can be used for evaluation on test sets

### Configuration and Settings

16. **`dropout_settings.py`** - Dropout configuration
    - Controls dropout rates for different components
    - Enables/disables dropout during inference

17. **`policy_specs.pbtxt`** - Policy specifications (protobuf format)

## Data Flow

### Training Flow

```
train.py
  ↓
utils.create_npz_dataset() → Loads .npz file, creates train/val splits
  ↓
utils.create_agent() → Creates Agent model
  ↓
agent_wrapper.Agent → Wraps ActorNetwork
  ↓
transformer_network.TransformerNetwork → Main network
  ↓
  ├─→ image_tokenizer.RT1ImageTokenizer → EfficientNet + TokenLearner
  ├─→ action_tokenizer.RT1ActionTokenizer → Execution parameters → tokens
  └─→ transformer.Transformer → Processes tokens
  ↓
Training loop: train_step() → loss → gradients → optimizer
  ↓
Checkpoint saving (best model based on validation metrics)
```

### Inference Flow

```
api_batch_server.py
  ↓
Load checkpoint → test_checkpoint_loading.custom_load_checkpoint()
  ↓
Create agent → agent_wrapper.Agent
  ↓
Receive image + action/food_type
  ↓
Load embedding from embeddings/ directory
  ↓
agent.forward() → transformer_network.TransformerNetwork
  ↓
  ├─→ image_tokenizer → image tokens
  ├─→ action_tokenizer → detokenize → execution parameters (4D coordinates)
  └─→ transformer → process sequence
  ↓
Return execution parameters (start_x, start_y, end_x, end_y)
```

## Data

- **Dataset File**: `/share/bhattacharjee/frank_data/dataset_sk_sc.npz`
- **Data Directory**: `/share/bhattacharjee/frank_data`
- **Train-Val Split**: `/share/bhattacharjee/frank_data/train_val_split_sk_sc.npz`

The dataset contains:
- Images: Food item images (236x236 pixels)
- Execution parameters: 4D coordinates (x1, y1, x2, y2) normalized to [0, 1], representing start and end points for manipulation actions
- Natural language embeddings: 512-dimensional embeddings for action-food combinations

## Checkpoint
`https://drive.google.com/drive/folders/1TglfMMy4vbHvKdPmLe8byVhO9Crbkrye?usp=share_link`
## Supported Actions and Food Items

**Actions**: skewering, scooping, twirling

**Food Items**: banana, blueberry, broccoli, brownie, cantaloupe, cherry_tomato, chicken, fettucine, grape, green_bean, honeydew, lettuce, mac_and_cheese, mashed_potato, meatball, oatmeal, pineapple, rice, sausage, spaghetti, strawberry, watermelon

Pre-computed embeddings for all action-food combinations are stored in the `embeddings/` directory.

## Model Architecture

The model is based on RT-1 architecture:

1. **Image Encoder**: EfficientNet-B3 with FiLM conditioning
2. **TokenLearner**: Learns compact token representations (8 tokens per image)
3. **Image Tokenizer**: Combines EfficientNet features with TokenLearner
4. **Action Tokenizer**: Discretizes execution parameters (4D coordinates) into tokens (vocab_size=256)
5. **Transformer**: 8-layer transformer with causal attention masks
6. **Output**: Action token logits → detokenized to execution parameters (4D coordinates: start_x, start_y, end_x, end_y)

## Training Configuration

Training can be configured via command-line flags in `train.py`:

- `--config_file`: Gin config file (default: `configs/transformer_mixin.gin`)
- `--checkpoint_dir`: Output directory for checkpoints
- `--pretrained_checkpoint`: Path to pretrained RT-1 checkpoint
- `--resume_checkpoint`: Path to checkpoint to resume from
- `--batch_size`: Training batch size (default: 16)
- `--num_epochs`: Number of epochs (default: 30)
- `--loss_type`: Loss function - `cross_entropy`, `l1`, or `smooth_l1` (default: `smooth_l1`)
- `--freeze_efficientnet`, `--freeze_tokenlearner`, `--freeze_transformer`: Freeze components

## API Usage

Once the API server is running, you can send requests:

```python
import requests
import base64

# Load image
with open('image.jpg', 'rb') as f:
    image_data = base64.b64encode(f.read()).decode('utf-8')

# Send prediction request
response = requests.post('http://localhost:8080/predict', json={
    'image': image_data,
    'action': 'skewering',
    'food_type': 'banana'
})

result = response.json()
prediction = result['prediction']
# Contains: start_x, start_y, end_x, end_y, skewer_x, skewer_y, direction_degrees
execution_params = [prediction['start_x'], prediction['start_y'], 
                    prediction['end_x'], prediction['end_y']]
```
