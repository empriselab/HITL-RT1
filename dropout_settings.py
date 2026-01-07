#!/usr/bin/env python3
"""Simple dropout settings for RT-1 model.

Modify these values to control dropout rates throughout the model.
"""

# # EfficientNet Backbone Dropouts
# BACKBONE_SPATIAL_DROPOUT = 0.0      # Spatial dropout after conv blocks
# BACKBONE_TOP_DROPOUT = 0.2          # Top layer dropout (default)
# BACKBONE_DROP_CONNECT = 0.2         # Drop connect rate (default)

# # TokenLearner Dropouts
# TOKEN_LEARNER_DROPOUT = 0.0         # TokenLearner MLP dropout

# # Action Embedding Dropout
# ACTION_EMBEDDING_DROPOUT = 0.0      # Action token embedding dropout

# # Transformer Dropouts
# TRANSFORMER_EMBEDDING_DROPOUT = 0.0 # Token/position embedding dropout
# TRANSFORMER_ATTENTION_DROPOUT = 0.1 # Multi-head attention dropout (default)
# TRANSFORMER_FF_DROPOUT = 0.1        # Feed-forward layer dropout (default)

# # Inference Dropout Control (for uncertainty quantification)
# ENABLE_INFERENCE_DROPOUT = False    # Enable dropout during inference

# EfficientNet Backbone Dropouts
BACKBONE_SPATIAL_DROPOUT = 0.1      # Spatial dropout after conv blocks
BACKBONE_TOP_DROPOUT = 0.1          # Top layer dropout (default)
BACKBONE_DROP_CONNECT = 0.1         # Drop connect rate (default)

# TokenLearner Dropouts
TOKEN_LEARNER_DROPOUT = 0.1        # TokenLearner MLP dropout

# Action Embedding Dropout
ACTION_EMBEDDING_DROPOUT = 0.1      # Action token embedding dropout

# Transformer Dropouts
TRANSFORMER_EMBEDDING_DROPOUT = 0.1 # Token/position embedding dropout
TRANSFORMER_ATTENTION_DROPOUT = 0.1 # Multi-head attention dropout (default)
TRANSFORMER_FF_DROPOUT = 0.1        # Feed-forward layer dropout (default)

# Inference Dropout Control (for uncertainty quantification)
ENABLE_INFERENCE_DROPOUT = True    # Enable dropout during inference
