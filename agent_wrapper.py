#!/usr/bin/env python3
"""Wrapper agent class to match checkpoint variable naming hierarchy."""

import tensorflow as tf
from typing import Any, Dict, Optional, Tuple

import transformer_network
from tokenizers import image_tokenizer, action_tokenizer
from film_efficientnet import pretrained_efficientnet_encoder
from tokenizers.image_tokenizer import RT1ImageTokenizer
from tokenizers.action_tokenizer import RT1ActionTokenizer
from transformer import Transformer
import dropout_settings


class Agent(tf.keras.Model):
    """Wrapper agent class that matches checkpoint variable naming hierarchy.
    
    This wrapper ensures that variable names follow the pattern:
    agent/_actor_network/_image_tokenizer/_tokenizer/net/...
    agent/_actor_network/_action_token_emb/...
    agent/_actor_network/transformer/...
    etc.
    """
    
    def __init__(self, input_tensor_spec, output_tensor_spec, **kwargs):
        """Initialize the wrapper with specs and build the model with correct naming."""
        super().__init__()
        
        # Store specs
        self._input_tensor_spec = input_tensor_spec
        self._output_tensor_spec = output_tensor_spec
        
        # Build the model with the correct hierarchy
        self._actor_network = ActorNetwork(
            input_tensor_spec=input_tensor_spec,
            output_tensor_spec=output_tensor_spec,
            name='_actor_network',
            **kwargs
        )
    
    def call(self, inputs, training=None):
        """Forward pass through the agent."""
        return self._actor_network(inputs, training=training)
    
    def get_logits(self, inputs, training=None):
        """Get the raw logits output."""
        return self._actor_network(inputs, training=training)
    
    
    @property
    def trainable_variables(self):
        """Get trainable variables from the wrapped network."""
        return self._actor_network.trainable_variables
    
    @property
    def variables(self):
        """Get all variables from the wrapped network."""
        return self._actor_network.variables


class ActorNetwork(tf.keras.Model):
    """Actor network that matches the checkpoint hierarchy."""
    
    def __init__(self, input_tensor_spec, output_tensor_spec, **kwargs):
        super().__init__()
        # Extract hyperparameters from kwargs or set defaults
        token_embedding_size = kwargs.get('token_embedding_size', 512)
        vocab_size = kwargs.get('vocab_size', 256)
        use_token_learner = kwargs.get('use_token_learner', True)
        num_tokens = kwargs.get('num_tokens', 8)
        kernel_regularizer = kwargs.get('kernel_regularizer', None)
        action_order = kwargs.get('action_order', None)
        num_layers = kwargs.get('num_layers', 8)
        layer_size = kwargs.get('layer_size', 128)
        num_heads = kwargs.get('num_heads', 8)
        feed_forward_size = kwargs.get('feed_forward_size', 512)
        dropout_rate = kwargs.get('dropout_rate', dropout_settings.TRANSFORMER_ATTENTION_DROPOUT)
        time_sequence_length = kwargs.get('time_sequence_length', 1)
        # Image tokenizer
        self._image_tokenizer = RT1ImageTokenizer(
            embedding_output_dim=token_embedding_size,
            use_token_learner=use_token_learner,
            num_tokens=num_tokens,
            kernel_regularizer=kernel_regularizer,
            name='_image_tokenizer'
        )
        # Action tokenizer
        self._action_tokenizer = RT1ActionTokenizer(
            output_tensor_spec,
            vocab_size=vocab_size,
            action_order=action_order
        )
        # Action token embedding
        self._action_token_emb = tf.keras.layers.Dense(
            token_embedding_size,
            name='_action_token_emb',
            kernel_regularizer=kernel_regularizer
        )
        # Action embedding dropout
        self._action_embedding_dropout = tf.keras.layers.Dropout(
            rate=kwargs.get('action_embedding_dropout_rate', dropout_settings.ACTION_EMBEDDING_DROPOUT),
            name='_action_embedding_dropout'
        )
        # Transformer
        self._transformer = Transformer(
            num_layers=num_layers,
            layer_size=layer_size,
            num_heads=num_heads,
            feed_forward_size=feed_forward_size,
            dropout_rate=dropout_rate,
            vocab_size=vocab_size,
            kernel_regularizer=kernel_regularizer,
            embedding_dropout_rate=kwargs.get('embedding_dropout_rate', dropout_settings.TRANSFORMER_EMBEDDING_DROPOUT)
        )
        # Output projection
        # self._output_projection = tf.keras.layers.Dense(
        #     vocab_size,
        #     name='_output_tokens'
        # )
        self._token_embedding_size = token_embedding_size
        self._vocab_size = vocab_size
        self._tokens_per_action = self._action_tokenizer.tokens_per_action
        self._tokens_per_context_image = self._image_tokenizer.tokens_per_context_image
        self._time_sequence_length = time_sequence_length
        # --- Attention mask generation (copied from transformer_network.py) ---
        self._generate_masks()

    def _get_action_index_for_token(self, k):
        if (k < 0 or k >= self._all_num_tokens):
            return -1
        n = k
        if n % self._single_time_step_num_tokens < self._tokens_per_context_image:
            return -1
        return int(n / self._single_time_step_num_tokens)

    def _generate_masks(self):
        self._single_time_step_num_tokens = (
            self._tokens_per_action + self._tokens_per_context_image)
        self._all_num_tokens = (
            self._time_sequence_length * self._single_time_step_num_tokens)
        # create mask for action prediction loss (not used here, but for completeness)
        self._action_tokens_mask = []
        for n in range(0, self._all_num_tokens, self._single_time_step_num_tokens):
            for x in range(0, self._tokens_per_action, 1):
                self._action_tokens_mask.append(x + n + self._tokens_per_context_image)
        self._action_tokens_mask = tf.constant(
            self._action_tokens_mask, dtype=tf.int32)
        # The look ahead mask ensures causality.
        self._default_attention_mask = tf.linalg.band_part(
            tf.ones((self._all_num_tokens, self._all_num_tokens)), -1, 0)
        import numpy as np
        action_mask = np.ndarray(
            shape=(self._all_num_tokens, self._all_num_tokens), dtype=int)
        for i in range(self._all_num_tokens):
            for j in range(self._all_num_tokens):
                action_i = self._get_action_index_for_token(i)
                action_j = self._get_action_index_for_token(j)
                mask = 0
                if action_i != -1 and action_j != -1:
                    if action_j < action_i:
                        mask = 1
                    if (action_j == action_i and j <= i):
                        mask = 1
                action_mask[i, j] = mask
        self._default_attention_mask -= action_mask

    def call(self, inputs, training=None):
        observations = inputs
        image = tf.expand_dims(observations['image'], axis=1)  # [B, 1, H, W, 3]
        nle = tf.expand_dims(observations['natural_language_embedding'], axis=1)  # [B, 1, 512]
        batch_size = tf.shape(image)[0]
        image_tokens = self._image_tokenizer(image, context=nle, training=training)  # [B, 1, num_tokens, emb]
        if 'bbox' in observations:
            action = {'bbox': observations['bbox']}
            action_tokens = self._action_tokenizer.tokenize(action)  # [B, tokens_per_action]
            action_tokens = tf.expand_dims(action_tokens, axis=1)  # [B, 1, tokens_per_action]
        else:
            action_tokens = tf.zeros([batch_size, 1, self._tokens_per_action], dtype=tf.int32)
        action_tokens_onehot = tf.one_hot(action_tokens, self._vocab_size)  # [B, 1, tokens_per_action, vocab]
        action_tokens_embedded = self._action_token_emb(action_tokens_onehot)  # [B, 1, tokens_per_action, emb]
        action_tokens_embedded = self._action_embedding_dropout(action_tokens_embedded, training=training)  # Apply dropout
        action_tokens_embedded = tf.expand_dims(action_tokens_embedded, axis=-2)  # [B, 1, tokens_per_action, 1, emb]
        image_tokens = tf.expand_dims(image_tokens, axis=3)  # [B, 1, num_tokens, 1, emb]
        input_token_sequence = tf.concat([image_tokens, action_tokens_embedded], axis=2)  # [B, 1, num_tokens+tokens_per_action, 1, emb]
        seq_len = self._tokens_per_context_image + self._tokens_per_action
        input_token_sequence = tf.reshape(input_token_sequence, [batch_size, seq_len, self._token_embedding_size])
        # Use the full mask logic
        attention_mask = self._default_attention_mask[:seq_len, :seq_len]
        attention_mask = tf.expand_dims(tf.expand_dims(attention_mask, 0), 0)  # [1, 1, seq_len, seq_len]
        
        # Get logits (transformer output is already logits)
        transformer_output, _ = self._transformer(
            input_token_sequence, training=training, attention_mask=attention_mask
        )
        
        action_token_start = self._tokens_per_context_image
        action_token_end = action_token_start + self._tokens_per_action
        
        # Extract action token logits
        action_token_logits = transformer_output[:, action_token_start:action_token_end, :]  # [batch, tokens_per_action, vocab_size]
        
        return action_token_logits 