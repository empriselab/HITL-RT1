#!/usr/bin/env python3
"""Script to test actual checkpoint loading and count variable matches/mismatches."""

import os
import gin
import tensorflow as tf
import numpy as np
from absl import app, flags, logging

import sequence_agent
import transformer_network
import agent_wrapper
from tokenizers import action_tokenizer
from tensor2robot.utils import tensorspec_utils
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import time_step as ts

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')


def create_bbox_action_spec():
    """Create action spec for bbox prediction."""
    action_spec = tensorspec_utils.TensorSpecStruct()
    action_spec.bbox = tensor_spec.BoundedTensorSpec(
        [4], dtype=tf.float32, name='bbox', minimum=0., maximum=1.)
    return action_spec

def create_observation_spec():
    """Create observation spec."""
    observation_spec = tensorspec_utils.TensorSpecStruct()
    observation_spec.image = tensor_spec.BoundedTensorSpec(
        [236, 236, 3], dtype=tf.float32, name='image', minimum=0., maximum=1.)
    observation_spec.natural_language_embedding = tensor_spec.TensorSpec(
        shape=[512], dtype=tf.float32, name='natural_language_embedding')
    return observation_spec

def create_agent(time_step_spec, action_spec):
    """Create agent using the wrapper class."""
    # Create the agent directly with specs
    agent = agent_wrapper.Agent(
        input_tensor_spec=time_step_spec.observation,
        output_tensor_spec=action_spec,
        time_sequence_length=6
    )
    
    # Build the network by calling it with dummy input
    dummy_observations = {}
    for key, spec in time_step_spec.observation.items():
        if hasattr(spec, 'shape') and hasattr(spec, 'dtype'):
            dummy_observations[key] = tf.zeros([1] + list(spec.shape), dtype=spec.dtype)
        else:
            dummy_observations[key] = tf.zeros([1, 512], dtype=tf.float32)
    
    # Call the network to build it
    _ = agent(dummy_observations, training=False)
    
    return agent

def efficientnet_batchnorm_mapping(c_name, model_name):
    if 'gamma' in model_name:
        c_name = c_name+'/gamma/.ATTRIBUTES/VARIABLE_VALUE'
    elif 'beta' in model_name:
        c_name = c_name+'/beta/.ATTRIBUTES/VARIABLE_VALUE'
    elif 'moving_mean' in model_name:
        c_name = c_name+'/moving_mean/.ATTRIBUTES/VARIABLE_VALUE'
    elif 'moving_variance' in model_name:
        c_name = c_name+'/moving_variance/.ATTRIBUTES/VARIABLE_VALUE'
    return c_name

def kernel_bias_mapping(c_name, model_name):
    if "kernel" in model_name:
        c_name = c_name+'/kernel/.ATTRIBUTES/VARIABLE_VALUE'
    elif "bias" in model_name:
        c_name = c_name+'/bias/.ATTRIBUTES/VARIABLE_VALUE'
    return c_name

def efficientnet_block_mapping(c_name, model_name, start_index):
    if "block1a" in model_name or "block1b" in model_name:
        if "dwconv" in model_name:
            c_name = f"agent/_actor_network/_image_tokenizer/_tokenizer/net/layer_with_weights-{start_index}/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUE"
        elif "project_bn" in model_name:
            c_name = f"agent/_actor_network/_image_tokenizer/_tokenizer/net/layer_with_weights-{start_index+5}"
            c_name = efficientnet_batchnorm_mapping(c_name, model_name)
        elif "_bn" in model_name:
            c_name = f"agent/_actor_network/_image_tokenizer/_tokenizer/net/layer_with_weights-{start_index+1}"
            c_name = efficientnet_batchnorm_mapping(c_name, model_name)
        elif "se_reduce" in model_name:
            c_name = f"agent/_actor_network/_image_tokenizer/_tokenizer/net/layer_with_weights-{start_index+2}"
            c_name = kernel_bias_mapping(c_name, model_name)
        elif "se_expand" in model_name:
            c_name = f"agent/_actor_network/_image_tokenizer/_tokenizer/net/layer_with_weights-{start_index+3}"
            c_name = kernel_bias_mapping(c_name, model_name)
        elif "project_conv" in model_name:
            c_name = f"agent/_actor_network/_image_tokenizer/_tokenizer/net/layer_with_weights-{start_index+4}"
            c_name = kernel_bias_mapping(c_name, model_name)
    else:
        if "expand_conv" in model_name:
            c_name = f"agent/_actor_network/_image_tokenizer/_tokenizer/net/layer_with_weights-{start_index}/kernel/.ATTRIBUTES/VARIABLE_VALUE"
        elif "project_bn" in model_name:
            c_name = f"agent/_actor_network/_image_tokenizer/_tokenizer/net/layer_with_weights-{start_index+7}"
            c_name = efficientnet_batchnorm_mapping(c_name, model_name)
        elif "expand_bn" in model_name:
            c_name = f"agent/_actor_network/_image_tokenizer/_tokenizer/net/layer_with_weights-{start_index+1}"
            c_name = efficientnet_batchnorm_mapping(c_name, model_name)
        elif "dwconv" in model_name:
            c_name = f"agent/_actor_network/_image_tokenizer/_tokenizer/net/layer_with_weights-{start_index+2}/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUE"
        elif "_bn" in model_name:
            c_name = f"agent/_actor_network/_image_tokenizer/_tokenizer/net/layer_with_weights-{start_index+3}"
            c_name = efficientnet_batchnorm_mapping(c_name, model_name)
        elif "se_reduce" in model_name:
            c_name = f"agent/_actor_network/_image_tokenizer/_tokenizer/net/layer_with_weights-{start_index+4}"
            c_name = kernel_bias_mapping(c_name, model_name)
        elif "se_expand" in model_name:
            c_name = f"agent/_actor_network/_image_tokenizer/_tokenizer/net/layer_with_weights-{start_index+5}"
            c_name = kernel_bias_mapping(c_name, model_name)
        elif "project_conv" in model_name:
            c_name = f"agent/_actor_network/_image_tokenizer/_tokenizer/net/layer_with_weights-{start_index+6}"
            c_name = kernel_bias_mapping(c_name, model_name)
    return c_name

def film_conditioning_mapping(c_name, model_name):
    if "film_conditioning_16" in c_name:
        c_name = "agent/_actor_network/_image_tokenizer/_tokenizer/film_layer/_projection_add/"
    elif "film_conditioning_26" in c_name:
        c_name = "agent/_actor_network/_image_tokenizer/_tokenizer/film_layer/_projection_mult/"
    return c_name

def map_model_name_to_checkpoint_name(model_name, pretrained=False):
    """Map model variable names to checkpoint variable names based on detailed mapping logic.
    If pretrained is True, then we use the pretrained checkpoint mapping logic.
    If pretrained is False, then we use the resumed training checkpoint mapping logic.
    """
    # Remove :0 suffix if present
    if model_name.endswith(':0'):
        model_name = model_name[:-2]
    
    # Initialize checkpoint name
    c_name = model_name
    
    # Basic top-level mappings
    if 'agent/actor_network/' in c_name:
        c_name = c_name.replace('agent/actor_network/', 'agent/_actor_network/')
    
    if 'efficient_net_encoder' in c_name:
        c_name = c_name.replace('efficient_net_encoder', '_image_tokenizer')
    if '/transformer/' in c_name:
        c_name = c_name.replace('/transformer/', '/_transformer/')

    if 'bbox_regression' in c_name:
        if "kernel" in c_name:
            c_name = "agent/_actor_network/_regression_output/kernel/.ATTRIBUTES/VARIABLE_VALUE"
        elif "bias" in c_name:
            c_name = "agent/_actor_network/_regression_output/bias/.ATTRIBUTES/VARIABLE_VALUE"
    
    # Handle transformer layers with detailed logic
    import re
    transformer_match = re.search(r'private__transformer_layer_(\d+)', c_name)
    transformer_no_number_match = re.search(r'private__transformer_layer', c_name)
    if transformer_match or transformer_no_number_match:
        if transformer_match:
            layer_num = transformer_match.group(1)
            # Start building the checkpoint name for transformer layers
            c_name = f'agent/_actor_network/_transformer/_layers/{layer_num}/'
        else:
            c_name = f'agent/_actor_network/_transformer/_layers/0/'
        
        # Handle multi-head attention components
        if 'multi_head_attention_' in model_name or "multi_head_attention" in model_name:
            if 'query' in model_name:
                c_name += 'mha1/_query_dense/'
            elif 'key' in model_name:
                c_name += 'mha1/_key_dense/'
            elif 'value' in model_name:
                c_name += 'mha1/_value_dense/'
            elif 'attention_output' in model_name:
                c_name += 'mha1/_output_dense/'
            
            # Add parameter suffix
            if 'bias' in model_name:
                c_name += 'bias/.ATTRIBUTES/VARIABLE_VALUE'
            elif 'kernel' in model_name:
                c_name += 'kernel/.ATTRIBUTES/VARIABLE_VALUE'
        
        # Handle layer normalization components
        elif 'layer_normalization_' in model_name or "layer_normalization" in model_name:
            if 'attention' in model_name or 'mha' in model_name:
                c_name += 'layernorm1/'
            else:
                c_name += 'layernorm2/'
            
            if 'beta' in model_name:
                c_name += 'beta/.ATTRIBUTES/VARIABLE_VALUE'
            elif 'gamma' in model_name:
                c_name += 'gamma/.ATTRIBUTES/VARIABLE_VALUE'
        
        # Handle feed-forward dense layers
        elif 'dense_' in model_name or 'dense' in model_name:
            c_name += 'ff/'
            if 'bias' in model_name:
                c_name += 'bias/.ATTRIBUTES/VARIABLE_VALUE'
            elif 'kernel' in model_name:
                c_name += 'kernel/.ATTRIBUTES/VARIABLE_VALUE'
    
    # Handle token learner module
    elif 'token_learner_module' in c_name:
        c_name = 'agent/_actor_network/_image_tokenizer/_token_learner/'
        
        if 'mlp_block/hidden_dense' in model_name:
            c_name += 'mlp/_hidden_layer/'
            if 'bias' in model_name:
                c_name += 'bias/.ATTRIBUTES/VARIABLE_VALUE'
            elif 'kernel' in model_name:
                c_name += 'kernel/.ATTRIBUTES/VARIABLE_VALUE'
        elif 'mlp_block/final_dense' in model_name:
            c_name += 'mlp/_output_layer/'
            if 'bias' in model_name:
                c_name += 'bias/.ATTRIBUTES/VARIABLE_VALUE'
            elif 'kernel' in model_name:
                c_name += 'kernel/.ATTRIBUTES/VARIABLE_VALUE'
        elif 'layer_normalization' in model_name:
            c_name += 'layernorm/'
            if 'beta' in model_name:
                c_name += 'beta/.ATTRIBUTES/VARIABLE_VALUE'
            elif 'gamma' in model_name:
                c_name += 'gamma/.ATTRIBUTES/VARIABLE_VALUE'
    # elif '_output_tokens' in c_name:
    #     c_name = 'agent/_actor_network/_transformer/_output_tokens/'
    #     if 'kernel' in model_name:
    #         c_name += 'kernel/.ATTRIBUTES/VARIABLE_VALUE'
    #     elif 'bias' in model_name:
    #         c_name += 'bias/.ATTRIBUTES/VARIABLE_VALUE'
    elif "film_conditioning" in model_name:
        if "film_conditioning/dense_11" in model_name:
            start_index = 9
            c_name = f"agent/_actor_network/_image_tokenizer/_tokenizer/net/layer_with_weights-{start_index}/_projection_add"
            c_name = kernel_bias_mapping(c_name, model_name)
        elif "film_conditioning/dense_12" in model_name:
            start_index = 9
            c_name = f"agent/_actor_network/_image_tokenizer/_tokenizer/net/layer_with_weights-{start_index}/_projection_mult"
            c_name = kernel_bias_mapping(c_name, model_name)
        elif "film_conditioning_1/dense_13" in model_name:
            start_index = 16
            c_name = f"agent/_actor_network/_image_tokenizer/_tokenizer/net/layer_with_weights-{start_index}/_projection_add"
            c_name = kernel_bias_mapping(c_name, model_name)
        elif "film_conditioning_1/dense_14" in model_name:
            start_index = 16
            c_name = f"agent/_actor_network/_image_tokenizer/_tokenizer/net/layer_with_weights-{start_index}/_projection_mult"
            c_name = kernel_bias_mapping(c_name, model_name)
        elif "film_conditioning_2/dense_15" in model_name:
            start_index = 25
            c_name = f"agent/_actor_network/_image_tokenizer/_tokenizer/net/layer_with_weights-{start_index}/_projection_add"
            c_name = kernel_bias_mapping(c_name, model_name)
        elif "film_conditioning_2/dense_16" in model_name:
            start_index = 25
            c_name = f"agent/_actor_network/_image_tokenizer/_tokenizer/net/layer_with_weights-{start_index}/_projection_mult"
            c_name = kernel_bias_mapping(c_name, model_name)
        elif "film_conditioning_3/dense_17" in model_name:
            start_index = 34
            c_name = f"agent/_actor_network/_image_tokenizer/_tokenizer/net/layer_with_weights-{start_index}/_projection_add"
            c_name = kernel_bias_mapping(c_name, model_name)
        elif "film_conditioning_3/dense_18" in model_name:
            start_index = 34
            c_name = f"agent/_actor_network/_image_tokenizer/_tokenizer/net/layer_with_weights-{start_index}/_projection_mult"
            c_name = kernel_bias_mapping(c_name, model_name)
        elif "film_conditioning_4/dense_19" in model_name:
            start_index = 43
            c_name = f"agent/_actor_network/_image_tokenizer/_tokenizer/net/layer_with_weights-{start_index}/_projection_add"
            c_name = kernel_bias_mapping(c_name, model_name)
        elif "film_conditioning_4/dense_20" in model_name:
            start_index = 43
            c_name = f"agent/_actor_network/_image_tokenizer/_tokenizer/net/layer_with_weights-{start_index}/_projection_mult"
            c_name = kernel_bias_mapping(c_name, model_name)
        elif "film_conditioning_5/dense_21" in model_name:
            start_index = 52
            c_name = f"agent/_actor_network/_image_tokenizer/_tokenizer/net/layer_with_weights-{start_index}/_projection_add"
            c_name = kernel_bias_mapping(c_name, model_name)
        elif "film_conditioning_5/dense_22" in model_name:
            start_index = 52
            c_name = f"agent/_actor_network/_image_tokenizer/_tokenizer/net/layer_with_weights-{start_index}/_projection_mult"
            c_name = kernel_bias_mapping(c_name, model_name)
        elif "film_conditioning_6/dense_23" in model_name:
            start_index = 61
            c_name = f"agent/_actor_network/_image_tokenizer/_tokenizer/net/layer_with_weights-{start_index}/_projection_add"
            c_name = kernel_bias_mapping(c_name, model_name)
        elif "film_conditioning_6/dense_24" in model_name:
            start_index = 61
            c_name = f"agent/_actor_network/_image_tokenizer/_tokenizer/net/layer_with_weights-{start_index}/_projection_mult"
            c_name = kernel_bias_mapping(c_name, model_name)
        elif "film_conditioning_7/dense_25" in model_name:
            start_index = 70
            c_name = f"agent/_actor_network/_image_tokenizer/_tokenizer/net/layer_with_weights-{start_index}/_projection_add"
            c_name = kernel_bias_mapping(c_name, model_name)
        elif "film_conditioning_7/dense_26" in model_name:
            start_index = 70
            c_name = f"agent/_actor_network/_image_tokenizer/_tokenizer/net/layer_with_weights-{start_index}/_projection_mult"
            c_name = kernel_bias_mapping(c_name, model_name)
        elif "film_conditioning_8/dense_27" in model_name:
            start_index = 79
            c_name = f"agent/_actor_network/_image_tokenizer/_tokenizer/net/layer_with_weights-{start_index}/_projection_add"
            c_name = kernel_bias_mapping(c_name, model_name)
        elif "film_conditioning_8/dense_28" in model_name:
            start_index = 79
            c_name = f"agent/_actor_network/_image_tokenizer/_tokenizer/net/layer_with_weights-{start_index}/_projection_mult"
            c_name = kernel_bias_mapping(c_name, model_name)
        elif "film_conditioning_9/dense_29" in model_name:
            start_index = 88
            c_name = f"agent/_actor_network/_image_tokenizer/_tokenizer/net/layer_with_weights-{start_index}/_projection_add"
            c_name = kernel_bias_mapping(c_name, model_name)
        elif "film_conditioning_9/dense_30" in model_name:
            start_index = 88
            c_name = f"agent/_actor_network/_image_tokenizer/_tokenizer/net/layer_with_weights-{start_index}/_projection_mult"
            c_name = kernel_bias_mapping(c_name, model_name)
        elif "film_conditioning_10/dense_31" in model_name:
            start_index = 97
            c_name = f"agent/_actor_network/_image_tokenizer/_tokenizer/net/layer_with_weights-{start_index}/_projection_add"
            c_name = kernel_bias_mapping(c_name, model_name)
        elif "film_conditioning_10/dense_32" in model_name:
            start_index = 97
            c_name = f"agent/_actor_network/_image_tokenizer/_tokenizer/net/layer_with_weights-{start_index}/_projection_mult"
            c_name = kernel_bias_mapping(c_name, model_name)
        elif "film_conditioning_11/dense_33" in model_name:
            start_index = 106
            c_name = f"agent/_actor_network/_image_tokenizer/_tokenizer/net/layer_with_weights-{start_index}/_projection_add"
            c_name = kernel_bias_mapping(c_name, model_name)
        elif "film_conditioning_11/dense_34" in model_name:
            start_index = 106
            c_name = f"agent/_actor_network/_image_tokenizer/_tokenizer/net/layer_with_weights-{start_index}/_projection_mult"
            c_name = kernel_bias_mapping(c_name, model_name)
        elif "film_conditioning_12/dense_35" in model_name:
            start_index = 115
            c_name = f"agent/_actor_network/_image_tokenizer/_tokenizer/net/layer_with_weights-{start_index}/_projection_add"
            c_name = kernel_bias_mapping(c_name, model_name)
        elif "film_conditioning_12/dense_36" in model_name:
            start_index = 115
            c_name = f"agent/_actor_network/_image_tokenizer/_tokenizer/net/layer_with_weights-{start_index}/_projection_mult"
            c_name = kernel_bias_mapping(c_name, model_name)
        elif "film_conditioning_13/dense_37" in model_name:
            start_index = 124
            c_name = f"agent/_actor_network/_image_tokenizer/_tokenizer/net/layer_with_weights-{start_index}/_projection_add"
            c_name = kernel_bias_mapping(c_name, model_name)
        elif "film_conditioning_13/dense_38" in model_name:
            start_index = 124
            c_name = f"agent/_actor_network/_image_tokenizer/_tokenizer/net/layer_with_weights-{start_index}/_projection_mult"
            c_name = kernel_bias_mapping(c_name, model_name)
        elif "film_conditioning_14/dense_39" in model_name:
            start_index = 133
            c_name = f"agent/_actor_network/_image_tokenizer/_tokenizer/net/layer_with_weights-{start_index}/_projection_add"
            c_name = kernel_bias_mapping(c_name, model_name)
        elif "film_conditioning_14/dense_40" in model_name:
            start_index = 133
            c_name = f"agent/_actor_network/_image_tokenizer/_tokenizer/net/layer_with_weights-{start_index}/_projection_mult"
            c_name = kernel_bias_mapping(c_name, model_name)
        elif "film_conditioning_15/dense_41" in model_name:
            start_index = 142
            c_name = f"agent/_actor_network/_image_tokenizer/_tokenizer/net/layer_with_weights-{start_index}/_projection_add"
            c_name = kernel_bias_mapping(c_name, model_name)
        elif "film_conditioning_15/dense_42" in model_name:
            start_index = 142
            c_name = f"agent/_actor_network/_image_tokenizer/_tokenizer/net/layer_with_weights-{start_index}/_projection_mult"
            c_name = kernel_bias_mapping(c_name, model_name)
        elif "film_conditioning_16/dense_43" in model_name:
            start_index = 151
            c_name = f"agent/_actor_network/_image_tokenizer/_tokenizer/net/layer_with_weights-{start_index}/_projection_add"
            c_name = kernel_bias_mapping(c_name, model_name)
        elif "film_conditioning_16/dense_44" in model_name:
            start_index = 151
            c_name = f"agent/_actor_network/_image_tokenizer/_tokenizer/net/layer_with_weights-{start_index}/_projection_mult"
            c_name = kernel_bias_mapping(c_name, model_name)
        elif "film_conditioning_17/dense_45" in model_name:
            start_index = 160
            c_name = f"agent/_actor_network/_image_tokenizer/_tokenizer/net/layer_with_weights-{start_index}/_projection_add"
            c_name = kernel_bias_mapping(c_name, model_name)
        elif "film_conditioning_17/dense_46" in model_name:
            start_index = 160
            c_name = f"agent/_actor_network/_image_tokenizer/_tokenizer/net/layer_with_weights-{start_index}/_projection_mult"
            c_name = kernel_bias_mapping(c_name, model_name)
        elif "film_conditioning_18/dense_47" in model_name:
            start_index = 169
            c_name = f"agent/_actor_network/_image_tokenizer/_tokenizer/net/layer_with_weights-{start_index}/_projection_add"
            c_name = kernel_bias_mapping(c_name, model_name)
        elif "film_conditioning_18/dense_48" in model_name:
            start_index = 169
            c_name = f"agent/_actor_network/_image_tokenizer/_tokenizer/net/layer_with_weights-{start_index}/_projection_mult"
            c_name = kernel_bias_mapping(c_name, model_name)
        elif "film_conditioning_19/dense_49" in model_name:
            start_index = 178
            c_name = f"agent/_actor_network/_image_tokenizer/_tokenizer/net/layer_with_weights-{start_index}/_projection_add"
            c_name = kernel_bias_mapping(c_name, model_name)
        elif "film_conditioning_19/dense_50" in model_name:
            start_index = 178
            c_name = f"agent/_actor_network/_image_tokenizer/_tokenizer/net/layer_with_weights-{start_index}/_projection_mult"
            c_name = kernel_bias_mapping(c_name, model_name)
        elif "film_conditioning_20/dense_51" in model_name:
            start_index = 187
            c_name = f"agent/_actor_network/_image_tokenizer/_tokenizer/net/layer_with_weights-{start_index}/_projection_add"
            c_name = kernel_bias_mapping(c_name, model_name)
        elif "film_conditioning_20/dense_52" in model_name:
            start_index = 187
            c_name = f"agent/_actor_network/_image_tokenizer/_tokenizer/net/layer_with_weights-{start_index}/_projection_mult"
            c_name = kernel_bias_mapping(c_name, model_name)
        elif "film_conditioning_21/dense_53" in model_name:
            start_index = 196
            c_name = f"agent/_actor_network/_image_tokenizer/_tokenizer/net/layer_with_weights-{start_index}/_projection_add"
            c_name = kernel_bias_mapping(c_name, model_name)
        elif "film_conditioning_21/dense_54" in model_name:
            start_index = 196
            c_name = f"agent/_actor_network/_image_tokenizer/_tokenizer/net/layer_with_weights-{start_index}/_projection_mult"
            c_name = kernel_bias_mapping(c_name, model_name)
        elif "film_conditioning_22/dense_55" in model_name:
            start_index = 205
            c_name = f"agent/_actor_network/_image_tokenizer/_tokenizer/net/layer_with_weights-{start_index}/_projection_add"
            c_name = kernel_bias_mapping(c_name, model_name)
        elif "film_conditioning_22/dense_56" in model_name:
            start_index = 205
            c_name = f"agent/_actor_network/_image_tokenizer/_tokenizer/net/layer_with_weights-{start_index}/_projection_mult"
            c_name = kernel_bias_mapping(c_name, model_name)
        elif "film_conditioning_23/dense_57" in model_name:
            start_index = 214
            c_name = f"agent/_actor_network/_image_tokenizer/_tokenizer/net/layer_with_weights-{start_index}/_projection_add"
            c_name = kernel_bias_mapping(c_name, model_name)
        elif "film_conditioning_23/dense_58" in model_name:
            start_index = 214
            c_name = f"agent/_actor_network/_image_tokenizer/_tokenizer/net/layer_with_weights-{start_index}/_projection_mult"
            c_name = kernel_bias_mapping(c_name, model_name)
        elif "film_conditioning_24/dense_59" in model_name:
            start_index = 223
            c_name = f"agent/_actor_network/_image_tokenizer/_tokenizer/net/layer_with_weights-{start_index}/_projection_add"
            c_name = kernel_bias_mapping(c_name, model_name)
        elif "film_conditioning_24/dense_60" in model_name:
            start_index = 223
            c_name = f"agent/_actor_network/_image_tokenizer/_tokenizer/net/layer_with_weights-{start_index}/_projection_mult"
            c_name = kernel_bias_mapping(c_name, model_name)
        elif "film_conditioning_25/dense_61" in model_name:
            start_index = 232
            c_name = f"agent/_actor_network/_image_tokenizer/_tokenizer/net/layer_with_weights-{start_index}/_projection_add"
            c_name = kernel_bias_mapping(c_name, model_name)
        elif "film_conditioning_25/dense_62" in model_name:
            start_index = 232
            c_name = f"agent/_actor_network/_image_tokenizer/_tokenizer/net/layer_with_weights-{start_index}/_projection_add"
            c_name = kernel_bias_mapping(c_name, model_name)
        elif "film_conditioning_26/dense_63" in model_name:
            c_name = f"agent/_actor_network/_image_tokenizer/_tokenizer/film_layer/_projection_add"
            c_name = kernel_bias_mapping(c_name, model_name)
        elif "film_conditioning_26/dense_64" in model_name:
            c_name = f"agent/_actor_network/_image_tokenizer/_tokenizer/film_layer/_projection_mult"
            c_name = kernel_bias_mapping(c_name, model_name)

    # Handle other components
    elif "dense_10" in model_name:
        c_name = 'agent/_actor_network/_transformer/_output_tokens/'
        if 'kernel' in model_name:
            c_name += 'kernel/.ATTRIBUTES/VARIABLE_VALUE'
        elif 'bias' in model_name:
            c_name += 'bias/.ATTRIBUTES/VARIABLE_VALUE'
    elif "dense_9" in model_name:
        c_name = "agent/_actor_network/_transformer/_position_emb/" 
        if "kernel" in model_name:
            c_name += "kernel/.ATTRIBUTES/VARIABLE_VALUE"
        elif "bias" in model_name:
            c_name += "bias/.ATTRIBUTES/VARIABLE_VALUE"
    elif "dense_8" in model_name:
        c_name = "agent/_actor_network/_transformer/_token_emb/"
        if "kernel" in model_name:
            c_name += "kernel/.ATTRIBUTES/VARIABLE_VALUE"
        elif "bias" in model_name:
            c_name += "bias/.ATTRIBUTES/VARIABLE_VALUE"

    else:
        # Action token embedding
        if 'dense_65' in c_name:
                c_name = "agent/_actor_network/_action_token_emb/"
                if "kernel" in model_name:
                    c_name += "kernel/.ATTRIBUTES/VARIABLE_VALUE"
                elif "bias" in model_name:
                    c_name += "bias/.ATTRIBUTES/VARIABLE_VALUE"
        
        # Image tokenizer (for non-token-learner parts)
        elif 'efficient_net_encoder' in model_name:
            if 'conv2d' in model_name:
                c_name = 'agent/_actor_network/_image_tokenizer/_tokenizer/conv1x1/kernel/.ATTRIBUTES/VARIABLE_VALUE'
            # film layers between efficientNet and TokenLearner
            elif 'film_conditioning_26' in c_name and 'dense_52' in c_name:
                c_name = 'agent/_actor_network/_image_tokenizer/_tokenizer/film_layer/_projection_add/'
                if 'kernel' in model_name:
                    c_name += 'kernel/.ATTRIBUTES/VARIABLE_VALUE'
                elif 'bias' in model_name:
                    c_name += 'bias/.ATTRIBUTES/VARIABLE_VALUE'
            elif 'film_conditioning_26' in c_name and 'dense_53' in c_name:
                c_name = 'agent/_actor_network/_image_tokenizer/_tokenizer/film_layer/_projection_mult/'
                if 'kernel' in model_name:
                    c_name += 'kernel/.ATTRIBUTES/VARIABLE_VALUE'
                elif 'bias' in model_name:
                    c_name += 'bias/.ATTRIBUTES/VARIABLE_VALUE'
        elif "stem_conv" in model_name:
            c_name = 'agent/_actor_network/_image_tokenizer/_tokenizer/net/layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE'
        elif "stem_bn" in model_name:
            c_name = 'agent/_actor_network/_image_tokenizer/_tokenizer/net/layer_with_weights-2'
            c_name = efficientnet_batchnorm_mapping(c_name, model_name)
        elif "top_conv" in model_name:
            c_name = 'agent/_actor_network/_image_tokenizer/_tokenizer/net/layer_with_weights-233/kernel/.ATTRIBUTES/VARIABLE_VALUE'
        elif "top_bn" in model_name:
            c_name = 'agent/_actor_network/_image_tokenizer/_tokenizer/net/layer_with_weights-234'
            c_name = efficientnet_batchnorm_mapping(c_name, model_name)
        elif "normalization" in model_name:
            if "count" in model_name:
                c_name = 'agent/_actor_network/_image_tokenizer/_tokenizer/net/layer_with_weights-0/count/.ATTRIBUTES/VARIABLE_VALUE'
            elif "mean" in model_name:
                c_name = 'agent/_actor_network/_image_tokenizer/_tokenizer/net/layer_with_weights-0/mean/.ATTRIBUTES/VARIABLE_VALUE'
            elif "variance" in model_name:
                c_name = 'agent/_actor_network/_image_tokenizer/_tokenizer/net/layer_with_weights-0/variance/.ATTRIBUTES/VARIABLE_VALUE'
        elif "block" in model_name:
            if "block1a" in model_name:
                start_index = 3
                c_name = efficientnet_block_mapping(c_name, model_name, start_index)
            elif "block1b" in model_name:
                start_index = 10
                c_name = efficientnet_block_mapping(c_name, model_name, start_index)
            elif "block2a" in model_name:
                start_index = 17
                c_name = efficientnet_block_mapping(c_name, model_name, start_index)
            elif "block2b" in model_name:
                start_index = 26
                c_name = efficientnet_block_mapping(c_name, model_name, start_index)
            elif "block2c" in model_name:
                start_index = 35
                c_name = efficientnet_block_mapping(c_name, model_name, start_index)
            elif "block3a" in model_name:
                start_index = 44
                c_name = efficientnet_block_mapping(c_name, model_name, start_index)
            elif "block3b" in model_name:
                start_index = 53
                c_name = efficientnet_block_mapping(c_name, model_name, start_index)
            elif "block3c" in model_name:
                start_index = 62
                c_name = efficientnet_block_mapping(c_name, model_name, start_index)
            elif "block4a" in model_name:
                start_index = 71
                c_name = efficientnet_block_mapping(c_name, model_name, start_index)
            elif "block4b" in model_name:
                start_index = 80
                c_name = efficientnet_block_mapping(c_name, model_name, start_index)
            elif "block4c" in model_name:
                start_index = 89
                c_name = efficientnet_block_mapping(c_name, model_name, start_index)
            elif "block4d" in model_name:
                start_index = 98
                c_name = efficientnet_block_mapping(c_name, model_name, start_index)
            elif "block4e" in model_name:
                start_index = 107
                c_name = efficientnet_block_mapping(c_name, model_name, start_index)
            elif "block5a" in model_name:
                start_index = 116
                c_name = efficientnet_block_mapping(c_name, model_name, start_index)
            elif "block5b" in model_name:
                start_index = 125
                c_name = efficientnet_block_mapping(c_name, model_name, start_index)
            elif "block5c" in model_name:
                start_index = 134
                c_name = efficientnet_block_mapping(c_name, model_name, start_index)
            elif "block5d" in model_name:
                start_index = 143
                c_name = efficientnet_block_mapping(c_name, model_name, start_index)
            elif "block5e" in model_name:
                start_index = 152
                c_name = efficientnet_block_mapping(c_name, model_name, start_index)
            elif "block6a" in model_name:
                start_index = 161
                c_name = efficientnet_block_mapping(c_name, model_name, start_index)
            elif "block6b" in model_name:
                start_index = 170
                c_name = efficientnet_block_mapping(c_name, model_name, start_index)
            elif "block6c" in model_name:
                start_index = 179
                c_name = efficientnet_block_mapping(c_name, model_name, start_index)
            elif "block6d" in model_name:
                start_index = 188
                c_name = efficientnet_block_mapping(c_name, model_name, start_index)
            elif "block6e" in model_name:
                start_index = 197
                c_name = efficientnet_block_mapping(c_name, model_name, start_index)
            elif "block6f" in model_name:
                start_index = 206
                c_name = efficientnet_block_mapping(c_name, model_name, start_index)
            elif "block7a" in model_name:
                start_index = 215
                c_name = efficientnet_block_mapping(c_name, model_name, start_index)
            elif "block7b" in model_name:
                start_index = 224
                c_name = efficientnet_block_mapping(c_name, model_name, start_index)

        # for block_name, layer_name in efficientnet_layer_mapping.items():
        #     if block_name in c_name:
        #         c_name = c_name.replace(block_name, layer_name)
        #         break
        
        # Add .ATTRIBUTES/VARIABLE_VALUE suffix for checkpoint format
        if not c_name.endswith('/.ATTRIBUTES/VARIABLE_VALUE'):
            c_name = c_name + '/.ATTRIBUTES/VARIABLE_VALUE'
    
    return c_name

def custom_load_checkpoint(model, checkpoint_path):
    """Custom checkpoint loading with name mapping."""
    print(f"\n=== MODEL VARIABLES IN CURRENT ARCHITECTURE ===")
    for var in model.variables:
        print(f"{var.name}: {var.shape}")
    print(f"Total model variables: {len(model.variables)}\n")
    print(f"\n" + "="*80)
    print("CUSTOM CHECKPOINT LOADING WITH NAME MAPPING")
    print("="*80)
    
    try:
        # Get checkpoint variables
        checkpoint_vars = dict(tf.train.list_variables(checkpoint_path))
        checkpoint_names = set(checkpoint_vars.keys())
        
        # Get model variables
        model_vars = {var.name: var for var in model.variables}
        model_names = set(model_vars.keys())
        
        print(f"Checkpoint variables: {len(checkpoint_names)}")
        print(f"Model variables: {len(model_names)}")
        
        # Create mapping from model names to checkpoint names
        name_mapping = {}
        print(f"\n🔄 CREATING NAME MAPPINGS...")
        for model_name in model_names:
            checkpoint_name = map_model_name_to_checkpoint_name(model_name)
            name_mapping[model_name] = checkpoint_name
        
        # Show sample mappings (before checking if they exist in checkpoint)
        print(f"\n📋 SAMPLE MAPPINGS (before checkpoint verification):")
        sample_mappings = list(name_mapping.items())[:15]  # Show first 15
        for i, (model_name, mapped_name) in enumerate(sample_mappings, 1):
            print(f"\n{i:2d}. MODEL:  {model_name}")
            print(f"    MAPPED: {mapped_name}")
        if len(name_mapping) > 15:
            print(f"\n    ... and {len(name_mapping) - 15} more mappings")
        
        # Count matches
        matched_vars = []
        unmatched_model_vars = []
        unmatched_checkpoint_vars = []
        
        for model_name, checkpoint_name in name_mapping.items():
            if checkpoint_name in checkpoint_names:
                matched_vars.append((model_name, checkpoint_name))
            else:
                unmatched_model_vars.append(model_name)
        
        # Find checkpoint variables not matched
        matched_checkpoint_names = set(name_mapping.values()) & checkpoint_names
        unmatched_checkpoint_vars = checkpoint_names - matched_checkpoint_names
        
        print(f"\n📊 MAPPING RESULTS:")
        print(f"  ✅ Matched variables: {len(matched_vars)}")
        print(f"  ❌ Model variables not in checkpoint: {len(unmatched_model_vars)}")
        print(f"  ❌ Checkpoint variables not in model: {len(unmatched_checkpoint_vars)}")
        
        # Analyze which mapping strategies worked best
        strategy_analysis = {
            'transformer_layers': 0,
            'token_learner': 0,
            'efficientnet': 0,
            'action_embedding': 0,
            'image_tokenizer': 0,
            'film_layers': 0,
            'other': 0
        }
        
        for model_name, checkpoint_name in matched_vars:
            if 'transformer/layers/' in checkpoint_name:
                strategy_analysis['transformer_layers'] += 1
            elif '_token_learner/' in checkpoint_name:
                strategy_analysis['token_learner'] += 1
            elif 'layer_with_weights-' in checkpoint_name:
                strategy_analysis['efficientnet'] += 1
            elif '_action_token_emb' in checkpoint_name:
                strategy_analysis['action_embedding'] += 1
            elif '_image_tokenizer' in checkpoint_name and '_token_learner' not in checkpoint_name:
                strategy_analysis['image_tokenizer'] += 1
            elif 'film_layer' in checkpoint_name:
                strategy_analysis['film_layers'] += 1
            else:
                strategy_analysis['other'] += 1
        
        print(f"\n📈 MAPPING STRATEGY ANALYSIS:")
        for strategy, count in strategy_analysis.items():
            if count > 0:
                print(f"  {strategy}: {count} variables")
        
        # Show sample mappings
        if matched_vars:
            print(f"\n✅ Sample successful mappings:")
            for i, (model_name, checkpoint_name) in enumerate(matched_vars[:20], 1):
                print(f"\n{i:2d}. MODEL:     {model_name}")
                print(f"    CHECKPOINT: {checkpoint_name}")
            if len(matched_vars) > 20:
                print(f"\n    ... and {len(matched_vars) - 20} more")
            
            # Show detailed examples of first 5 matches
            print(f"\n📋 DETAILED EXAMPLES (first 5 matches):")
            for i, (model_name, checkpoint_name) in enumerate(matched_vars[:5]):
                print(f"\n{i+1}. MATCH:")
                print(f"   Model:     {model_name}")
                print(f"   Checkpoint: {checkpoint_name}")
                
                # Show the mapping transformation
                model_clean = model_name.split(':')[0]
                print(f"   Mapping:   {model_clean} → {checkpoint_name}")
                
                # Get variable shapes if available
                try:
                    model_var = model_vars[model_name]
                    checkpoint_shape = checkpoint_vars[checkpoint_name]
                    print(f"   Shapes:    Model {model_var.shape} ← Checkpoint {checkpoint_shape}")
                except:
                    pass
        
        if unmatched_model_vars:
            print(f"\n❌ ALL UNMATCHED MODEL VARIABLES ({len(unmatched_model_vars)} total):")
            for i, var_name in enumerate(unmatched_model_vars, 1):
                mapped_name = map_model_name_to_checkpoint_name(var_name)
                print(f"\n{i:4d}. MODEL:  {var_name}")
                print(f"     MAPPED: {mapped_name}")
            
            # Also save unmatched variables to a file for easier analysis
            unmatched_file = "unmatched_variables.txt"
            with open(unmatched_file, 'w') as f:
                f.write(f"ALL UNMATCHED MODEL VARIABLES ({len(unmatched_model_vars)} total)\n")
                f.write("=" * 60 + "\n\n")
                for i, var_name in enumerate(unmatched_model_vars, 1):
                    mapped_name = map_model_name_to_checkpoint_name(var_name)
                    f.write(f"{i:4d}. MODEL:  {var_name}\n")
                    f.write(f"     MAPPED: {mapped_name}\n\n")
            print(f"\n💾 All unmatched variables saved to '{unmatched_file}'")
        
        # Load matched variables
        if matched_vars:
            print(f"\n🔄 Loading {len(matched_vars)} variables...")
            
            # Create a custom checkpoint reader
            reader = tf.train.load_checkpoint(checkpoint_path)
            
            loaded_count = 0
            for model_name, checkpoint_name in matched_vars:
                try:
                    # Get the variable from checkpoint
                    checkpoint_tensor = reader.get_tensor(checkpoint_name)
                    
                    # Get the model variable
                    model_var = model_vars[model_name]
                    
                    # Check shape compatibility
                    if checkpoint_tensor.shape == model_var.shape:
                        # Assign the checkpoint value to the model variable
                        model_var.assign(checkpoint_tensor)
                        loaded_count += 1
                    else:
                        print(f"  ⚠️  Shape mismatch: {model_name} ({model_var.shape}) vs {checkpoint_name} ({checkpoint_tensor.shape})")
                except Exception as e:
                    print(f"  ❌ Failed to load {model_name}: {e}")
            
            print(f"✅ Successfully loaded {loaded_count}/{len(matched_vars)} variables")
            return loaded_count, len(matched_vars)
        else:
            print("❌ No variables matched for loading")
            return 0, 0
            
    except Exception as e:
        print(f"❌ ERROR in custom loading: {e}")
        return 0, 0

def get_weight_stats(weights):
    """Return a dict of weight name to a tuple of (mean, std) for quick comparison."""
    stats = {}
    for w in weights:
        arr = w.numpy()
        stats[w.name] = (arr.mean(), arr.std())
    return stats

def compare_weight_stats(before_stats, after_stats, logger=print):
    """Compare two weight stats dicts and log/print which weights changed."""
    changed = []
    unchanged = []
    for name in before_stats:
        if name in after_stats:
            before = before_stats[name]
            after = after_stats[name]
            # If mean or std changes by more than a tiny epsilon, consider changed
            if not (np.isclose(before[0], after[0], atol=1e-6) and np.isclose(before[1], after[1], atol=1e-6)):
                changed.append(name)
            else:
                unchanged.append(name)
    logger(f"Weights changed after loading: {len(changed)}")
    logger(f"Weights unchanged after loading: {len(unchanged)}")
    return changed, unchanged

def print_checkpoint_loading_details(model, checkpoint_path):
    """Print detailed information about checkpoint loading."""
    try:
        # List variables in checkpoint
        checkpoint_vars = dict(tf.train.list_variables(checkpoint_path))
        checkpoint_names = set(checkpoint_vars.keys())
        
        # List model variable names (remove :0 suffix)
        model_names = set([v.name.split(':')[0] for v in model.variables])
        
        # Find matches and mismatches
        loaded = checkpoint_names & model_names
        not_loaded = model_names - checkpoint_names
        not_in_model = checkpoint_names - model_names

        print(f"\n" + "="*80)
        print("CHECKPOINT VARIABLE ANALYSIS")
        print("="*80)
        print(f"Variables in checkpoint: {len(checkpoint_names)}")
        print(f"Variables in model: {len(model_names)}")
        print(f"Variables loaded from checkpoint: {len(loaded)}")
        print(f"Variables in model NOT loaded from checkpoint: {len(not_loaded)}")
        print(f"Variables in checkpoint NOT in model: {len(not_in_model)}")
        
        if loaded:
            print(f"\n✅ Variables loaded from checkpoint:")
            for name in sorted(loaded)[:10]:  # Show first 10
                print(f"  {name}")
            if len(loaded) > 10:
                print(f"  ... and {len(loaded) - 10} more")
        
        if not_loaded:
            print(f"\n❌ Variables in model NOT loaded from checkpoint:")
            for name in sorted(not_loaded)[:10]:  # Show first 10
                print(f"  {name}")
            if len(not_loaded) > 10:
                print(f"  ... and {len(not_loaded) - 10} more")
        
        if not_in_model:
            print(f"\n⚠️  Variables in checkpoint NOT in model:")
            for name in sorted(not_in_model)[:10]:  # Show first 10
                print(f"  {name}")
            if len(not_in_model) > 10:
                print(f"  ... and {len(not_in_model) - 10} more")
        
        # Calculate success rate
        if len(checkpoint_names) > 0:
            success_rate = len(loaded) / len(checkpoint_names) * 100
            print(f"\n📈 Name-based success rate: {success_rate:.1f}% ({len(loaded)}/{len(checkpoint_names)})")
        
        return loaded, not_loaded, not_in_model
        
    except Exception as e:
        print(f"❌ ERROR: Could not analyze checkpoint variables: {e}")
        return set(), set(), set()

def test_checkpoint_loading():
    """Test loading the checkpoint and count matches/mismatches."""
    
    # Load gin config
    gin.parse_config_file(FLAGS.config_file)
    
    # Create specs
    observation_spec = create_observation_spec()
    action_spec = create_bbox_action_spec()
    
    # Create time step spec
    time_step_spec = ts.time_step_spec(observation_spec=observation_spec)
    
    # Create agent (EfficientNet loads ImageNet weights)
    agent = create_agent(time_step_spec, action_spec)
    
    # Print all model variable names
    print("="*80)
    print("ALL MODEL VARIABLE NAMES")
    print("="*80)
    model_variables = {var.name: var for var in agent.variables}
    for i, (var_name, var) in enumerate(model_variables.items(), 1):
        print(f"{i:4d}. {var_name}")
    print(f"\nTotal model variables: {len(model_variables)}")
    
    # Capture stats AFTER EfficientNet loading but BEFORE RT-1 loading
    before_stats = get_weight_stats(agent.variables)
    
    # Try to load the checkpoint
    checkpoint_path = FLAGS.checkpoint_path
    print(f"\nAttempting to load checkpoint from: {checkpoint_path}")
    print(f"Current working directory: {os.getcwd()}")
    
    # List files in the checkpoint directory for debugging
    checkpoint_dir = os.path.dirname(checkpoint_path)
    if os.path.exists(checkpoint_dir):
        print(f"Files in {checkpoint_dir}:")
        for file in os.listdir(checkpoint_dir):
            print(f"  {file}")
    else:
        print(f"❌ ERROR: Checkpoint directory not found: {checkpoint_dir}")
        return
    
    try:
        # Check if the checkpoint file exists
        if not os.path.exists(f"{checkpoint_path}.index"):
            print(f"❌ ERROR: Checkpoint index file not found: {checkpoint_path}.index")
            return
        
        print(f"Found checkpoint files:")
        print(f"  - {checkpoint_path}")
        print(f"  - {checkpoint_path}.index")
        
        # Use custom loading function
        loaded_count, total_matched = custom_load_checkpoint(agent, checkpoint_path)
        
        # Capture stats after custom loading
        after_stats = get_weight_stats(agent.variables)
        
        # Compare weight stats
        changed_weights, unchanged_weights = compare_weight_stats(before_stats, after_stats)
        
        print(f"\n" + "="*80)
        print("WEIGHT CHANGE ANALYSIS")
        print("="*80)
        print(f"✅ Weights that changed (loaded from checkpoint): {len(changed_weights)}")
        print(f"❌ Weights that unchanged (randomly initialized): {len(unchanged_weights)}")
        
        if changed_weights:
            print(f"\n✅ Sample changed weights:")
            for weight_name in changed_weights[:10]:
                print(f"  {weight_name}")
            if len(changed_weights) > 10:
                print(f"  ... and {len(changed_weights) - 10} more")
        
        if unchanged_weights:
            print(f"\n❌ Sample unchanged weights:")
            for weight_name in unchanged_weights[:10]:
                print(f"  {weight_name}")
            if len(unchanged_weights) > 10:
                print(f"  ... and {len(unchanged_weights) - 10} more")
        
        # Calculate weight-based success rate
        total_weights = len(changed_weights) + len(unchanged_weights)
        if total_weights > 0:
            weight_success_rate = len(changed_weights) / total_weights * 100
            print(f"\n📈 Weight-based success rate: {weight_success_rate:.1f}% ({len(changed_weights)}/{total_weights})")
        
        print(f"\n📊 CUSTOM LOADING SUMMARY:")
        print(f"  Variables matched by name mapping: {total_matched}")
        print(f"  Variables successfully loaded: {loaded_count}")
        print(f"  Variables with weight changes: {len(changed_weights)}")
        
        if loaded_count > 0:
            print("✅ Custom checkpoint loading completed successfully!")
        else:
            print("❌ No variables were loaded with custom mapping")
    
    except Exception as e:
        print(f"❌ ERROR: Failed to load checkpoint: {e}")
        
        # Check if this is a shape mismatch (which means names are matching)
        if "shape" in str(e).lower():
            print("This suggests the checkpoint format or path is not compatible.")
            print("Please check:")
            print("1. The checkpoint path is correct")
            print("2. The checkpoint format is compatible")
            print("3. The model architecture matches the checkpoint")
    
    print("✅ Checkpoint loading test completed successfully!")

def main(argv):
    test_checkpoint_loading()

if __name__ == '__main__':
    app.run(main) 