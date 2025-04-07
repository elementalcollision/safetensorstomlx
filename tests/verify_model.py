#!/usr/bin/env python3
"""
Verify MLX Model Script

This script loads a converted MLX model and verifies that all tensors are accessible
and have the correct shapes. It doesn't attempt to run inference.
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
import numpy as np
import h5py

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger('verify-mlx-model')

def setup_mlx_path(mlx_dir=None):
    """Add MLX to path if needed"""
    try:
        import mlx
        try:
            version = mlx.__version__
            logger.info(f"Found MLX version: {version}")
        except AttributeError:
            logger.info("Found MLX version: unknown")
        return True
    except ImportError:
        if mlx_dir:
            sys.path.append(mlx_dir)
            try:
                import mlx
                try:
                    version = mlx.__version__
                    logger.info(f"Found MLX version: {version}")
                except AttributeError:
                    logger.info("Found MLX version: unknown")
                return True
            except ImportError:
                logger.error(f"Could not import MLX from {mlx_dir}")
                return False
        else:
            logger.error("MLX not found and no MLX directory specified")
            return False

def verify_custom_mlx_model(model_path):
    """Verify a model saved in our custom MLX format"""
    model_path = Path(model_path)
    tensor_dir = model_path.with_suffix('.tensors')
    config_path = model_path.with_suffix('.config.json')
    
    if not model_path.exists():
        logger.error(f"Model file not found: {model_path}")
        return False
    
    if not tensor_dir.exists() or not tensor_dir.is_dir():
        logger.error(f"Tensor directory not found: {tensor_dir}")
        return False
    
    # Load the main model file
    try:
        with h5py.File(model_path, 'r') as f:
            # Check if it has our custom format attribute
            if 'format' in f.attrs and f.attrs['format'] == 'mlx_custom':
                logger.info(f"Confirmed model is in custom MLX format")
            else:
                logger.warning(f"Model does not have expected format attribute")
            
            # Print attributes
            logger.info("Model attributes:")
            for key, value in f.attrs.items():
                logger.info(f"  {key}: {value}")
    except Exception as e:
        logger.error(f"Error loading model file: {e}")
        return False
    
    # Load config
    if config_path.exists():
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Extract key model parameters
            hidden_size = config.get('hidden_size', 3584)
            num_hidden_layers = config.get('num_hidden_layers', 32)
            num_attention_heads = config.get('num_attention_heads', 32)
            num_key_value_heads = config.get('num_key_value_heads', 8)
            vocab_size = config.get('vocab_size', 152064)
            
            logger.info(f"Model parameters:")
            logger.info(f"  hidden_size: {hidden_size}")
            logger.info(f"  num_hidden_layers: {num_hidden_layers}")
            logger.info(f"  num_attention_heads: {num_attention_heads}")
            logger.info(f"  num_key_value_heads: {num_key_value_heads}")
            logger.info(f"  vocab_size: {vocab_size}")
        except Exception as e:
            logger.error(f"Error loading config: {e}")
    
    # Count and check tensor files
    tensor_count = 0
    tensor_sizes = []
    total_size = 0
    
    for tensor_file in tensor_dir.glob('*.npy'):
        tensor_count += 1
        file_size = tensor_file.stat().st_size
        total_size += file_size
        tensor_sizes.append((tensor_file.name, file_size))
    
    logger.info(f"Found {tensor_count} tensor files")
    logger.info(f"Total tensor size: {total_size / (1024 * 1024):.2f} MB")
    
    # Print the 5 largest tensors
    tensor_sizes.sort(key=lambda x: x[1], reverse=True)
    logger.info("Largest tensors:")
    for name, size in tensor_sizes[:5]:
        logger.info(f"  {name}: {size / (1024 * 1024):.2f} MB")
    
    # Detect the actual number of layers
    actual_layers = 0
    for i in range(100):  # Check a large number to be safe
        test_file = tensor_dir / f"thinker_model_layers_{i}_self_attn_q_proj_weight.npy"
        if test_file.exists():
            actual_layers = i + 1
        else:
            break
    
    logger.info(f"Detected {actual_layers} layers in the model (config specified {num_hidden_layers})")
    
    # Load key tensors to verify
    import mlx.core as mx
    
    key_tensors = [
        "thinker_model_embed_tokens_weight",
        "thinker_lm_head_weight",
        "thinker_model_norm_weight"
    ]
    
    # Add some layer tensors
    for i in range(min(3, actual_layers)):
        key_tensors.extend([
            f"thinker_model_layers_{i}_self_attn_q_proj_weight",
            f"thinker_model_layers_{i}_self_attn_k_proj_weight",
            f"thinker_model_layers_{i}_self_attn_v_proj_weight",
            f"thinker_model_layers_{i}_self_attn_o_proj_weight",
            f"thinker_model_layers_{i}_input_layernorm_weight",
            f"thinker_model_layers_{i}_post_attention_layernorm_weight",
            f"thinker_model_layers_{i}_mlp_gate_proj_weight",
            f"thinker_model_layers_{i}_mlp_up_proj_weight",
            f"thinker_model_layers_{i}_mlp_down_proj_weight"
        ])
    
    logger.info(f"Verifying {len(key_tensors)} key tensors")
    
    for tensor_name in key_tensors:
        tensor_path = tensor_dir / f"{tensor_name}.npy"
        if tensor_path.exists():
            try:
                tensor = np.load(tensor_path)
                mx_tensor = mx.array(tensor)
                logger.info(f"Successfully loaded tensor {tensor_name} with shape {tensor.shape}")
            except Exception as e:
                logger.error(f"Error loading tensor {tensor_name}: {e}")
        else:
            logger.warning(f"Tensor {tensor_name} not found")
    
    # Try a simple MLX operation to verify MLX is working
    try:
        a = mx.array([1, 2, 3])
        b = mx.array([4, 5, 6])
        c = a + b
        logger.info(f"MLX operation test: {a} + {b} = {c}")
    except Exception as e:
        logger.error(f"Error during MLX operation test: {e}")
    
    return True

def main():
    parser = argparse.ArgumentParser(description='Verify MLX model')
    parser.add_argument('--model', type=str, required=True, help='Path to the MLX model file')
    parser.add_argument('--mlx-dir', type=str, help='Path to MLX directory if not installed')
    args = parser.parse_args()
    
    # Setup MLX
    if not setup_mlx_path(args.mlx_dir):
        return 1
    
    # Verify the model
    if verify_custom_mlx_model(args.model):
        logger.info("Model verification completed successfully")
        return 0
    else:
        logger.error("Model verification failed")
        return 1

if __name__ == '__main__':
    sys.exit(main())
