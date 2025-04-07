#!/usr/bin/env python3
"""
Test script to verify the integrity of a converted MLX model.
This script loads the model and prints its structure to verify it was converted correctly.
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
logger = logging.getLogger('test-mlx-model')

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

def test_custom_mlx_model(model_path):
    """Test a model saved in our custom MLX format"""
    model_path = Path(model_path)
    tensor_dir = model_path.with_suffix('.tensors')
    
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
    
    # Check metadata file
    metadata_file = model_path.with_suffix('.metadata.json')
    if metadata_file.exists():
        try:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
                logger.info(f"Metadata contains {len(metadata)} entries")
                logger.info(f"Tensor keys: {list(metadata.keys())[:10]}...")
        except Exception as e:
            logger.error(f"Error loading metadata: {e}")
    else:
        logger.warning(f"Metadata file not found: {metadata_file}")
    
    # Check config file
    config_file = model_path.with_suffix('.config.json')
    if config_file.exists():
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
                logger.info(f"Model config:")
                for key, value in list(config.items())[:10]:
                    logger.info(f"  {key}: {value}")
                
                # Check model size
                if 'hidden_size' in config:
                    logger.info(f"Model hidden size: {config['hidden_size']}")
                if 'num_hidden_layers' in config:
                    logger.info(f"Number of layers: {config['num_hidden_layers']}")
                if 'vocab_size' in config:
                    logger.info(f"Vocabulary size: {config['vocab_size']}")
        except Exception as e:
            logger.error(f"Error loading config: {e}")
    else:
        logger.warning(f"Config file not found: {config_file}")
    
    # Check tokenizer file
    tokenizer_file = model_path.with_suffix('.tokenizer.json')
    if tokenizer_file.exists():
        logger.info(f"Tokenizer file found: {tokenizer_file}")
    else:
        logger.warning(f"Tokenizer file not found: {tokenizer_file}")
    
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
    
    # Load a few tensors to verify they're valid
    if tensor_sizes:
        for name, _ in tensor_sizes[:3]:
            try:
                tensor_path = tensor_dir / name
                tensor = np.load(tensor_path)
                logger.info(f"Successfully loaded tensor {name} with shape {tensor.shape} and dtype {tensor.dtype}")
            except Exception as e:
                logger.error(f"Error loading tensor {name}: {e}")
    
    return True

def test_standard_mlx_model(model_path):
    """Test a model saved in standard MLX format"""
    try:
        import mlx.core as mx
        
        logger.info(f"Loading model from {model_path}")
        weights = mx.load(model_path)
        
        logger.info(f"Model contains {len(weights)} tensors")
        
        # Print some keys
        keys = list(weights.keys())
        logger.info(f"First 10 keys: {keys[:10]}")
        
        # Print some tensor shapes
        for key in keys[:5]:
            logger.info(f"Tensor {key} shape: {weights[key].shape}")
        
        # Calculate total size
        total_size = 0
        for key, tensor in weights.items():
            # Get size in bytes
            size = tensor.size * tensor.dtype.itemsize
            total_size += size
        
        logger.info(f"Total model size: {total_size / (1024 * 1024):.2f} MB")
        
        return True
    except Exception as e:
        logger.error(f"Error testing standard MLX model: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return False

def main():
    parser = argparse.ArgumentParser(description='Test MLX model integrity')
    parser.add_argument('--model', type=str, required=True, help='Path to the MLX model file')
    parser.add_argument('--mlx-dir', type=str, help='Path to MLX directory if not installed')
    args = parser.parse_args()
    
    # Setup MLX
    if not setup_mlx_path(args.mlx_dir):
        return 1
    
    model_path = Path(args.model)
    if not model_path.exists():
        logger.error(f"Model file not found: {model_path}")
        return 1
    
    # Check if it's a custom format
    tensor_dir = model_path.with_suffix('.tensors')
    if tensor_dir.exists() and tensor_dir.is_dir():
        logger.info(f"Detected custom MLX format")
        if test_custom_mlx_model(model_path):
            logger.info(f"Custom MLX model test passed")
            return 0
        else:
            logger.error(f"Custom MLX model test failed")
            return 1
    else:
        # Try standard format
        logger.info(f"Testing standard MLX format")
        if test_standard_mlx_model(model_path):
            logger.info(f"Standard MLX model test passed")
            return 0
        else:
            logger.error(f"Standard MLX model test failed")
            return 1

if __name__ == '__main__':
    sys.exit(main())
