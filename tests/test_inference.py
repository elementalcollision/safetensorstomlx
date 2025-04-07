#!/usr/bin/env python3
"""
Test inference script for MLX models converted from SafeTensors.
This script loads the model and tokenizer, then runs inference on a sample prompt.
"""

import os
import sys
import json
import time
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
logger = logging.getLogger('test-inference')

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

class CustomMLXModel:
    """A wrapper for models saved in our custom MLX format"""
    
    def __init__(self, model_path):
        self.model_path = Path(model_path)
        self.tensor_dir = self.model_path.with_suffix('.tensors')
        self.config_path = self.model_path.with_suffix('.config.json')
        self.metadata_path = self.model_path.with_suffix('.metadata.json')
        self.tokenizer_path = self.model_path.with_suffix('.tokenizer.json')
        
        # Load metadata
        with h5py.File(self.model_path, 'r') as f:
            self.num_tensors = f.attrs.get('num_tensors', 0)
        
        # Load config
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                self.config = json.load(f)
        else:
            self.config = {}
        
        # Initialize weights cache
        self.weights_cache = {}
    
    def get_tensor(self, name):
        """Load a tensor by name"""
        if name in self.weights_cache:
            return self.weights_cache[name]
        
        tensor_path = self.tensor_dir / f"{name}.npy"
        if not tensor_path.exists():
            raise ValueError(f"Tensor {name} not found")
        
        tensor = np.load(tensor_path)
        
        # Convert to MLX array
        import mlx.core as mx
        mx_tensor = mx.array(tensor)
        
        # Cache for future use
        self.weights_cache[name] = mx_tensor
        
        return mx_tensor
    
    def get_tokenizer(self):
        """Load the tokenizer"""
        if not self.tokenizer_path.exists():
            raise ValueError(f"Tokenizer not found at {self.tokenizer_path}")
        
        # Use transformers tokenizer if available
        try:
            from transformers import PreTrainedTokenizerFast
            tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(self.tokenizer_path))
            
            # Set special tokens based on config
            if 'bos_token_id' in self.config:
                tokenizer.bos_token_id = self.config['bos_token_id']
            if 'eos_token_id' in self.config:
                tokenizer.eos_token_id = self.config['eos_token_id']
            if 'pad_token_id' in self.config:
                tokenizer.pad_token_id = self.config['pad_token_id']
            
            return tokenizer
        except ImportError:
            logger.warning("transformers not available, using basic tokenizer")
            
            # Implement a basic tokenizer
            try:
                # Try to read the tokenizer file with detailed error handling
                with open(self.tokenizer_path, 'r', encoding='utf-8') as f:
                    try:
                        tokenizer_data = json.load(f)
                        logger.info(f"Successfully loaded tokenizer data with {len(tokenizer_data)} entries")
                    except json.JSONDecodeError as e:
                        # Read the first few lines to debug
                        f.seek(0)
                        first_lines = ''.join([f.readline() for _ in range(5)])
                        logger.error(f"JSON decode error: {e}")
                        logger.error(f"First few lines of tokenizer file:\n{first_lines}")
                        raise
            except Exception as e:
                logger.error(f"Error reading tokenizer file: {e}")
                raise
            
            # This is a very basic implementation
            class BasicTokenizer:
                def __init__(self, tokenizer_data):
                    self.tokenizer_data = tokenizer_data
                    self.bos_token_id = 151643  # Default for Qwen models
                    self.eos_token_id = 151644
                
                def encode(self, text):
                    # This is just a placeholder - would need actual implementation
                    return [self.bos_token_id] + [0] * 10  # Dummy tokens
                
                def decode(self, ids):
                    # This is just a placeholder - would need actual implementation
                    return "Decoded text (placeholder)"
            
            return BasicTokenizer(tokenizer_data)

def load_model_and_tokenizer(model_path):
    """Load the model and tokenizer"""
    model_path = Path(model_path)
    
    # Check if it's our custom format
    tensor_dir = model_path.with_suffix('.tensors')
    if tensor_dir.exists() and tensor_dir.is_dir():
        logger.info("Loading custom MLX format model")
        
        try:
            # Load the model
            model = CustomMLXModel(model_path)
            logger.info(f"Successfully loaded model from {model_path}")
            
            # Load the tokenizer
            try:
                tokenizer = model.get_tokenizer()
                logger.info(f"Successfully loaded tokenizer")
            except Exception as e:
                logger.error(f"Error loading tokenizer: {e}")
                # Create a dummy tokenizer for testing
                class DummyTokenizer:
                    def encode(self, text):
                        return [0] * 10
                    def decode(self, ids):
                        return "[Dummy tokenizer output]"
                tokenizer = DummyTokenizer()
                logger.info("Using dummy tokenizer for testing")
            
            return model, tokenizer
        except Exception as e:
            logger.error(f"Error loading custom model: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            raise
    else:
        # Try standard MLX format
        logger.info("Loading standard MLX format model")
        
        import mlx.core as mx
        
        # Load weights
        weights = mx.load(str(model_path))
        
        # Load tokenizer from the same directory
        tokenizer_path = model_path.parent / "tokenizer.json"
        if not tokenizer_path.exists():
            tokenizer_path = model_path.with_suffix('.tokenizer.json')
        
        if tokenizer_path.exists():
            try:
                from transformers import PreTrainedTokenizerFast
                tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(tokenizer_path))
            except Exception as e:
                logger.error(f"Error loading tokenizer: {e}")
                # Create a dummy tokenizer for testing
                class DummyTokenizer:
                    def encode(self, text):
                        return [0] * 10
                    def decode(self, ids):
                        return "[Dummy tokenizer output]"
                tokenizer = DummyTokenizer()
                logger.info("Using dummy tokenizer for testing")
        else:
            logger.warning(f"Tokenizer not found at {tokenizer_path}, using dummy tokenizer")
            # Create a dummy tokenizer for testing
            class DummyTokenizer:
                def encode(self, text):
                    return [0] * 10
                def decode(self, ids):
                    return "[Dummy tokenizer output]"
            tokenizer = DummyTokenizer()
        
        return weights, tokenizer

def run_simple_inference(model, tokenizer, prompt, max_tokens=100):
    """Run a simple inference test"""
    logger.info(f"Running inference with prompt: {prompt}")
    
    # This is a simplified implementation - in a real scenario, 
    # you would need to implement the actual model architecture
    
    # For custom format models, we'll just check if key tensors exist
    if isinstance(model, CustomMLXModel):
        logger.info("Testing tensor loading for custom format model")
        
        # Check if key tensors exist
        key_tensors = [
            "thinker_model_embed_tokens_weight",
            "thinker_lm_head_weight",
            "thinker_model_norm_weight"
        ]
        
        for tensor_name in key_tensors:
            try:
                tensor = model.get_tensor(tensor_name)
                logger.info(f"Successfully loaded tensor {tensor_name} with shape {tensor.shape}")
            except Exception as e:
                logger.error(f"Error loading tensor {tensor_name}: {e}")
        
        # Since we don't have the full model implementation, we'll return a placeholder
        logger.info("Full inference not implemented for custom format - this is just a test of tensor loading")
        return "This is a placeholder response. Full model inference would require implementing the model architecture."
    
    # For standard MLX models
    else:
        # This would need to be implemented based on the model architecture
        logger.info("Full inference not implemented for standard format - this is just a test of model loading")
        return "This is a placeholder response. Full model inference would require implementing the model architecture."

def main():
    parser = argparse.ArgumentParser(description='Test MLX model inference')
    parser.add_argument('--model', type=str, required=True, help='Path to the MLX model file')
    parser.add_argument('--prompt', type=str, default="Hello, how are you today?", 
                        help='Prompt for inference')
    parser.add_argument('--max-tokens', type=int, default=100, 
                        help='Maximum number of tokens to generate')
    parser.add_argument('--mlx-dir', type=str, help='Path to MLX directory if not installed')
    args = parser.parse_args()
    
    # Setup MLX
    if not setup_mlx_path(args.mlx_dir):
        return 1
    
    try:
        # Load model and tokenizer
        model, tokenizer = load_model_and_tokenizer(args.model)
        
        # Run inference
        start_time = time.time()
        output = run_simple_inference(model, tokenizer, args.prompt, args.max_tokens)
        end_time = time.time()
        
        logger.info(f"Inference completed in {end_time - start_time:.2f} seconds")
        logger.info(f"Output: {output}")
        
        return 0
    except Exception as e:
        logger.error(f"Error during inference: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return 1

if __name__ == '__main__':
    sys.exit(main())
