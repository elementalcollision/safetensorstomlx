#!/usr/bin/env python3
"""
Qwen2.5-Omni MLX Inference Script

This script implements the Qwen2.5-Omni model architecture in MLX and provides
a simple interface for text generation using the converted model.
"""

import os
import sys
import json
import time
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import h5py

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger('qwen-mlx-inference')

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
        
        # Extract key model parameters from config
        self.hidden_size = self.config.get('hidden_size', 3584)
        self.num_hidden_layers = self.config.get('num_hidden_layers', 32)
        self.num_attention_heads = self.config.get('num_attention_heads', 32)
        self.num_key_value_heads = self.config.get('num_key_value_heads', 8)
        self.intermediate_size = self.config.get('intermediate_size', 14336)
        self.vocab_size = self.config.get('vocab_size', 152064)
        self.max_position_embeddings = self.config.get('max_position_embeddings', 32768)
        self.rope_theta = self.config.get('rope_theta', 1000000.0)
        self.head_dim = self.hidden_size // self.num_attention_heads
        
        logger.info(f"Model parameters:")
        logger.info(f"  hidden_size: {self.hidden_size}")
        logger.info(f"  num_hidden_layers: {self.num_hidden_layers}")
        logger.info(f"  num_attention_heads: {self.num_attention_heads}")
        logger.info(f"  num_key_value_heads: {self.num_key_value_heads}")
        logger.info(f"  intermediate_size: {self.intermediate_size}")
        logger.info(f"  vocab_size: {self.vocab_size}")
    
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
        except Exception as e:
            logger.warning(f"Error loading tokenizer with transformers: {e}")
            
            # Implement a basic tokenizer
            try:
                # Try to load the original tokenizer.json from the model directory
                original_tokenizer_path = self.model_path.parent / "tokenizer.json"
                if original_tokenizer_path.exists():
                    with open(original_tokenizer_path, 'r', encoding='utf-8') as f:
                        tokenizer_data = json.load(f)
                        logger.info(f"Loaded original tokenizer from {original_tokenizer_path}")
                else:
                    # Try to read our saved tokenizer file
                    with open(self.tokenizer_path, 'r', encoding='utf-8') as f:
                        tokenizer_data = json.load(f)
            except Exception as e:
                logger.error(f"Error reading tokenizer file: {e}")
                # Create a minimal tokenizer data
                tokenizer_data = {"model": {"type": "BPE"}, "added_tokens": []}
                logger.warning("Using minimal tokenizer data")
            
            # This is a very basic implementation
            class BasicTokenizer:
                def __init__(self, tokenizer_data):
                    self.tokenizer_data = tokenizer_data
                    self.bos_token_id = 151643  # Default for Qwen models
                    self.eos_token_id = 151644
                
                def encode(self, text, return_tensors=None):
                    # This is just a placeholder - would need actual implementation
                    tokens = [self.bos_token_id] + [i % 100 + 1 for i in range(len(text))]
                    
                    if return_tensors == "mlx":
                        import mlx.core as mx
                        return mx.array([tokens])
                    return tokens
                
                def decode(self, ids):
                    # This is just a placeholder - would need actual implementation
                    if hasattr(ids, "tolist"):
                        ids = ids.tolist()
                    return f"[Decoded text of length {len(ids)}]"
            
            return BasicTokenizer(tokenizer_data)

class QwenMLXModel:
    """Implementation of Qwen2.5-Omni model in MLX"""
    
    def __init__(self, model):
        """
        Initialize the Qwen MLX model
        
        Args:
            model: CustomMLXModel instance with loaded weights
        """
        import mlx.core as mx
        import mlx.nn as nn
        
        self.model = model
        self.mx = mx
        self.nn = nn
        
        # Initialize model parameters
        self.params = {}
        self.load_weights()
    
    def load_weights(self):
        """Load weights from the model"""
        logger.info("Loading model weights")
        
        # Load key tensors
        try:
            # Embedding weights
            self.params["embed_tokens.weight"] = self.model.get_tensor("thinker_model_embed_tokens_weight")
            
            # Final layer norm
            self.params["norm.weight"] = self.model.get_tensor("thinker_model_norm_weight")
            
            # LM head
            self.params["lm_head.weight"] = self.model.get_tensor("thinker_lm_head_weight")
            
            # Detect the actual number of layers
            actual_layers = 0
            for i in range(100):  # Check a large number to be safe
                try:
                    # Try to load a tensor from this layer
                    test_tensor = self.model.get_tensor(f"thinker_model_layers_{i}_self_attn_q_proj_weight")
                    actual_layers = i + 1
                except ValueError:
                    # If tensor not found, we've reached the end of the layers
                    break
            
            logger.info(f"Detected {actual_layers} layers in the model (config specified {self.model.num_hidden_layers})")
            self.model.num_hidden_layers = actual_layers
            
            # Load transformer layers
            for i in range(self.model.num_hidden_layers):
                # Attention weights
                self.params[f"layers.{i}.self_attn.q_proj.weight"] = self.model.get_tensor(f"thinker_model_layers_{i}_self_attn_q_proj_weight")
                self.params[f"layers.{i}.self_attn.k_proj.weight"] = self.model.get_tensor(f"thinker_model_layers_{i}_self_attn_k_proj_weight")
                self.params[f"layers.{i}.self_attn.v_proj.weight"] = self.model.get_tensor(f"thinker_model_layers_{i}_self_attn_v_proj_weight")
                self.params[f"layers.{i}.self_attn.o_proj.weight"] = self.model.get_tensor(f"thinker_model_layers_{i}_self_attn_o_proj_weight")
                
                # Layer norms
                self.params[f"layers.{i}.input_layernorm.weight"] = self.model.get_tensor(f"thinker_model_layers_{i}_input_layernorm_weight")
                self.params[f"layers.{i}.post_attention_layernorm.weight"] = self.model.get_tensor(f"thinker_model_layers_{i}_post_attention_layernorm_weight")
                
                # MLP weights
                self.params[f"layers.{i}.mlp.gate_proj.weight"] = self.model.get_tensor(f"thinker_model_layers_{i}_mlp_gate_proj_weight")
                self.params[f"layers.{i}.mlp.up_proj.weight"] = self.model.get_tensor(f"thinker_model_layers_{i}_mlp_up_proj_weight")
                self.params[f"layers.{i}.mlp.down_proj.weight"] = self.model.get_tensor(f"thinker_model_layers_{i}_mlp_down_proj_weight")
            
            logger.info(f"Successfully loaded {len(self.params)} weight tensors")
        except Exception as e:
            logger.error(f"Error loading weights: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise
    
    def precompute_freqs_cis(self, seq_len):
        """Precompute the frequency cis for rotary embeddings"""
        import math
        theta = float(self.model.rope_theta)
        dim = int(self.model.head_dim)
        
        # Create the base frequencies
        inv_freq = 1.0 / (theta ** (self.mx.arange(0, dim, 2, dtype=self.mx.float32) / dim))
        
        # Create position indices
        t = self.mx.arange(seq_len, dtype=self.mx.float32)
        
        # Compute the outer product for all position and frequency combinations
        freqs = self.mx.outer(t, inv_freq)
        
        # Convert to complex numbers using Euler's formula
        freqs_cis = self.mx.complex(self.mx.cos(freqs), self.mx.sin(freqs))
        
        return freqs_cis
    
    def apply_rotary_emb(self, x, freqs_cis):
        """Apply rotary embeddings to the query and key tensors"""
        x_complex = self.mx.complex(x[..., ::2], x[..., 1::2])
        x_rotated = x_complex * freqs_cis.reshape(1, freqs_cis.shape[0], 1, freqs_cis.shape[1])
        x_out = self.mx.concatenate(
            [self.mx.real(x_rotated), self.mx.imag(x_rotated)], axis=-1
        )
        return x_out
    
    def forward_attention(self, x, freqs_cis, layer_idx, mask=None):
        """Forward pass through a single attention layer"""
        B, T, C = x.shape  # batch, seq_len, hidden_size
        
        # Layer normalization
        input_layernorm_weight = self.params[f"layers.{layer_idx}.input_layernorm.weight"]
        ln_x = x * (1.0 / self.mx.sqrt(self.mx.mean(x * x, axis=-1, keepdims=True) + 1e-5)) * input_layernorm_weight
        
        # QKV projections
        q_weight = self.params[f"layers.{layer_idx}.self_attn.q_proj.weight"]
        k_weight = self.params[f"layers.{layer_idx}.self_attn.k_proj.weight"]
        v_weight = self.params[f"layers.{layer_idx}.self_attn.v_proj.weight"]
        
        q = self.mx.matmul(ln_x, q_weight.T)
        k = self.mx.matmul(ln_x, k_weight.T)
        v = self.mx.matmul(ln_x, v_weight.T)
        
        # Reshape for multi-head attention
        head_dim = C // self.model.num_attention_heads
        q = q.reshape(B, T, self.model.num_attention_heads, head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, self.model.num_key_value_heads, head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, self.model.num_key_value_heads, head_dim).transpose(0, 2, 1, 3)
        
        # Apply rotary embeddings
        q = self.apply_rotary_emb(q, freqs_cis)
        k = self.apply_rotary_emb(k, freqs_cis)
        
        # If using grouped-query attention, repeat k and v
        if self.model.num_key_value_heads < self.model.num_attention_heads:
            k = self.mx.repeat(
                k, self.model.num_attention_heads // self.model.num_key_value_heads, axis=1
            )
            v = self.mx.repeat(
                v, self.model.num_attention_heads // self.model.num_key_value_heads, axis=1
            )
        
        # Attention
        scores = self.mx.matmul(q, k.transpose(0, 1, 3, 2)) / self.mx.sqrt(head_dim)
        
        # Apply attention mask if provided
        if mask is not None:
            scores = scores + mask
        
        # Softmax and dropout
        attn = self.mx.softmax(scores, axis=-1)
        
        # Compute attention output
        out = self.mx.matmul(attn, v)
        out = out.transpose(0, 2, 1, 3).reshape(B, T, C)
        
        # Output projection
        o_weight = self.params[f"layers.{layer_idx}.self_attn.o_proj.weight"]
        out = self.mx.matmul(out, o_weight.T)
        
        return out
    
    def forward_mlp(self, x, layer_idx):
        """Forward pass through a single MLP layer"""
        # Layer normalization
        post_attention_layernorm_weight = self.params[f"layers.{layer_idx}.post_attention_layernorm.weight"]
        ln_x = x * (1.0 / self.mx.sqrt(self.mx.mean(x * x, axis=-1, keepdims=True) + 1e-5)) * post_attention_layernorm_weight
        
        # MLP projections
        gate_weight = self.params[f"layers.{layer_idx}.mlp.gate_proj.weight"]
        up_weight = self.params[f"layers.{layer_idx}.mlp.up_proj.weight"]
        down_weight = self.params[f"layers.{layer_idx}.mlp.down_proj.weight"]
        
        gate = self.mx.matmul(ln_x, gate_weight.T)
        up = self.mx.matmul(ln_x, up_weight.T)
        
        # SwiGLU activation
        act = self.mx.silu(gate) * up
        
        # Output projection
        out = self.mx.matmul(act, down_weight.T)
        
        return out
    
    def forward(self, input_ids, attention_mask=None):
        """Forward pass through the model"""
        B, T = input_ids.shape  # batch, seq_len
        
        # Get embeddings
        embed_weight = self.params["embed_tokens.weight"]
        h = self.mx.take(embed_weight, input_ids, axis=0)
        
        # Precompute rotary embeddings
        freqs_cis = self.precompute_freqs_cis(T)
        
        # Create causal mask
        if attention_mask is None:
            mask = self.mx.full((1, 1, T, T), -self.mx.inf)
            mask = self.mx.triu(mask, k=1)
        else:
            # Convert attention_mask to causal mask
            mask = (1 - attention_mask[:, None, None, :]) * -self.mx.inf
        
        # Forward through layers
        for i in range(self.model.num_hidden_layers):
            # Attention
            attn_output = self.forward_attention(h, freqs_cis, i, mask)
            h = h + attn_output
            
            # MLP
            mlp_output = self.forward_mlp(h, i)
            h = h + mlp_output
        
        # Final layer norm
        norm_weight = self.params["norm.weight"]
        h = h * (1.0 / self.mx.sqrt(self.mx.mean(h * h, axis=-1, keepdims=True) + 1e-5)) * norm_weight
        
        # LM head
        lm_head_weight = self.params["lm_head.weight"]
        logits = self.mx.matmul(h, lm_head_weight.T)
        
        return logits
    
    def generate(self, input_ids, max_length=100, temperature=0.8, top_p=0.95):
        """Generate text from the model"""
        import mlx.core as mx
        
        B, T = input_ids.shape
        
        for _ in range(max_length):
            # Get the last token's logits
            logits = self.forward(input_ids)[:, -1, :]
            
            # Apply temperature
            if temperature > 0:
                logits = logits / temperature
            
            # Apply top-p sampling
            if top_p < 1.0:
                sorted_logits, sorted_indices = mx.sort(logits, axis=-1, descending=True)
                cumulative_probs = mx.cumsum(mx.softmax(sorted_logits, axis=-1), axis=-1)
                
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift the indices to the right to keep also the first token above the threshold
                sorted_indices_to_remove = mx.concatenate(
                    [mx.zeros_like(sorted_indices_to_remove[:, :1]), sorted_indices_to_remove[:, :-1]], 
                    axis=1
                )
                
                # Scatter sorted indices to original order
                indices_to_remove = mx.zeros_like(logits, dtype=mx.bool)
                for b in range(B):
                    indices_to_remove[b, sorted_indices[b]] = sorted_indices_to_remove[b]
                
                logits = mx.where(indices_to_remove, -float('inf'), logits)
            
            # Sample from the distribution
            probs = mx.softmax(logits, axis=-1)
            next_token = mx.random.categorical(probs, num_samples=1)
            
            # Append to input_ids
            input_ids = mx.concatenate([input_ids, next_token], axis=1)
            
            # Check if we've generated an EOS token
            if (next_token == self.model.config.get('eos_token_id', 151644)).any():
                break
        
        return input_ids

def load_model_and_tokenizer(model_path, mlx_dir=None):
    """Load the model and tokenizer"""
    # Setup MLX
    if not setup_mlx_path(mlx_dir):
        raise RuntimeError("Failed to setup MLX")
    
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
                    def __init__(self):
                        self.bos_token_id = 151643  # Default for Qwen models
                        self.eos_token_id = 151644
                    
                    def encode(self, text, return_tensors=None):
                        # Simple encoding for testing
                        tokens = [self.bos_token_id] + [i % 100 + 1 for i in range(len(text))]
                        
                        import mlx.core as mx
                        if return_tensors == "mlx":
                            return mx.array([tokens])
                        return tokens
                    
                    def decode(self, ids):
                        # Simple decoding for testing
                        if hasattr(ids, "tolist"):
                            ids = ids.tolist()
                        return f"[Decoded text of length {len(ids)}]"
                
                tokenizer = DummyTokenizer()
                logger.info("Using dummy tokenizer for testing")
            
            # Create the MLX model
            mlx_model = QwenMLXModel(model)
            
            return mlx_model, tokenizer
        except Exception as e:
            logger.error(f"Error loading custom model: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            raise
    else:
        raise ValueError(f"Model format not supported: {model_path}")

def main():
    parser = argparse.ArgumentParser(description='Qwen MLX Inference')
    parser.add_argument('--model', type=str, required=True, help='Path to the MLX model file')
    parser.add_argument('--prompt', type=str, default="Hello, how are you today?", 
                        help='Prompt for inference')
    parser.add_argument('--max-length', type=int, default=100, 
                        help='Maximum number of tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.8,
                        help='Sampling temperature')
    parser.add_argument('--top-p', type=float, default=0.95,
                        help='Top-p sampling parameter')
    parser.add_argument('--mlx-dir', type=str, help='Path to MLX directory if not installed')
    parser.add_argument('--test-only', action='store_true', 
                        help='Only test loading the model without generation')
    args = parser.parse_args()
    
    try:
        # Load model and tokenizer
        model, tokenizer = load_model_and_tokenizer(args.model, args.mlx_dir)
        
        if args.test_only:
            logger.info("Test-only mode: Successfully loaded model and tokenizer")
            return 0
        
        # Encode the prompt
        import mlx.core as mx
        input_ids = tokenizer.encode(args.prompt, return_tensors="mlx")
        
        logger.info(f"Running inference with prompt: {args.prompt}")
        
        # Generate text
        start_time = time.time()
        
        try:
            output_ids = model.generate(
                input_ids, 
                max_length=args.max_length,
                temperature=args.temperature,
                top_p=args.top_p
            )
            
            # Decode the output
            output_text = tokenizer.decode(output_ids[0])
            
            end_time = time.time()
            logger.info(f"Generation completed in {end_time - start_time:.2f} seconds")
            logger.info(f"Output: {output_text}")
        except Exception as e:
            logger.error(f"Error during generation: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            
            # Try a simpler test - just run the forward pass
            logger.info("Trying simple forward pass instead")
            logits = model.forward(input_ids)
            logger.info(f"Forward pass successful, output shape: {logits.shape}")
            
            # Get the top 5 predicted tokens
            top_logits, top_indices = mx.topk(logits[0, -1], k=5)
            top_probs = mx.softmax(top_logits)
            
            logger.info("Top 5 predicted tokens:")
            for i, (idx, prob) in enumerate(zip(top_indices.tolist(), top_probs.tolist())):
                token_text = tokenizer.decode([idx])
                logger.info(f"  {i+1}. Token {idx}: {token_text} (prob: {prob:.4f})")
        
        return 0
    except Exception as e:
        logger.error(f"Error during inference: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return 1

if __name__ == '__main__':
    sys.exit(main())
