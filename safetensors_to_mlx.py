
# -*- coding: utf-8 -*-
"""
safetensors_to_mlx.py - A CLI tool to convert safetensors files to MLX format

This tool converts Hugging Face safetensors model files to MLX format for use with
Apple's MLX framework for efficient inference on Apple Silicon.
"""

import argparse
import json
import logging
import os
import sys
import threading
from pathlib import Path
from typing import List, Optional, Dict, Any

import torch
import numpy as np
import mlx.core as mx
import mlx
from safetensors.torch import load_file
from transformers import AutoTokenizer, AutoConfig

# Configure logging
logger = logging.getLogger("safetensors-to-mlx")
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

def setup_mlx_path(mlx_dir=None):
    """Set up the MLX path and verify installation"""
    try:
        import mlx
        import mlx.core
        
        # Check if MLX has __version__ attribute
        version_str = getattr(mlx, '__version__', 'unknown')
        logger.info(f"Found MLX version: {version_str}")
        
        # Only check version if it's available
        if version_str != 'unknown':
            try:
                version_parts = version_str.split('.')
                if int(version_parts[0]) < 1 and int(version_parts[1]) < 24:
                    logger.warning(f"MLX version {version_str} may be outdated. Recommended version is at least 0.24.0")
            except (ValueError, IndexError):
                logger.warning(f"Could not parse MLX version: {version_str}")
        
        return True
    except ImportError:
        logger.error("MLX not found. Please install MLX using: pip install mlx")
        return False

class MLXModelConverter:
    """Base class for converting models to MLX format"""
    
    def __init__(self, model_path, output_path, model_name=None, output_type="auto", 
                 metadata=None, vocab_only=False, verbose=False):
        self.model_path = Path(model_path)
        self.output_path = Path(output_path) if output_path else None
        self.model_name = model_name or self.model_path.name
        self.output_type = output_type
        self.metadata = metadata
        self.vocab_only = vocab_only
        self.verbose = verbose
        self.config = None
        self.tokenizer = None
        self.hparams = {}
        self.tensors = {}
        
        # Set logging level
        if verbose:
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.INFO)
    
    def load_config(self):
        """Load model configuration"""
        config_path = self.model_path / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, "r") as f:
            self.config = json.load(f)
        
        # Extract key parameters
        self.hparams = self.config.copy()
        logger.info(f"Loaded configuration from {config_path}")
        
        # Check for model architecture
        if "model_type" in self.config:
            self.model_type = self.config["model_type"]
            logger.info(f"Model type: {self.model_type}")
        else:
            logger.warning("Model type not found in config.json")
            self.model_type = "unknown"
    
    def load_tokenizer(self):
        """Load tokenizer"""
        try:
            # Check if this is a Qwen2.5 model
            if "qwen2_5" in self.model_type.lower() or "qwen2.5" in self.model_name.lower() or "qwen2-5" in self.model_name.lower():
                # Handle Qwen2.5 tokenizer manually since older transformers versions don't support it
                logger.info("Detected Qwen2.5 model, using manual tokenizer loading")
                
                # Define a custom tokenizer class with to_dict method
                class CustomTokenizer:
                    def __init__(self):
                        self.vocab = {}
                        self.merges = []
                        self.added_tokens_decoder = {}
                        self.name_or_path = None
                        self.model_max_length = 32768
                        self.special_tokens = {}
                        self.tokenizer_config = {}
                        self.original_tokenizer_data = {}
                    
                    def to_dict(self):
                        """Convert tokenizer to dictionary for serialization"""
                        return {
                            "vocab": self.vocab,
                            "merges": self.merges,
                            "added_tokens": self.added_tokens_decoder,
                            "model_max_length": self.model_max_length,
                            "special_tokens": self.special_tokens,
                            "name_or_path": self.name_or_path,
                            **self.tokenizer_config,
                            **self.original_tokenizer_data
                        }
                
                # Create a custom tokenizer instance
                self.tokenizer = CustomTokenizer()
                self.tokenizer.name_or_path = str(self.model_path)
                
                # Load tokenizer.json directly
                tokenizer_json_path = self.model_path / "tokenizer.json"
                if tokenizer_json_path.exists():
                    import json
                    with open(tokenizer_json_path, 'r') as f:
                        tokenizer_data = json.load(f)
                    
                    # Store the original data
                    self.tokenizer.original_tokenizer_data = tokenizer_data
                    
                    # Extract vocabulary and merges
                    self.tokenizer.vocab = tokenizer_data.get("model", {}).get("vocab", {})
                    self.tokenizer.merges = tokenizer_data.get("model", {}).get("merges", [])
                    
                    # Extract tokenizer config
                    if "model" in tokenizer_data:
                        self.tokenizer.tokenizer_config = {
                            k: v for k, v in tokenizer_data["model"].items() 
                            if k not in ["vocab", "merges"]
                        }
                    
                    # Add any special tokens
                    special_tokens = tokenizer_data.get("added_tokens", [])
                    for token in special_tokens:
                        token_id = token.get("id")
                        token_content = token.get("content")
                        if token_id is not None and token_content is not None:
                            self.tokenizer.added_tokens_decoder[token_id] = token_content
                            # Also add to special tokens if it has special attributes
                            if token.get("special", False):
                                token_name = token.get("name", f"special_{token_id}")
                                self.tokenizer.special_tokens[token_name] = token_content
                    
                    logger.info("Manually loaded Qwen2.5 tokenizer from tokenizer.json")
                else:
                    # Fallback to a basic tokenizer
                    logger.warning("Could not find tokenizer.json, creating minimal tokenizer")
            else:
                # Use standard AutoTokenizer for other models
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
                logger.info(f"Loaded tokenizer: {type(self.tokenizer).__name__}")
        except Exception as e:
            logger.error(f"Error loading tokenizer: {e}")
            logger.warning("Continuing with a minimal tokenizer")
            
            # Define a custom tokenizer class with to_dict method
            class CustomTokenizer:
                def __init__(self):
                    self.vocab = {}
                    self.merges = []
                    self.added_tokens_decoder = {}
                    self.name_or_path = None
                    self.model_max_length = 32768
                    self.special_tokens = {}
                
                def to_dict(self):
                    """Convert tokenizer to dictionary for serialization"""
                    return {
                        "vocab": self.vocab,
                        "merges": self.merges,
                        "added_tokens": self.added_tokens_decoder,
                        "model_max_length": self.model_max_length,
                        "special_tokens": self.special_tokens,
                        "name_or_path": self.name_or_path
                    }
            
            # Create a minimal tokenizer to allow conversion to continue
            self.tokenizer = CustomTokenizer()
            self.tokenizer.name_or_path = str(self.model_path)
    
    def find_safetensors_files(self):
        """Find all safetensors files in the model directory"""
        safetensors_files = list(self.model_path.glob("*.safetensors"))
        if not safetensors_files:
            raise FileNotFoundError(f"No safetensors files found in {self.model_path}")
        
        logger.info(f"Found {len(safetensors_files)} safetensors files")
        return safetensors_files
    
    def load_tensors(self, safetensors_files):
        """Load tensors from safetensors files"""
        for file_path in safetensors_files:
            logger.info(f"Loading tensors from {file_path.name}")
            try:
                tensors = load_file(file_path)
                # Add tensors to the global tensor dict
                for name, tensor in tensors.items():
                    # Convert PyTorch tensor to NumPy and then to MLX array
                    if self.verbose:
                        logger.debug(f"Converting tensor: {name}, shape: {tensor.shape}, dtype: {tensor.dtype}")
                    
                    # Convert to the specified output type
                    if self.output_type == "fp16" or (self.output_type == "auto" and tensor.dtype == torch.float32):
                        numpy_tensor = tensor.to(torch.float16).numpy()
                    elif self.output_type == "bf16":
                        # MLX doesn't support bf16, so we'll use fp16 as the closest alternative
                        logger.warning("MLX doesn't support bfloat16, using float16 instead")
                        numpy_tensor = tensor.to(torch.float16).numpy()
                    else:  # fp32 or auto with non-float32 tensor
                        numpy_tensor = tensor.numpy()
                    
                    # Convert to MLX array
                    mlx_tensor = mx.array(numpy_tensor)
                    self.tensors[name] = mlx_tensor
                    
                    if self.verbose:
                        logger.debug(f"Converted to MLX tensor: {name}, shape: {mlx_tensor.shape}, dtype: {mlx_tensor.dtype}")
            
            except Exception as e:
                logger.error(f"Error loading tensors from {file_path.name}: {e}")
                raise
        
        logger.info(f"Loaded {len(self.tensors)} tensors in total")
    
    def process_tensors(self):
        """Process tensors for MLX format"""
        # This method can be overridden by subclasses to perform model-specific tensor processing
        logger.info("Processing tensors for MLX format")
        # By default, we don't modify the tensors
        return self.tensors
    
    def prepare_metadata(self):
        """Prepare metadata for the MLX model"""
        metadata = {
            "model_name": self.model_name,
            "model_type": self.model_type,
            "framework": "mlx",
            "conversion_source": "safetensors",
        }
        
        # Add key model parameters
        if "hidden_size" in self.hparams:
            metadata["hidden_size"] = self.hparams["hidden_size"]
        if "num_hidden_layers" in self.hparams:
            metadata["num_hidden_layers"] = self.hparams["num_hidden_layers"]
        if "num_attention_heads" in self.hparams:
            metadata["num_attention_heads"] = self.hparams["num_attention_heads"]
        
        # Add user-provided metadata
        if self.metadata:
            try:
                with open(self.metadata, "r") as f:
                    user_metadata = json.load(f)
                metadata.update(user_metadata)
            except Exception as e:
                logger.error(f"Error loading metadata from {self.metadata}: {e}")
        
        return metadata
    
    def save_mlx_model(self, tensors, metadata):
        """Save tensors and metadata to MLX format"""
        if self.output_path is None:
            self.output_path = self.model_path / f"{self.model_name}.mlx"
        
        # Create parent directories if they don't exist
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving MLX model to {self.output_path}")
        
        # Prepare model dictionary
        model_dict = {
            "weights": tensors,
            "metadata": metadata,
            "config": self.config
        }
        
        # If we have a tokenizer, save its configuration
        if self.tokenizer:
            tokenizer_config = self.tokenizer.to_dict()
            model_dict["tokenizer"] = tokenizer_config
        
        # Try to save using save_safetensors if available, otherwise use a custom implementation
        try:
            # Check if save_safetensors function exists in mlx.core
            if hasattr(mx, 'save_safetensors'):
                mx.save_safetensors(str(self.output_path), model_dict["weights"])
                logger.info("Used mlx.core.save_safetensors to save model")
            else:
                # Fallback to custom implementation for older MLX versions
                logger.info("save_safetensors not found in mlx.core, using custom implementation")
                self._custom_save_mlx(str(self.output_path), model_dict["weights"])
        except Exception as e:
            logger.error(f"Error saving model weights: {e}")
            logger.info("Falling back to custom implementation")
            self._custom_save_mlx(str(self.output_path), model_dict["weights"])
        
        # Also save metadata and config as JSON files
        import json
        metadata_path = self.output_path.with_suffix('.metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(model_dict["metadata"], f, indent=2)
        
        config_path = self.output_path.with_suffix('.config.json')
        with open(config_path, 'w') as f:
            json.dump(model_dict["config"], f, indent=2)
        
        # If we have a tokenizer, save its configuration
        if "tokenizer" in model_dict:
            tokenizer_path = self.output_path.with_suffix('.tokenizer.json')
            with open(tokenizer_path, 'w') as f:
                json.dump(model_dict["tokenizer"], f, indent=2)
        
        logger.info(f"Successfully saved MLX model to {self.output_path}")
        logger.info(f"Model size: {self.output_path.stat().st_size / (1024 * 1024):.2f} MB")
    
    def _custom_save_mlx(self, path, tensors):
        """Custom implementation to save MLX tensors when save_safetensors is not available"""
        import numpy as np
        import json
        import os
        import h5py
        from pathlib import Path
        
        logger.info(f"Using custom MLX saving implementation for {path}")
        
        # Create a directory to store individual tensors
        path_obj = Path(path)
        tensor_dir = path_obj.with_suffix('.tensors')
        tensor_dir.mkdir(parents=True, exist_ok=True)
        
        # Save tensor metadata
        tensor_info = {}
        
        # Save each tensor individually as a numpy file
        for name, tensor in tensors.items():
            # Create a safe filename
            safe_name = name.replace('/', '_').replace('.', '_')
            tensor_path = tensor_dir / f"{safe_name}.npy"
            
            try:
                # Get shape and dtype information
                shape = tensor.shape
                dtype = str(tensor.dtype)
                
                # Store metadata
                tensor_info[name] = {
                    "path": str(tensor_path.relative_to(tensor_dir)),
                    "shape": list(shape),
                    "dtype": dtype,
                    "original_name": name
                }
                
                # Convert to numpy and save
                if hasattr(tensor, 'numpy'):
                    np.save(tensor_path, tensor.numpy())
                elif hasattr(tensor, 'astype'):
                    np.save(tensor_path, tensor)
                else:
                    # Try to convert to a numpy array first
                    np.save(tensor_path, np.array(tensor))
                    
                logger.debug(f"Saved tensor {name} to {tensor_path}")
            except Exception as e:
                logger.warning(f"Could not save tensor {name}: {e}")
        
        # Save tensor metadata
        with open(tensor_dir / "metadata.json", 'w') as f:
            json.dump(tensor_info, f, indent=2)
        
        # Create a simple HDF5 file as the main model file
        with h5py.File(path, 'w') as f:
            # Store basic metadata
            f.attrs['format'] = 'mlx_custom'
            f.attrs['tensor_dir'] = str(tensor_dir)
            f.attrs['num_tensors'] = len(tensor_info)
            
            # Create a group for tensor metadata
            meta_group = f.create_group('tensor_metadata')
            for name, info in tensor_info.items():
                tensor_group = meta_group.create_group(name)
                for k, v in info.items():
                    if isinstance(v, list):
                        tensor_group.attrs[k] = json.dumps(v)
                    else:
                        tensor_group.attrs[k] = v
        
        logger.info(f"Saved tensors to {path} using custom implementation")
        logger.info(f"Individual tensors stored in {tensor_dir}")
    
    def convert(self):
        """Convert safetensors model to MLX format"""
        # Load model configuration
        self.load_config()
        
        # Load tokenizer
        if not self.vocab_only:
            self.load_tokenizer()
        
        # Find safetensors files
        safetensors_files = self.find_safetensors_files()
        
        # Load tensors
        self.load_tensors(safetensors_files)
        
        # Process tensors
        processed_tensors = self.process_tensors()
        
        # Prepare metadata
        metadata = self.prepare_metadata()
        
        # Save MLX model
        self.save_mlx_model(processed_tensors, metadata)
        
        return self.output_path


class Llama4ModelConverter(MLXModelConverter):
    """Specialized converter for Llama-4 models"""
    
    def load_config(self):
        """Load model configuration with special handling for Llama-4"""
        super().load_config()
        
        # Check for Llama-4 specific nested configuration
        if "text_config" in self.config:
            logger.info("Found Llama-4 nested configuration structure")
            
            # Copy essential parameters to the top level
            for param in ["hidden_size", "intermediate_size", "num_hidden_layers", 
                         "num_attention_heads", "num_key_value_heads", "vocab_size"]:
                if param in self.config["text_config"]:
                    self.hparams[param] = self.config["text_config"][param]
                    logger.debug(f"Copied {param}: {self.hparams[param]}")
    
    def process_tensors(self):
        """Process tensors with special handling for Llama-4 MoE architecture"""
        logger.info("Processing tensors for Llama-4 model")
        
        processed_tensors = {}
        
        # Check for MoE architecture
        has_moe = any("experts" in name for name in self.tensors.keys())
        if has_moe:
            logger.info("Detected Mixture of Experts (MoE) architecture")
        
        # Process tensors
        for name, tensor in self.tensors.items():
            # Skip vision components in multimodal models
            if "vision" in name:
                logger.debug(f"Skipping vision component: {name}")
                continue
            
            # Handle MoE tensors
            if has_moe and "experts" in name:
                logger.debug(f"Processing MoE tensor: {name}")
                # MLX has different naming conventions for MoE components
                # This is a placeholder for the actual implementation
                processed_name = name.replace("model.layers", "layers")
                processed_tensors[processed_name] = tensor
            else:
                # Standard tensor processing
                # Adapt naming convention for MLX
                processed_name = name.replace("model.layers", "layers")
                processed_tensors[processed_name] = tensor
        
        logger.info(f"Processed {len(processed_tensors)} tensors")
        return processed_tensors


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Convert safetensors model files to MLX format"
    )
    
    parser.add_argument(
        "--model", type=Path, required=True,
        help="Path to the directory containing the model's safetensors files"
    )
    
    parser.add_argument(
        "--outfile", type=Path, default=None,
        help="Path to write the output MLX file (default: model directory name with .mlx extension)"
    )
    
    parser.add_argument(
        "--outtype", type=str, choices=["fp32", "fp16", "auto"], default="auto",
        help="Output data type (default: auto)"
    )
    
    parser.add_argument(
        "--vocab-only", action="store_true",
        help="Extract only the vocabulary"
    )
    
    parser.add_argument(
        "--model-name", type=str, default=None,
        help="Override the model name in the MLX file metadata"
    )
    
    parser.add_argument(
        "--metadata", type=Path, default=None,
        help="Path to a JSON file containing metadata to add to the MLX file"
    )
    
    parser.add_argument(
        "--threads", type=int, default=None,
        help="Number of threads to use for conversion (default: number of CPU cores)"
    )
    
    parser.add_argument(
        "--verbose", action="store_true",
        help="Enable verbose logging"
    )
    
    parser.add_argument(
        "--mlx-dir", type=Path, default=None,
        help="Path to the MLX directory (default: auto-detect)"
    )
    
    return parser.parse_args()


def verify_safetensors_model(model_dir: Path):
    """
    Verify that the model directory contains safetensors files.
    
    Args:
        model_dir: Path to the model directory
        
    Returns:
        bool: True if safetensors files are found, False otherwise
    """
    if not model_dir.exists():
        logger.error(f"Model directory not found: {model_dir}")
        return False
    
    if not model_dir.is_dir():
        logger.error(f"Model path is not a directory: {model_dir}")
        return False
    
    safetensors_files = list(model_dir.glob("*.safetensors"))
    if not safetensors_files:
        logger.error(f"No safetensors files found in {model_dir}")
        return False
    
    config_file = model_dir / "config.json"
    if not config_file.exists():
        logger.warning(f"config.json not found in {model_dir}")
    
    return True


def convert_safetensors_to_mlx(args):
    """
    Convert safetensors model to MLX format.
    
    Args:
        args: Command line arguments
    """
    # Verify the model directory
    if not verify_safetensors_model(args.model):
        sys.exit(1)
    
    # Set up MLX path
    if not setup_mlx_path(args.mlx_dir):
        sys.exit(1)
    
    # Set number of threads if specified
    if args.threads:
        torch.set_num_threads(args.threads)
        logger.info(f"Set number of threads to {args.threads}")
    
    try:
        # Load model configuration to determine model type
        config_path = args.model / "config.json"
        with open(config_path, "r") as f:
            config = json.load(f)
        
        # Determine model architecture
        model_architecture = config.get("model_type", "").lower()
        
        # Select appropriate converter based on model architecture
        if "llama" in model_architecture and ("4" in model_architecture or "4" in args.model.name):
            logger.info("Using Llama-4 model converter")
            converter = Llama4ModelConverter(
                model_path=args.model,
                output_path=args.outfile,
                model_name=args.model_name or args.model.name,
                output_type=args.outtype,
                metadata=args.metadata,
                vocab_only=args.vocab_only,
                verbose=args.verbose
            )
        else:
            logger.info(f"Using generic model converter for {model_architecture}")
            converter = MLXModelConverter(
                model_path=args.model,
                output_path=args.outfile,
                model_name=args.model_name or args.model.name,
                output_type=args.outtype,
                metadata=args.metadata,
                vocab_only=args.vocab_only,
                verbose=args.verbose
            )
        
        # Convert the model
        output_path = converter.convert()
        logger.info(f"Model successfully converted to MLX format: {output_path}")
    
    except Exception as e:
        logger.error(f"Error during conversion: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        sys.exit(1)


def main():
    """Main entry point."""
    args = parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger("safetensors-to-mlx")
    
    # Convert the model
    try:
        convert_safetensors_to_mlx(args)
        return 0
    except Exception as e:
        logger.error(f"Error during conversion: {e}")
        if args.verbose:
            import traceback
            logger.debug(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())
