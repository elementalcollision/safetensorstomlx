#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
optimize_mlx.py - A CLI tool to optimize MLX models for efficient inference on Apple Silicon

This tool optimizes MLX models for better performance and smaller size on Apple Silicon devices.
"""

import argparse
import logging
import os
import sys
import json
import re
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
from collections import defaultdict

import numpy as np
import mlx.core as mx
import mlx

# Configure logging
logger = logging.getLogger("optimize-mlx")
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


def setup_mlx_path(mlx_dir=None):
    """Set up the MLX path and verify installation"""
    try:
        import mlx
        import mlx.core
        logger.info(f"Found MLX version: {mlx.__version__}")
        # Check if version is at least 0.24.0
        version_parts = mlx.__version__.split('.')
        if int(version_parts[0]) < 1 and int(version_parts[1]) < 24:
            logger.warning(f"MLX version {mlx.__version__} may be outdated. Recommended version is at least 0.24.0")
        return True
    except ImportError:
        logger.error("MLX not found. Please install MLX using: pip install mlx")
        return False


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Optimize MLX models for efficient inference on Apple Silicon"
    )
    
    parser.add_argument(
        "--model", type=Path, required=True,
        help="Path to the input MLX model file"
    )
    
    parser.add_argument(
        "--outfile", type=Path, default=None,
        help="Path to write the output optimized MLX file (default: same directory as input with optimization type suffix)"
    )
    
    parser.add_argument(
        "--type", type=str, choices=["fp32", "fp16", "auto"], default="fp16",
        help="Optimization type (default: fp16)"
    )
    
    parser.add_argument(
        "--threads", type=int, default=None,
        help="Number of threads to use for optimization (default: number of CPU cores)"
    )
    
    parser.add_argument(
        "--allow-reoptimize", action="store_true",
        help="Allow reoptimizing tensors that have already been optimized"
    )
    
    parser.add_argument(
        "--leave-output-tensor", action="store_true",
        help="Leave output.weight unoptimized (may improve quality)"
    )
    
    parser.add_argument(
        "--output-tensor-type", type=str, choices=["fp32", "fp16"], default=None,
        help="Use this type for the output.weight tensor"
    )
    
    parser.add_argument(
        "--token-embedding-type", type=str, choices=["fp32", "fp16"], default=None,
        help="Use this type for the token embeddings tensor"
    )
    
    parser.add_argument(
        "--analyze-model", action="store_true",
        help="Analyze model structure before optimization to identify tensor distribution and MoE components"
    )
    
    parser.add_argument(
        "--moe-expert-optimization", type=str, choices=["fp32", "fp16", "same"], default="same",
        help="Optimization type for MoE expert layers"
    )
    
    parser.add_argument(
        "--moe-router-optimization", type=str, choices=["fp32", "fp16", "same"], default="same",
        help="Optimization type for MoE router layers"
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


def analyze_model_structure(input_file: Path, verbose: bool = False):
    """
    Analyze the structure of a MLX model to understand tensor distribution and identify MoE components.
    
    Args:
        input_file: Path to the input MLX file
        verbose: Whether to print detailed analysis information
        
    Returns:
        Dictionary containing analysis results
    """
    logger.info(f"Analyzing model structure: {input_file}")
    
    try:
        # Load the model - handle different file formats
        try:
            # First try loading as safetensors
            weights = mx.load_safetensors(str(input_file))
            
            # Try to load metadata and config from accompanying JSON files
            import json
            metadata = {}
            config = {}
            
            metadata_path = input_file.with_suffix('.metadata.json')
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
            
            config_path = input_file.with_suffix('.config.json')
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    
            tokenizer = {}
            tokenizer_path = input_file.with_suffix('.tokenizer.json')
            if tokenizer_path.exists():
                with open(tokenizer_path, 'r') as f:
                    tokenizer = json.load(f)
                    
            model_dict = {
                "weights": weights,
                "metadata": metadata,
                "config": config
            }
            
            if tokenizer:
                model_dict["tokenizer"] = tokenizer
                
            logger.info(f"Loaded model from safetensors format: {input_file}")
        except Exception as e:
            logger.warning(f"Could not load as safetensors, trying generic load: {e}")
            try:
                # Try loading as a generic MLX save file
                model_dict = mx.load(str(input_file))
                weights = model_dict.get("weights", {})
                metadata = model_dict.get("metadata", {})
                config = model_dict.get("config", {})
                logger.info(f"Loaded model from generic format: {input_file}")
            except Exception as e2:
                logger.error(f"Failed to load model: {e2}")
                raise
        
        # Initialize analysis results
        results = {
            "total_tensors": len(weights),
            "tensor_types": defaultdict(int),
            "tensor_shapes": {},
            "has_moe": False,
            "moe_tensors": [],
            "router_tensors": [],
            "expert_tensors": [],
            "output_tensor": None,
            "token_embedding_tensor": None,
        }
        
        # Analyze tensor types and shapes
        total_size_bytes = 0
        for name, tensor in weights.items():
            # Count tensor types
            dtype_str = str(tensor.dtype)
            results["tensor_types"][dtype_str] += 1
            
            # Record tensor shapes
            shape_str = str(tensor.shape)
            if shape_str not in results["tensor_shapes"]:
                results["tensor_shapes"][shape_str] = []
            results["tensor_shapes"][shape_str].append(name)
            
            # Calculate tensor size
            tensor_size = tensor.nbytes
            total_size_bytes += tensor_size
            
            # Check for MoE components
            if "expert" in name or "experts" in name:
                results["has_moe"] = True
                results["expert_tensors"].append(name)
            elif "router" in name or "gate" in name:
                results["has_moe"] = True
                results["router_tensors"].append(name)
            
            # Check for output tensor
            if name.endswith("output.weight") or name.endswith("lm_head.weight"):
                results["output_tensor"] = name
            
            # Check for token embedding tensor
            if name.endswith("token_embeddings.weight") or name.endswith("embed_tokens.weight"):
                results["token_embedding_tensor"] = name
            
            if verbose:
                logger.debug(f"Tensor: {name}, Shape: {tensor.shape}, Type: {tensor.dtype}, Size: {tensor_size/1024/1024:.2f} MB")
        
        # Combine expert and router tensors into moe_tensors
        results["moe_tensors"] = results["expert_tensors"] + results["router_tensors"]
        
        # Calculate total size
        results["total_size_mb"] = total_size_bytes / (1024 * 1024)
        
        # Log summary
        logger.info(f"Model size: {results['total_size_mb']:.2f} MB")
        logger.info(f"Total tensors: {results['total_tensors']}")
        logger.info(f"Tensor types: {dict(results['tensor_types'])}")
        logger.info(f"Has MoE architecture: {results['has_moe']}")
        if results["has_moe"]:
            logger.info(f"MoE expert tensors: {len(results['expert_tensors'])}")
            logger.info(f"MoE router tensors: {len(results['router_tensors'])}")
        
        # Save analysis to file
        analysis_file = input_file.parent / "model_analysis.json"
        with open(analysis_file, "w") as f:
            # Convert defaultdict to dict for JSON serialization
            serializable_results = {k: v if not isinstance(v, defaultdict) else dict(v) for k, v in results.items()}
            # Convert tensor shapes dict values from lists to counts for readability
            serializable_results["tensor_shapes"] = {k: len(v) for k, v in serializable_results["tensor_shapes"].items()}
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Saved model analysis to {analysis_file}")
        
        return results
    
    except Exception as e:
        logger.error(f"Error analyzing model: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return {"error": str(e)}


def optimize_mlx_model(args):
    """
    Optimize a MLX model for efficient inference on Apple Silicon.
    
    Args:
        args: Command line arguments
    """
    # Verify the model file exists
    if not args.model.exists():
        logger.error(f"Model file not found: {args.model}")
        return 1
    
    # Set up MLX path
    if not setup_mlx_path(args.mlx_dir):
        return 1
    
    # Analyze model structure if requested
    if args.analyze_model:
        analysis_results = analyze_model_structure(args.model, verbose=args.verbose)
        if "error" in analysis_results:
            return 1
    else:
        # Quick analysis to detect MoE architecture
        try:
            model_dict = mx.load(str(args.model))
            weights = model_dict.get("weights", {})
            has_moe = any("expert" in name or "experts" in name or "router" in name or "gate" in name 
                          for name in weights.keys())
            if has_moe:
                logger.info("Detected Mixture of Experts (MoE) architecture")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return 1
    
    # Determine output file path
    if args.outfile:
        outfile = args.outfile
    else:
        # Generate a default output filename in the same directory as the model
        stem = args.model.stem
        suffix = f"-{args.type}"
        outfile = args.model.parent / f"{stem}{suffix}.mlx"
    
    logger.info(f"Output file: {outfile}")
    
    try:
        # Load the model
        logger.info(f"Loading model: {args.model}")
        model_dict = mx.load(str(args.model))
        weights = model_dict.get("weights", {})
        metadata = model_dict.get("metadata", {})
        config = model_dict.get("config", {})
        tokenizer = model_dict.get("tokenizer", {})
        
        # Update metadata
        metadata["optimization_type"] = args.type
        metadata["optimization_date"] = str(import_datetime().now().isoformat())
        
        # Optimize weights
        logger.info(f"Optimizing model weights to {args.type}")
        optimized_weights = {}
        
        for name, tensor in weights.items():
            # Determine the target dtype for this tensor
            target_dtype = None
            
            # Check if this is an output tensor that should be left unoptimized
            if args.leave_output_tensor and (name.endswith("output.weight") or name.endswith("lm_head.weight")):
                if args.output_tensor_type:
                    target_dtype = args.output_tensor_type
                else:
                    # Keep original dtype
                    target_dtype = str(tensor.dtype)
                logger.debug(f"Leaving output tensor {name} as {target_dtype}")
            
            # Check if this is a token embedding tensor with a specific type
            elif args.token_embedding_type and (name.endswith("token_embeddings.weight") or name.endswith("embed_tokens.weight")):
                target_dtype = args.token_embedding_type
                logger.debug(f"Setting token embedding tensor {name} to {target_dtype}")
            
            # Check if this is a MoE expert tensor
            elif has_moe and ("expert" in name or "experts" in name) and args.moe_expert_optimization != "same":
                target_dtype = args.moe_expert_optimization
                logger.debug(f"Setting MoE expert tensor {name} to {target_dtype}")
            
            # Check if this is a MoE router tensor
            elif has_moe and ("router" in name or "gate" in name) and args.moe_router_optimization != "same":
                target_dtype = args.moe_router_optimization
                logger.debug(f"Setting MoE router tensor {name} to {target_dtype}")
            
            # Default optimization type
            else:
                target_dtype = args.type
            
            # Skip optimization if tensor is already in the target format and reoptimization is not allowed
            if not args.allow_reoptimize and str(tensor.dtype) == target_dtype:
                logger.debug(f"Tensor {name} already in {target_dtype} format, skipping")
                optimized_weights[name] = tensor
                continue
            
            # Convert tensor to the target dtype
            if target_dtype == "fp16":
                optimized_weights[name] = tensor.astype(mx.float16)
            elif target_dtype == "bf16":
                # MLX doesn't support bf16, so we'll use fp16 as the closest alternative
                logger.warning("MLX doesn't support bfloat16, using float16 instead")
                optimized_weights[name] = tensor.astype(mx.float16)
            else:  # fp32
                optimized_weights[name] = tensor.astype(mx.float32)
        
        # Create optimized model dictionary
        optimized_model = {
            "weights": optimized_weights,
            "metadata": metadata,
            "config": config
        }
        
        # Add tokenizer if present
        if tokenizer:
            optimized_model["tokenizer"] = tokenizer
        
        # Save optimized model
        logger.info(f"Saving optimized model to {outfile}")
        
        # Save weights in safetensors format
        mx.save_safetensors(str(outfile), optimized_model["weights"])
        
        # Save metadata and config as separate JSON files
        import json
        metadata_path = outfile.with_suffix('.metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(optimized_model["metadata"], f, indent=2)
        
        config_path = outfile.with_suffix('.config.json')
        with open(config_path, 'w') as f:
            json.dump(optimized_model["config"], f, indent=2)
        
        # Save tokenizer if present
        if "tokenizer" in optimized_model:
            tokenizer_path = outfile.with_suffix('.tokenizer.json')
            with open(tokenizer_path, 'w') as f:
                json.dump(optimized_model["tokenizer"], f, indent=2)
        
        # Calculate size reduction
        original_size = args.model.stat().st_size / (1024 * 1024)
        optimized_size = outfile.stat().st_size / (1024 * 1024)
        size_reduction = (original_size - optimized_size) / original_size * 100
        
        logger.info(f"Original size: {original_size:.2f} MB")
        logger.info(f"Optimized size: {optimized_size:.2f} MB")
        logger.info(f"Size reduction: {size_reduction:.2f}%")
        
        return 0
    
    except Exception as e:
        logger.error(f"Error during optimization: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return 1


def import_datetime():
    """Import datetime module"""
    import datetime
    return datetime


def main():
    """Main entry point."""
    args = parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger("optimize-mlx")
    
    # Check if we're just analyzing the model structure
    if args.analyze_model and args.type == "auto":
        # In this case, we're just analyzing the model without optimizing
        logger.info("Running in analysis-only mode")
        try:
            # Verify the model file exists
            if not args.model.exists():
                logger.error(f"Model file not found: {args.model}")
                return 1
                
            # Analyze the model structure
            analysis_results = analyze_model_structure(args.model, verbose=True)
            
            if "error" in analysis_results:
                logger.error(f"Error analyzing model: {analysis_results['error']}")
                return 1
                
            # Provide optimization recommendations based on analysis
            has_moe = analysis_results.get("has_moe", False)
            if has_moe:
                logger.info("\n===== Optimization Recommendations for MoE Model =====")
                logger.info("This model contains Mixture of Experts (MoE) architecture.")
                logger.info("Recommended optimization settings:")
                logger.info("  1. For better quality: --type fp16 --leave-output-tensor --moe-expert-optimization fp32")
                logger.info("  2. For better size: --type fp16 --moe-expert-optimization fp16")
                logger.info("  3. For balanced approach: --type fp16 --moe-router-optimization fp32")
            else:
                logger.info("\n===== Optimization Recommendations =====")
                logger.info("  1. For better quality: --type fp16 --leave-output-tensor")
                logger.info("  2. For better size: --type fp16")
                logger.info("  3. For balanced approach: --type fp16 --token-embedding-type fp32")
                
            return 0
        except Exception as e:
            logger.error(f"Error during model analysis: {e}")
            if args.verbose:
                import traceback
                logger.debug(traceback.format_exc())
            return 1
    
    # Optimize the model
    try:
        return optimize_mlx_model(args)
    except Exception as e:
        logger.error(f"Error during optimization: {e}")
        if args.verbose:
            import traceback
            logger.debug(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())
