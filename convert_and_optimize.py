#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
convert_and_optimize.py - A two-step tool for MoE models that first converts SafeTensors to MLX,
then applies optimization to the MLX file.

This approach addresses the challenge with Llama-4 Scout and other MoE models that may already be in a
compressed format that can't be further optimized using standard MLX optimization tools.
"""

import argparse
import logging
import os
import subprocess
import sys
import tempfile
import shutil
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple

# Configure logging
logger = logging.getLogger("convert-and-optimize")
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
        description="Two-step tool for MoE models: first converts SafeTensors to MLX, then optimizes"
    )
    
    parser.add_argument(
        "--safetensors-dir", type=Path, required=True,
        help="Directory containing SafeTensors model files"
    )
    
    parser.add_argument(
        "--outfile", type=Path, default=None,
        help="Path to write the final optimized MLX file"
    )
    
    parser.add_argument(
        "--outdir", type=Path, default=None,
        help="Output directory for the final optimized MLX model (if --outfile not specified)"
    )
    
    parser.add_argument(
        "--type", type=str, choices=["fp32", "fp16", "auto"], default="fp16",
        help="Optimization type for the final model (default: fp16)"
    )
    
    parser.add_argument(
        "--intermediate-type", type=str, choices=["fp32", "fp16"], default="fp16",
        help="Format for the intermediate MLX file (default: fp16)"
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
        "--mlx-dir", type=Path, default=None,
        help="Path to MLX directory (if not automatically detected)"
    )
    
    parser.add_argument(
        "--keep-intermediate", action="store_true",
        help="Keep the intermediate uncompressed MLX file"
    )
    
    parser.add_argument(
        "--verbose", action="store_true",
        help="Enable verbose output"
    )
    
    parser.add_argument(
        "--threads", type=int, default=None,
        help="Number of threads to use for optimization"
    )
    
    parser.add_argument(
        "--allow-reoptimize", action="store_true",
        help="Allow reoptimizing tensors that are already optimized"
    )
    
    parser.add_argument(
        "--leave-output-tensor", action="store_true",
        help="Leave the output tensor in the original format (fp16/fp32)"
    )
    
    parser.add_argument(
        "--output-tensor-type", type=str, choices=["fp32", "fp16"], default=None,
        help="Output tensor type (fp32, fp16)"
    )
    
    parser.add_argument(
        "--token-embedding-type", type=str, choices=["fp32", "fp16"], default=None,
        help="Token embedding tensor type (fp32, fp16)"
    )
    
    return parser.parse_args()

def convert_safetensors_to_mlx(args):
    """
    Convert SafeTensors model to MLX format.
    
    Args:
        args: Command line arguments
        
    Returns:
        Path to the generated MLX file
    """
    # Verify the safetensors directory exists
    if not args.safetensors_dir.exists() or not args.safetensors_dir.is_dir():
        logger.error(f"SafeTensors directory not found: {args.safetensors_dir}")
        return None
    
    # Create a temporary directory for the intermediate MLX file if needed
    temp_dir = None
    if args.outfile is None and args.outdir is None:
        temp_dir = tempfile.mkdtemp(prefix="mlx_convert_")
        logger.info(f"Created temporary directory for intermediate files: {temp_dir}")
        intermediate_file = Path(temp_dir) / f"{args.safetensors_dir.name}.mlx"
    else:
        # Determine the intermediate file path
        if args.outfile:
            # Use the same directory as the final output file
            intermediate_file = args.outfile.parent / f"{args.safetensors_dir.name}_intermediate.mlx"
        else:
            # Use the specified output directory
            intermediate_file = args.outdir / f"{args.safetensors_dir.name}_intermediate.mlx"
    
    logger.info(f"Intermediate MLX file: {intermediate_file}")
    
    # Import the safetensors_to_mlx module
    try:
        import importlib.util
        script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
        convert_script_path = script_dir / "safetensors_to_mlx.py"
        
        if not convert_script_path.exists():
            logger.error(f"Could not find safetensors_to_mlx.py script at {convert_script_path}")
            return None
        
        spec = importlib.util.spec_from_file_location("safetensors_to_mlx", str(convert_script_path))
        convert_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(convert_module)
        
        # Call the conversion function
        logger.info(f"Converting SafeTensors model to MLX format: {args.safetensors_dir}")
        
        # Create a custom args object for the conversion function
        from argparse import Namespace
        convert_args = Namespace(
            model=args.safetensors_dir,
            outfile=intermediate_file,
            outtype=args.intermediate_type,
            vocab_only=False,
            model_name=args.safetensors_dir.name,
            metadata=None,
            threads=args.threads,
            verbose=args.verbose,
            mlx_dir=args.mlx_dir
        )
        
        # Call the conversion function
        convert_module.convert_safetensors_to_mlx(convert_args)
        
        # Verify the intermediate file was created
        if not intermediate_file.exists():
            logger.error(f"Failed to create intermediate MLX file: {intermediate_file}")
            return None
        
        logger.info(f"Successfully converted SafeTensors model to MLX format: {intermediate_file}")
        return intermediate_file, temp_dir
    
    except Exception as e:
        logger.error(f"Error during conversion: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return None

def optimize_mlx_model(args, intermediate_file):
    """
    Optimize the MLX model.
    
    Args:
        args: Command line arguments
        intermediate_file: Path to the intermediate MLX file
        
    Returns:
        Return code from the optimization process
    """
    # Determine the output file path
    if args.outfile:
        outfile = args.outfile
    elif args.outdir:
        outfile = args.outdir / f"{args.safetensors_dir.name}_{args.type}.mlx"
    else:
        outfile = intermediate_file.parent / f"{args.safetensors_dir.name}_{args.type}.mlx"
    
    logger.info(f"Output optimized MLX file: {outfile}")
    
    # Import the optimize_mlx module
    try:
        import importlib.util
        script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
        optimize_script_path = script_dir / "optimize_mlx.py"
        
        if not optimize_script_path.exists():
            logger.error(f"Could not find optimize_mlx.py script at {optimize_script_path}")
            return 1
        
        spec = importlib.util.spec_from_file_location("optimize_mlx", str(optimize_script_path))
        optimize_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(optimize_module)
        
        # Call the optimization function
        logger.info(f"Optimizing MLX model: {intermediate_file}")
        
        # Create a custom args object for the optimization function
        from argparse import Namespace
        optimize_args = Namespace(
            model=intermediate_file,
            outfile=outfile,
            type=args.type,
            threads=args.threads,
            allow_reoptimize=args.allow_reoptimize,
            leave_output_tensor=args.leave_output_tensor,
            output_tensor_type=args.output_tensor_type,
            token_embedding_type=args.token_embedding_type,
            analyze_model=False,
            moe_expert_optimization=args.moe_expert_optimization,
            moe_router_optimization=args.moe_router_optimization,
            verbose=args.verbose,
            mlx_dir=args.mlx_dir
        )
        
        # Call the optimization function
        return_code = optimize_module.optimize_mlx_model(optimize_args)
        
        # Clean up the intermediate file if not keeping it
        if not args.keep_intermediate and intermediate_file.exists():
            logger.info(f"Removing intermediate MLX file: {intermediate_file}")
            intermediate_file.unlink()
        
        # Verify the output file was created
        if return_code == 0 and not outfile.exists():
            logger.error(f"Failed to create optimized MLX file: {outfile}")
            return 1
        
        if return_code == 0:
            logger.info(f"Successfully optimized MLX model: {outfile}")
            logger.info(f"Final model size: {outfile.stat().st_size / (1024 * 1024):.2f} MB")
        
        return return_code
    
    except Exception as e:
        logger.error(f"Error during optimization: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return 1

def main():
    """Main entry point."""
    args = parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logger.setLevel(log_level)
    
    try:
        # Set up MLX path
        if not setup_mlx_path(args.mlx_dir):
            return 1
        
        # Step 1: Convert SafeTensors to MLX
        result = convert_safetensors_to_mlx(args)
        if not result:
            return 1
        
        intermediate_file, temp_dir = result
        
        # Step 2: Optimize the MLX model
        return_code = optimize_mlx_model(args, intermediate_file)
        
        # Clean up temporary directory if created
        if temp_dir:
            logger.info(f"Cleaning up temporary directory: {temp_dir}")
            shutil.rmtree(temp_dir)
        
        return return_code
    
    except Exception as e:
        logger.error(f"Error: {e}")
        if args.verbose:
            import traceback
            logger.debug(traceback.format_exc())
        return 1

if __name__ == "__main__":
    sys.exit(main())
