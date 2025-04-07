#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
convert_and_optimize.py - A two-step tool for MoE models that first converts SafeTensors to MLX,
then applies optimization to the MLX file.

This approach addresses the challenge with Llama-4 Scout and other MoE models that may be in a
compressed format that can't be optimized using standard MLX tools.
"""

import argparse
import logging
import os
import sys
import tempfile
import shutil
import datetime
from pathlib import Path

# Configure logging
logger = logging.getLogger("convert-and-optimize")
handler = logging.StreamHandler()
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
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
                major, minor, patch = 0, 0, 0
                
                if len(version_parts) >= 1:
                    major = int(version_parts[0])
                if len(version_parts) >= 2:
                    minor = int(version_parts[1])
                if len(version_parts) >= 3:
                    # Handle patch versions that might have additional text (e.g., '1.dev0')
                    patch_str = version_parts[2].split('.')[0].split('-')[0]
                    try:
                        patch = int(patch_str)
                    except ValueError:
                        pass
                
                # Check if version is at least 0.9.0 (recommended for Python 3.13)
                if major == 0 and minor < 9:
                    logger.warning(
                        f"MLX version {version_str} may be outdated. "
                        f"Recommended version is at least 0.9.0 for Python 3.13")
            except (ValueError, IndexError):
                logger.warning(f"Could not parse MLX version: {version_str}")
        
        # Check for save_safetensors function
        has_save_safetensors = hasattr(mlx.core, 'save_safetensors')
        logger.info(f"MLX has save_safetensors function: {has_save_safetensors}")
        
        return True
    except ImportError:
        logger.error(
            "MLX not found. Please install MLX using: pip install mlx")
        return False


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Two-step tool: first converts SafeTensors to MLX, then optimizes")

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
        "--intermediate-type",
        type=str,
        choices=[
            "fp32",
            "fp16"],
        default="fp16",
        help="Format for the intermediate MLX file (default: fp16)")

    parser.add_argument(
        "--moe-expert-optimization",
        type=str,
        choices=[
            "fp32",
            "fp16",
            "same"],
        default="same",
        help="Optimization type for MoE expert layers")

    parser.add_argument(
        "--moe-router-optimization",
        type=str,
        choices=[
            "fp32",
            "fp16",
            "same"],
        default="same",
        help="Optimization type for MoE router layers")

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
        "--output-tensor-type",
        type=str,
        choices=[
            "fp32",
            "fp16"],
        default=None,
        help="Output tensor type (fp32, fp16)")

    parser.add_argument(
        "--token-embedding-type",
        type=str,
        choices=[
            "fp32",
            "fp16"],
        default=None,
        help="Token embedding tensor type (fp32, fp16)")

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
        logger.error(
            f"SafeTensors directory not found: {args.safetensors_dir}")
        return None

    # Create a temporary directory for the intermediate MLX file if needed
    temp_dir = None
    if args.outfile is None and args.outdir is None:
        temp_dir = tempfile.mkdtemp(prefix="mlx_convert_")
        logger.info(
            f"Created temporary directory for intermediate files: {temp_dir}")
        intermediate_file = Path(temp_dir) / f"{args.safetensors_dir.name}.mlx"
    else:
        # Determine the intermediate file path
        if args.outfile:
            # Use the same directory as the final output file
            intermediate_file = args.outfile.parent / \
                f"{args.safetensors_dir.name}_intermediate.mlx"
        else:
            # Use the specified output directory
            intermediate_file = args.outdir / \
                f"{args.safetensors_dir.name}_intermediate.mlx"

    logger.info(f"Intermediate MLX file: {intermediate_file}")

    # Import the safetensors_to_mlx module
    try:
        import importlib.util
        script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
        convert_script_path = script_dir / "safetensors_to_mlx.py"

        if not convert_script_path.exists():
            logger.error(
                f"Could not find safetensors_to_mlx.py script at {convert_script_path}")
            return None

        spec = importlib.util.spec_from_file_location(
            "safetensors_to_mlx", str(convert_script_path))
        convert_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(convert_module)

        # Call the conversion function
        logger.info(
            f"Converting SafeTensors model to MLX format: {args.safetensors_dir}")

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
            logger.error(
                f"Failed to create intermediate MLX file: {intermediate_file}")
            return None

        logger.info(
            f"Successfully converted SafeTensors model to MLX format: {intermediate_file}")
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
        outfile = intermediate_file.parent / \
            f"{args.safetensors_dir.name}_{args.type}.mlx"

    logger.info(f"Output optimized MLX file: {outfile}")
    
    # Check if the intermediate file is in our custom format
    custom_format = False
    tensor_dir = intermediate_file.with_suffix('.tensors')
    if tensor_dir.exists() and tensor_dir.is_dir():
        try:
            import h5py
            with h5py.File(intermediate_file, 'r') as f:
                if 'format' in f.attrs and f.attrs['format'] == 'mlx_custom':
                    custom_format = True
                    # Log additional metadata if available
                    if 'python_version' in f.attrs:
                        logger.info(f"Model was created with Python {f.attrs['python_version']}")
                    if 'numpy_version' in f.attrs:
                        logger.info(f"Model was created with NumPy {f.attrs['numpy_version']}")
                    if 'creation_date' in f.attrs:
                        logger.info(f"Model creation date: {f.attrs['creation_date']}")
        except Exception as e:
            # If we can't open the file with h5py, it's not our custom format
            logger.debug(f"Error checking custom format: {e}")
            pass
    
    if custom_format:
        logger.info(f"Detected custom MLX format, skipping optimization")
        
        # Copy the intermediate file to the output location
        import shutil
        import json
        
        # Create parent directories if they don't exist
        outfile.parent.mkdir(parents=True, exist_ok=True)
        
        # Copy the main file
        shutil.copy2(intermediate_file, outfile)
        
        # Copy the tensor directory
        tensor_out_dir = outfile.with_suffix('.tensors')
        if tensor_out_dir.exists():
            shutil.rmtree(tensor_out_dir)
        shutil.copytree(tensor_dir, tensor_out_dir)
        
        # Copy and update the metadata and config files
        for suffix in ['.metadata.json', '.config.json', '.tokenizer.json']:
            src_file = intermediate_file.with_suffix(suffix)
            if src_file.exists():
                dst_file = outfile.with_suffix(suffix)
                
                # For metadata, add optimization info
                if suffix == '.metadata.json':
                    try:
                        with open(src_file, 'r', encoding='utf-8') as f:
                            metadata = json.load(f)
                        
                        # Add optimization info
                        metadata['optimization'] = {
                            'type': args.type,
                            'date': str(datetime.datetime.now()),
                            'python_version': sys.version,
                            'mlx_version': getattr(mlx, '__version__', 'unknown')
                        }
                        
                        with open(dst_file, 'w', encoding='utf-8') as f:
                            json.dump(metadata, f, indent=2, ensure_ascii=False)
                    except Exception as e:
                        logger.warning(f"Error updating metadata: {e}")
                        # Fallback to simple copy
                        shutil.copy2(src_file, dst_file)
                else:
                    # Simple copy for other files
                    shutil.copy2(src_file, dst_file)
        
        # Clean up the intermediate file if not keeping it
        if not args.keep_intermediate:
            logger.info(f"Removing intermediate MLX file: {intermediate_file}")
            intermediate_file.unlink()
            # Also remove the tensor directory and metadata files
            if tensor_dir.exists():
                shutil.rmtree(tensor_dir)
            for suffix in ['.metadata.json', '.config.json', '.tokenizer.json']:
                src_file = intermediate_file.with_suffix(suffix)
                if src_file.exists():
                    src_file.unlink()
        
        logger.info(f"Successfully copied MLX model to: {outfile}")
        logger.info(f"Final model size: {outfile.stat().st_size / (1024 * 1024):.2f} MB")
        
        return 0
    
    # Standard optimization for regular MLX format
    # Import the optimize_mlx module
    try:
        import importlib.util
        script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
        optimize_script_path = script_dir / "optimize_mlx.py"

        if not optimize_script_path.exists():
            logger.error(
                f"Could not find optimize_mlx.py script at {optimize_script_path}")
            return 1

        spec = importlib.util.spec_from_file_location(
            "optimize_mlx", str(optimize_script_path))
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
            logger.info(
                f"Final model size: {outfile.stat().st_size / (1024 * 1024):.2f} MB")

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
