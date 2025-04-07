# SafeTensors to MLX Converter

A toolkit for working with Hugging Face models and MLX format for use with Apple's [MLX framework](https://github.com/ml-explore/mlx). It includes tools to convert SafeTensors to MLX format and to optimize MLX models for efficient inference on Apple Silicon.

## Features

### SafeTensors to MLX Conversion
- Converts SafeTensors model files to MLX format
- Supports Llama-4 models with Mixture of Experts (MoE) architecture
- Supports Qwen2.5 models including Qwen2.5-Omni-7B
- Handles multimodal models by skipping vision components
- Supports custom tokenizer formats used in Llama-4 and Qwen2.5 models
- Custom MLX format implementation for older MLX versions
- Automatically detects the MLX directory or allows custom path specification

### MLX Optimization
- Optimizes MLX models for efficient inference on Apple Silicon
- Supports various optimization options
- Automatically names output files based on optimization type
- Provides size comparison between original and optimized models
- Special handling for Mixture of Experts (MoE) models
- Model structure analysis to optimize conversion

### Two-Step Conversion and Optimization for MoE Models
- Single script that handles both conversion and optimization in one command
- Specifically designed for Llama-4 Scout and other MoE models
- Creates intermediate MLX files before optimization
- Solves the issue of models that are already in a compressed format

## Requirements

- Python 3.13.2 (also compatible with Python 3.8 or higher)
- PyTorch 2.2.0+
- MLX framework 0.9.0+ (also supports older versions with custom implementation)
- Apple Silicon Mac (M1/M2/M3 or newer)
- NumPy 1.26.0+
- h5py 3.10.0+ (for custom MLX format)

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/safetensors-to-mlx.git
   cd safetensors-to-mlx
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### SafeTensors to MLX Conversion

```bash
python safetensors_to_mlx.py --model /path/to/model --outfile /path/to/output.mlx
```

### MLX Optimization

```bash
python optimize_mlx.py --model /path/to/model.mlx --type fp16
```

### Two-Step Conversion and Optimization for MoE Models

```bash
python convert_and_optimize.py --safetensors-dir /path/to/model --type fp16
```

### SafeTensors to MLX Command Line Options

- `--model`: Path to the directory containing the model's SafeTensors files (required)
- `--outfile`: Path to write the output MLX file (default: model directory name with .mlx extension)
- `--outtype`: Output data type (default: auto)
  - Options: fp32, fp16, auto
- `--vocab-only`: Extract only the vocabulary
- `--model-name`: Override the model name in the MLX file metadata
- `--metadata`: Path to a JSON file containing metadata to add to the MLX file
- `--threads`: Number of threads to use for conversion (default: number of CPU cores)
- `--verbose`: Enable verbose logging
- `--mlx-dir`: Path to the MLX directory (default: auto-detect)

## Examples

### Basic SafeTensors to MLX Conversion

```bash
python safetensors_to_mlx.py --model /path/to/Llama-4-Scout-17B-16E-Instruct
```

### Specifying Output Format and MLX Directory

```bash
python safetensors_to_mlx.py --model /path/to/Llama-4-Scout-17B-16E-Instruct --outtype fp16
```

### Converting Only the Vocabulary

```bash
python safetensors_to_mlx.py --model /path/to/Llama-4-Scout-17B-16E-Instruct --vocab-only
```

### Basic MLX Optimization

```bash
python optimize_mlx.py --model /path/to/model.mlx --type fp16
```

### Testing and Verification

The repository includes several testing and verification scripts in the `tests` directory:

```bash
# Test model loading and tensor verification
python tests/verify_model.py --model /path/to/model.mlx

# Test basic model inference
python tests/test_inference.py --model /path/to/model.mlx

# Test Qwen2.5 model inference
python tests/qwen_mlx_inference.py --model /path/to/model.mlx --test-only
```

### MLX Optimization with Custom Output Path

```bash
python optimize_mlx.py --model /path/to/model.mlx --type fp16 --outfile /path/to/output-fp16.mlx
```

### Model Structure Analysis (MoE Detection)

```bash
python optimize_mlx.py --model /path/to/model.mlx --analyze-model --type auto
```

### MoE-Specific Optimization

```bash
python optimize_mlx.py --model /path/to/model.mlx --type fp16 --moe-expert-optimization fp16 --moe-router-optimization fp32
```

### Basic Two-Step Conversion and Optimization for MoE Models

```bash
python convert_and_optimize.py --safetensors-dir /path/to/Llama-4-Scout-17B-16E-Instruct --type fp16
```

### Advanced Two-Step Conversion with Different Optimization Types

```bash
python convert_and_optimize.py --safetensors-dir /path/to/Llama-4-Scout-17B-16E-Instruct \
  --intermediate-type fp32 --type fp16 --moe-expert-optimization fp16 --moe-router-optimization fp32 \
  --keep-intermediate
```

### Complete Conversion Pipeline (Manual Method)

```bash
# Step 1: Convert SafeTensors to MLX
python safetensors_to_mlx.py --model /path/to/Llama-4-Scout-17B-16E-Instruct

# Step 2: Optimize the resulting MLX file
python optimize_mlx.py --model /path/to/Llama-4-Scout-17B-16E-Instruct.mlx --type fp16
```

## Supported Models

This tool has been tested with:
- Llama-4 models (including the Mixture of Experts variants)
- Other models supported by MLX's conversion utilities

## How It Works

The script leverages MLX's conversion utilities to handle the conversion process. The converted models are saved in the safetensors format for maximum compatibility, with accompanying JSON files for metadata, configuration, and tokenizer information.

It adds special handling for Llama-4 models, including:

1. Support for the Mixture of Experts (MoE) architecture by properly handling router weights and expert layers
2. Custom tokenizer handling for the new tokenizer format used in Llama-4
3. Proper handling of nested configuration parameters in Llama-4 models
4. Skipping multimodal components for vision-language models

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [MLX framework](https://github.com/ml-explore/mlx) for the core conversion utilities
- Hugging Face for the SafeTensors format
- The original [safetensors-to-gguf](https://github.com/yourusername/safetensors-to-gguf) project for inspiration

## Current Limitations

This is an early version of the converter and may have limitations with certain model architectures. Future updates will address these compatibility issues.

### Notes on Data Types

MLX currently supports fp32 and fp16 data types. While some other frameworks support bf16 (bfloat16), MLX does not have native support for this format. When bf16 is requested, the converter will automatically use fp16 instead and issue a warning.
