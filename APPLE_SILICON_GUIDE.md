# LTX-2 Complete Guide for Apple Silicon Mac

This guide will help you deploy and run LTX-2 on your Apple Silicon Mac Studio with 64GB RAM using Metal Performance Shaders (MPS).

## Prerequisites

- **Hardware**: Apple Silicon Mac (M1/M2/M3/M4) with at least 32GB RAM (64GB recommended)
- **OS**: macOS 12.3 or later (for MPS support)
- **Storage**: At least 60GB free disk space for models
- **Python**: Python 3.10 or later

## Quick Start

```bash
# Step 1: Install dependencies
uv sync --frozen
source .venv/bin/activate

# Step 2: Authenticate with HuggingFace (required for Gemma 3)
huggingface-cli login

# Step 3: Download models
./download_models.sh
# Choose option 1 (BF16 Dev) for best MPS compatibility

# Step 4: Generate your first video!
python -m ltx_pipelines.ti2vid_two_stages \
    --checkpoint-path ./models/ltx-2-19b-dev.safetensors \
    --distilled-lora ./models/ltx-2-19b-distilled-lora-384.safetensors 0.8 \
    --spatial-upsampler-path ./models/ltx-2-spatial-upscaler-x2-1.0.safetensors \
    --gemma-root ./models/gemma-3-12b-it \
    --prompt "A serene mountain landscape with flowing river" \
    --output-path output.mp4 \
    --height 512 \
    --width 768 \
    --num-frames 33 \
    --seed 42
```

## Model Options

The download script ([`download_models.py`](download_models.py)) offers 4 model checkpoint options:

### 1. BF16 Dev (Recommended for MPS) ✅

- **File**: `ltx-2-19b-dev.safetensors`
- **Size**: ~40GB
- **Pros**: Full compatibility with MPS, no Triton dependency, best quality
- **Cons**: Larger file size
- **Best for**: Apple Silicon users who want maximum compatibility and quality

### 2. FP8 Dev

- **File**: `ltx-2-19b-dev-fp8.safetensors`
- **Size**: ~20GB
- **Pros**: Smaller file size, lower memory usage
- **Cons**: Requires Triton for optimal performance (CUDA-only), falls back to BF16 on MPS
- **Best for**: Users with limited disk space

### 3. BF16 Distilled

- **File**: `ltx-2-19b-distilled.safetensors`
- **Size**: ~40GB
- **Pros**: Alternative checkpoint, may have different characteristics
- **Cons**: Larger file size
- **Best for**: Users who want to experiment with different checkpoints

### 4. FP8 Distilled

- **File**: `ltx-2-19b-distilled-fp8.safetensors`
- **Size**: ~20GB
- **Pros**: Smaller file size
- **Cons**: Requires Triton for optimal performance (CUDA-only)
- **Best for**: Users with limited disk space who want to try the distilled checkpoint

## What's Been Modified for MPS Support

1. **Device Detection** ([`packages/ltx-pipelines/src/ltx_pipelines/utils/helpers.py`](packages/ltx-pipelines/src/ltx_pipelines/utils/helpers.py)):

   - Updated `get_device()` to detect and use MPS when available
   - Added `synchronize_device()` helper for cross-platform device synchronization
   - Updated `cleanup_memory()` to handle MPS memory management

2. **Pipeline Updates**: All pipeline files updated to use device-agnostic helpers:

   - [`ti2vid_two_stages.py`](packages/ltx-pipelines/src/ltx_pipelines/ti2vid_two_stages.py)
   - [`ti2vid_one_stage.py`](packages/ltx-pipelines/src/ltx_pipelines/ti2vid_one_stage.py)
   - [`distilled.py`](packages/ltx-pipelines/src/ltx_pipelines/distilled.py)
   - [`ic_lora.py`](packages/ltx-pipelines/src/ltx_pipelines/ic_lora.py)
   - [`keyframe_interpolation.py`](packages/ltx-pipelines/src/ltx_pipelines/keyframe_interpolation.py)

3. **Core Loader** ([`packages/ltx-core/src/ltx_core/loader/single_gpu_model_builder.py`](packages/ltx-core/src/ltx_core/loader/single_gpu_model_builder.py)):

   - Fixed hardcoded CUDA device to auto-detect MPS/CUDA/CPU

4. **Triton Dependency** ([`packages/ltx-core/src/ltx_core/loader/fuse_loras.py`](packages/ltx-core/src/ltx_core/loader/fuse_loras.py)):
   - Made Triton import optional (CUDA-only library not available on macOS)
   - FP8 LoRA fusion will fall back to standard precision on MPS

## Installation Steps

### Step 1: Install UV Package Manager

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
# Or: brew install uv
```

### Step 2: Setup Environment

```bash
cd /Users/yaqub.mahmoud/github/LTX-2
uv sync --frozen
source .venv/bin/activate
```

### Step 3: Authenticate with HuggingFace

The Gemma 3 text encoder is a gated model requiring authentication:

1. Visit https://huggingface.co/google/gemma-3-12b-it-qat-q4_0-unquantized and accept the license
2. Create a token at https://huggingface.co/settings/tokens
3. Login:
   ```bash
   huggingface-cli login
   ```

### Step 4: Download Models

```bash
chmod +x download_models.sh
./download_models.sh
```

Choose your preferred model when prompted (Option 1 recommended for MPS).

### Step 5: Test Your Setup

```bash
python test_mps.py
```

## Usage Examples

### Basic Text-to-Video (BF16 Dev Model)

```bash
python -m ltx_pipelines.ti2vid_two_stages \
    --checkpoint-path ./models/ltx-2-19b-dev.safetensors \
    --distilled-lora ./models/ltx-2-19b-distilled-lora-384.safetensors 0.8 \
    --spatial-upsampler-path ./models/ltx-2-spatial-upscaler-x2-1.0.safetensors \
    --gemma-root ./models/gemma-3-12b-it \
    --prompt "A serene mountain landscape with flowing river, cinematic lighting" \
    --negative-prompt "blurry, low quality, distorted" \
    --output-path output.mp4 \
    --height 512 \
    --width 768 \
    --num-frames 121 \
    --frame-rate 25.0 \
    --num-inference-steps 40 \
    --cfg-guidance-scale 3.0 \
    --seed 42
```

### Using Distilled Model

```bash
python -m ltx_pipelines.ti2vid_two_stages \
    --checkpoint-path ./models/ltx-2-19b-distilled.safetensors \
    --distilled-lora ./models/ltx-2-19b-distilled-lora-384.safetensors 0.8 \
    --spatial-upsampler-path ./models/ltx-2-spatial-upscaler-x2-1.0.safetensors \
    --gemma-root ./models/gemma-3-12b-it \
    --prompt "Your creative prompt here" \
    --output-path output.mp4 \
    --height 512 \
    --width 768 \
    --num-frames 33 \
    --seed 42
```

### Image-to-Video Generation

```bash
python -m ltx_pipelines.ti2vid_two_stages \
    --checkpoint-path ./models/ltx-2-19b-dev.safetensors \
    --distilled-lora ./models/ltx-2-19b-distilled-lora-384.safetensors 0.8 \
    --spatial-upsampler-path ./models/ltx-2-spatial-upscaler-x2-1.0.safetensors \
    --gemma-root ./models/gemma-3-12b-it \
    --prompt "Camera slowly pans across the landscape" \
    --negative-prompt "blurry, low quality" \
    --images input_image.jpg 0 1.0 \
    --output-path output.mp4 \
    --height 512 \
    --width 768 \
    --num-frames 121 \
    --frame-rate 25.0 \
    --num-inference-steps 40 \
    --cfg-guidance-scale 3.0 \
    --seed 42
```

## Performance Tips for Apple Silicon

1. **Use BF16 Model**: The BF16 version (`ltx-2-19b-dev.safetensors`) is recommended for full MPS compatibility.

2. **Start Small**: Begin with lower resolutions (512x768) and fewer frames (33-65) to test, then scale up.

3. **Memory Management**: With 64GB RAM, you should be able to run full-resolution generations (1024x1536, 121 frames), but it will be slower than on CUDA GPUs.

4. **Reduce Inference Steps**: You can reduce `num_inference_steps` from 40 to 20-30 for faster generation with minimal quality loss.

5. **Monitor Activity Monitor**: Keep an eye on memory usage during generation.

## Expected Performance

On your Mac Studio with 64GB RAM:

- **Small test** (512x768, 33 frames, 20 steps): ~5-10 minutes
- **Medium quality** (512x768, 121 frames, 30 steps): ~15-30 minutes
- **High quality** (1024x1536, 121 frames, 40 steps): ~45-90 minutes

These are estimates and will vary based on prompt complexity and system load.

## Troubleshooting

### MPS Not Available

If you get "MPS is not available":

- Ensure you're on macOS 12.3 or later
- Verify you have an Apple Silicon Mac (not Intel)
- Update to the latest macOS version

### Out of Memory Errors

If you run out of memory:

- Reduce resolution (try 512x768 instead of 1024x1536)
- Reduce number of frames (try 65 instead of 121)
- Close other applications
- Try the FP8 model for lower memory usage

### Slow Generation

MPS is generally slower than CUDA GPUs, but you can:

- Use the distilled pipeline for faster inference
- Reduce inference steps (20-30 instead of 40)
- Start with smaller resolutions and scale up

### Import Errors

If you get import errors:

```bash
# Reinstall dependencies
uv sync --frozen
source .venv/bin/activate
```

### Authentication Errors (401/403)

If you get authentication errors when downloading Gemma 3:

1. Visit https://huggingface.co/google/gemma-3-12b-it-qat-q4_0-unquantized
2. Accept the license agreement
3. Create a token at https://huggingface.co/settings/tokens
4. Run `huggingface-cli login` and paste your token
5. Delete the failed download: `rm -rf ./models/gemma-3-12b-it`
6. Run `./download_models.sh` again

## Additional Resources

- **Main README**: [`README.md`](README.md)
- **Pipeline Documentation**: [`packages/ltx-pipelines/README.md`](packages/ltx-pipelines/README.md)
- **Core Documentation**: [`packages/ltx-core/README.md`](packages/ltx-core/README.md)
- **HuggingFace Model**: https://huggingface.co/Lightricks/LTX-2
- **Discord Community**: https://discord.gg/ltxplatform

## Support

If you encounter issues:

1. Check the troubleshooting section above
2. Review the error messages carefully
3. Join the Discord community for help
4. Open an issue on GitHub with detailed error logs

---

**Note**: This setup has been specifically configured for Apple Silicon Macs. The modifications ensure proper MPS device detection and memory management for optimal performance on your Mac Studio.
