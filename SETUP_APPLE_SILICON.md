# LTX-2 Setup Guide for Apple Silicon Mac

This guide will help you deploy and run LTX-2 on your Apple Silicon Mac Studio with 64GB RAM using Metal Performance Shaders (MPS).

## Prerequisites

- **Hardware**: Apple Silicon Mac (M1/M2/M3/M4) with at least 32GB RAM (64GB recommended)
- **OS**: macOS 12.3 or later (for MPS support)
- **Storage**: At least 60GB free disk space for models
- **Python**: Python 3.10 or later

## What's Been Modified

The following changes have been made to enable Apple Silicon (MPS) support:

1. **Device Detection** ([`packages/ltx-pipelines/src/ltx_pipelines/utils/helpers.py`](packages/ltx-pipelines/src/ltx_pipelines/utils/helpers.py)):

   - Updated `get_device()` to detect and use MPS when available
   - Added `synchronize_device()` helper for cross-platform device synchronization
   - Updated `cleanup_memory()` to handle MPS memory management

2. **Pipeline Updates**: All pipeline files have been updated to use the new device-agnostic helpers:

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

If you don't have `uv` installed:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Or using Homebrew:

```bash
brew install uv
```

### Step 2: Clone and Setup Environment

```bash
# Navigate to the LTX-2 directory (you're already here!)
cd /Users/yaqub.mahmoud/github/LTX-2

# Install dependencies using uv
uv sync --frozen

# Activate the virtual environment
source .venv/bin/activate
```

### Step 3: Download Models (Recommended: Use BF16 Model for MPS)

**Important Note**: Due to Triton being CUDA-only, FP8 model loading has limitations on MPS. For best compatibility on Apple Silicon, you have two options:

**Option A: Use BF16 Model (Recommended for MPS)**

```bash
# Edit download_models.sh and change the model download to:
# ltx-2-19b-dev.safetensors (BF16 version)
```

**Option B: Use FP8 Model (May have limitations)**

```bash
# Use the default FP8 model
# Note: FP8 LoRA fusion will fall back to BF16 on MPS
```

For now, let's proceed with the download script as-is:

The models are large (40-50GB total). Make sure you have enough disk space and a stable internet connection.

```bash
# Make the download script executable
chmod +x download_models.sh

# Run the download script
./download_models.sh
```

This will download:

- LTX-2 19B FP8 Model (~20GB) - FP8 version is recommended for Mac
- Spatial Upscaler (~2GB)
- Distilled LoRA (~1GB)
- Gemma 3 Text Encoder (~25GB)

**Note**: The download may take 30-60 minutes depending on your internet speed.

### Step 5: Verify Installation

Run the test script to verify everything is working:

```bash
# Make the test script executable
chmod +x test_mps.py

# Run the test
python test_mps.py
```

This will:

1. Check if MPS is available
2. Verify all model files are present
3. Run a quick test generation (small resolution, short video)
4. Save output to `test_output.mp4`

The test should take 5-15 minutes on your Mac Studio with 64GB RAM.

## Usage Examples

### Basic Text-to-Video Generation

```bash
python -m ltx_pipelines.ti2vid_two_stages \
    --checkpoint-path ./models/ltx-2-19b-dev-fp8.safetensors \
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
    --seed 42 \
    --enable-fp8
```

### Image-to-Video Generation

```bash
python -m ltx_pipelines.ti2vid_two_stages \
    --checkpoint-path ./models/ltx-2-19b-dev-fp8.safetensors \
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
    --seed 42 \
    --enable-fp8
```

### Python API Usage

```python
from ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP, LoraPathStrengthAndSDOps
from ltx_pipelines.ti2vid_two_stages import TI2VidTwoStagesPipeline

# Initialize pipeline
distilled_lora = [
    LoraPathStrengthAndSDOps(
        "./models/ltx-2-19b-distilled-lora-384.safetensors",
        0.8,
        LTXV_LORA_COMFY_RENAMING_MAP
    ),
]

pipeline = TI2VidTwoStagesPipeline(
    checkpoint_path="./models/ltx-2-19b-dev-fp8.safetensors",
    distilled_lora=distilled_lora,
    spatial_upsampler_path="./models/ltx-2-spatial-upscaler-x2-1.0.safetensors",
    gemma_root="./models/gemma-3-12b-it",
    loras=[],
    fp8transformer=True,  # Recommended for Mac
)

# Generate video
video, audio = pipeline(
    prompt="A beautiful sunset over the ocean with waves crashing",
    negative_prompt="blurry, low quality",
    seed=42,
    height=512,
    width=768,
    num_frames=121,
    frame_rate=25.0,
    num_inference_steps=40,
    cfg_guidance_scale=3.0,
    images=[],  # Or [("image.jpg", 0, 1.0)] for image conditioning
)

# Save output
from ltx_pipelines.utils.constants import AUDIO_SAMPLE_RATE
from ltx_pipelines.utils.media_io import encode_video
from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number

tiling_config = TilingConfig.default()
video_chunks_number = get_video_chunks_number(121, tiling_config)

encode_video(
    video=video,
    fps=25.0,
    audio=audio,
    audio_sample_rate=AUDIO_SAMPLE_RATE,
    output_path="output.mp4",
    video_chunks_number=video_chunks_number,
)
```

## Performance Tips for Apple Silicon

1. **Use FP8 Model**: The FP8 version (`ltx-2-19b-dev-fp8.safetensors`) is recommended for better memory efficiency on Mac.

2. **Enable FP8 Transformer**: Always use `--enable-fp8` flag or `fp8transformer=True` in Python.

3. **Start Small**: Begin with lower resolutions (512x768) and fewer frames (33-65) to test, then scale up.

4. **Memory Management**: With 64GB RAM, you should be able to run full-resolution generations (1024x1536, 121 frames), but it will be slower than on CUDA GPUs.

5. **Reduce Inference Steps**: You can reduce `num_inference_steps` from 40 to 20-30 for faster generation with minimal quality loss.

6. **Monitor Activity Monitor**: Keep an eye on memory usage during generation.

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
- Ensure you're using the FP8 model with `--enable-fp8`

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
