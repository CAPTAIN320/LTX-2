# LTX-2 for Apple Silicon Mac

Complete guide for deploying and using LTX-2 on Apple Silicon Macs with Metal Performance Shaders (MPS).

## Prerequisites

- **Hardware**: Apple Silicon Mac (M1/M2/M3/M4) with at least 32GB RAM (64GB recommended)
- **OS**: macOS 12.3 or later (for MPS support)
- **Storage**: At least 60GB free disk space for models
- **Python**: Python 3.10 or later

## Quick Start

```bash
# 1. Install dependencies
uv sync --frozen
source .venv/bin/activate

# 2. Authenticate with HuggingFace (required for Gemma 3)
huggingface-cli login

# 3. Download models
./download_models.sh
# Choose option 1 (BF16 Dev) for best MPS compatibility

# 4. Generate your first video!
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

---

## Table of Contents

1. [Model Options](#model-options)
2. [Installation](#installation)
3. [Usage Examples](#usage-examples)
4. [Command-Line Arguments](#command-line-arguments)
5. [Performance Tips](#performance-tips)
6. [Troubleshooting](#troubleshooting)
7. [What's Been Modified](#whats-been-modified)

---

## Model Options

The download script ([`download_models.py`](download_models.py)) offers 4 model checkpoint options:

### 1. BF16 Dev (Recommended for MPS) ✅

- **File**: `ltx-2-19b-dev.safetensors`
- **Size**: ~40GB
- **Usage**: `--checkpoint-path ./models/ltx-2-19b-dev.safetensors`
- **Pros**: Full MPS compatibility, no Triton dependency, best quality
- **Best for**: Apple Silicon users who want maximum compatibility

### 2. FP8 Dev

- **File**: `ltx-2-19b-dev-fp8.safetensors`
- **Size**: ~20GB
- **Usage**: `--checkpoint-path ./models/ltx-2-19b-dev-fp8.safetensors --enable-fp8`
- **Pros**: Smaller file size, lower memory usage
- **Note**: Falls back to BF16 for LoRA operations on MPS

### 3. BF16 Distilled

- **File**: `ltx-2-19b-distilled.safetensors`
- **Size**: ~40GB
- **Usage**: `--checkpoint-path ./models/ltx-2-19b-distilled.safetensors`
- **Pros**: Alternative checkpoint with different characteristics
- **Best for**: Experimenting with different model versions

### 4. FP8 Distilled

- **File**: `ltx-2-19b-distilled-fp8.safetensors`
- **Size**: ~20GB
- **Usage**: `--checkpoint-path ./models/ltx-2-19b-distilled-fp8.safetensors --enable-fp8`
- **Pros**: Smallest file size
- **Best for**: Limited disk space

---

## Installation

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

Downloads include:

- Selected LTX-2 19B Model (~20-40GB)
- Spatial Upscaler (~2GB)
- Distilled LoRA (~1GB)
- Gemma 3 Text Encoder (~25GB)

**Total**: ~50-70GB depending on model choice

### Step 5: Test Your Setup

```bash
python test_mps.py
```

---

## Usage Examples

### Basic Text-to-Video (with audio)

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

### Text-to-Video (without audio) 🔇

Add the `--no-audio` flag to disable audio:

```bash
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
    --seed 42 \
    --no-audio
```

**Benefits**: Saves ~10-20% generation time, smaller output file

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

### Image-to-Video

```bash
python -m ltx_pipelines.ti2vid_two_stages \
    --checkpoint-path ./models/ltx-2-19b-dev.safetensors \
    --distilled-lora ./models/ltx-2-19b-distilled-lora-384.safetensors 0.8 \
    --spatial-upsampler-path ./models/ltx-2-spatial-upscaler-x2-1.0.safetensors \
    --gemma-root ./models/gemma-3-12b-it \
    --prompt "Camera slowly pans across the landscape" \
    --images input_image.jpg 0 1.0 \
    --output-path output.mp4 \
    --height 512 \
    --width 768 \
    --num-frames 121 \
    --seed 42
```

---

## Command-Line Arguments

### Required Arguments

- `--checkpoint-path`: Path to LTX-2 model checkpoint
- `--gemma-root`: Path to Gemma text encoder directory
- `--prompt`: Text description of the video to generate
- `--output-path`: Where to save the output video
- `--distilled-lora`: Path and strength for distilled LoRA (two-stage pipelines)
- `--spatial-upsampler-path`: Path to spatial upsampler (two-stage pipelines)

### Optional Arguments

- `--negative-prompt`: What to avoid in the video
- `--seed`: Random seed for reproducibility (default: 42)
- `--height`: Video height in pixels, divisible by 64 (default: 512)
- `--width`: Video width in pixels, divisible by 64 (default: 768)
- `--num-frames`: Number of frames, must be (8 × K) + 1 (default: 121)
- `--frame-rate`: Frames per second (default: 25.0)
- `--num-inference-steps`: Denoising steps (default: 40)
- `--cfg-guidance-scale`: Prompt adherence strength (default: 3.0)
- `--images`: Image conditioning (format: PATH FRAME_IDX STRENGTH)
- `--lora`: Additional LoRA models (format: PATH STRENGTH)
- `--enable-fp8`: Enable FP8 mode (use with FP8 models)
- `--enhance-prompt`: Use Gemma to enhance your prompt
- `--no-audio`: Disable audio generation 🔇

---

## Performance Tips

### For Faster Generation

1. **Reduce frames**: Use 33 or 65 instead of 121
2. **Lower resolution**: Start with 512x768
3. **Fewer steps**: Use 20-30 instead of 40
4. **Disable audio**: Add `--no-audio` flag (saves ~10-20% time)
5. **Use FP8 model**: Smaller memory footprint

### For Best Quality

1. **Use BF16 Dev model**: Best compatibility and quality
2. **Higher resolution**: 1024x1536 (requires more memory)
3. **More frames**: 121 or 257 frames
4. **More steps**: 40-50 inference steps
5. **Fine-tune CFG**: Experiment with 2.5-4.0 guidance scale

### Expected Performance on Mac Studio (64GB RAM)

| Configuration                   | Time Estimate | With --no-audio |
| ------------------------------- | ------------- | --------------- |
| 512x768, 33 frames, 20 steps    | ~5-10 min     | ~4-8 min        |
| 512x768, 121 frames, 30 steps   | ~15-30 min    | ~12-24 min      |
| 1024x1536, 121 frames, 40 steps | ~45-90 min    | ~36-72 min      |

---

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
- Use FP8 model for lower memory usage
- Add `--no-audio` flag

### Slow Generation

MPS is generally slower than CUDA GPUs, but you can:

- Use the distilled pipeline for faster inference
- Reduce inference steps (20-30 instead of 40)
- Start with smaller resolutions and scale up
- Add `--no-audio` flag

### Authentication Errors (401/403)

If you get authentication errors when downloading Gemma 3:

1. Visit https://huggingface.co/google/gemma-3-12b-it-qat-q4_0-unquantized
2. Accept the license agreement
3. Create a token at https://huggingface.co/settings/tokens
4. Run `huggingface-cli login` and paste your token
5. Delete the failed download: `rm -rf ./models/gemma-3-12b-it`
6. Run `./download_models.sh` again

### Import Errors

If you get import errors:

```bash
uv sync --frozen
source .venv/bin/activate
```

---

## What's Been Modified for MPS Support

The following changes enable Apple Silicon (MPS) support:

### 1. Device Detection & Memory Management

**File**: [`packages/ltx-pipelines/src/ltx_pipelines/utils/helpers.py`](packages/ltx-pipelines/src/ltx_pipelines/utils/helpers.py)

- Updated `get_device()` to detect and use MPS when available
- Added `synchronize_device()` helper for cross-platform device synchronization
- Updated `cleanup_memory()` to handle MPS memory management

### 2. Pipeline Updates

All pipeline files updated to use device-agnostic helpers:

- [`ti2vid_two_stages.py`](packages/ltx-pipelines/src/ltx_pipelines/ti2vid_two_stages.py)
- [`ti2vid_one_stage.py`](packages/ltx-pipelines/src/ltx_pipelines/ti2vid_one_stage.py)
- [`distilled.py`](packages/ltx-pipelines/src/ltx_pipelines/distilled.py)
- [`ic_lora.py`](packages/ltx-pipelines/src/ltx_pipelines/ic_lora.py)
- [`keyframe_interpolation.py`](packages/ltx-pipelines/src/ltx_pipelines/keyframe_interpolation.py)

### 3. Core Loader

**File**: [`packages/ltx-core/src/ltx_core/loader/single_gpu_model_builder.py`](packages/ltx-core/src/ltx_core/loader/single_gpu_model_builder.py)

- Fixed hardcoded CUDA device to auto-detect MPS/CUDA/CPU

### 4. Triton Dependency

**File**: [`packages/ltx-core/src/ltx_core/loader/fuse_loras.py`](packages/ltx-core/src/ltx_core/loader/fuse_loras.py)

- Made Triton import optional (CUDA-only library not available on macOS)
- FP8 LoRA fusion falls back to standard precision on MPS

### 5. Tokenizer Loading

**File**: [`packages/ltx-core/src/ltx_core/text_encoders/gemma/encoders/base_encoder.py`](packages/ltx-core/src/ltx_core/text_encoders/gemma/encoders/base_encoder.py)

- Updated to accept `tokenizer.json` if `tokenizer.model` is missing

### 6. Audio Control

**File**: [`packages/ltx-pipelines/src/ltx_pipelines/utils/args.py`](packages/ltx-pipelines/src/ltx_pipelines/utils/args.py)

- Added `--no-audio` flag to all pipelines for disabling audio output

---

## Python API Usage

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
    checkpoint_path="./models/ltx-2-19b-dev.safetensors",  # Or any other model
    distilled_lora=distilled_lora,
    spatial_upsampler_path="./models/ltx-2-spatial-upscaler-x2-1.0.safetensors",
    gemma_root="./models/gemma-3-12b-it",
    loras=[],
    fp8transformer=False,  # Set to True if using FP8 model
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

# Save output (with or without audio)
from ltx_pipelines.utils.constants import AUDIO_SAMPLE_RATE
from ltx_pipelines.utils.media_io import encode_video
from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number

tiling_config = TilingConfig.default()
video_chunks_number = get_video_chunks_number(121, tiling_config)

encode_video(
    video=video,
    fps=25.0,
    audio=None,  # Set to None to disable audio, or use audio variable
    audio_sample_rate=None,  # Set to None when audio is None
    output_path="output.mp4",
    video_chunks_number=video_chunks_number,
)
```

---

## Additional Resources

- **Main README**: [`README.md`](README.md)
- **Pipeline Documentation**: [`packages/ltx-pipelines/README.md`](packages/ltx-pipelines/README.md)
- **Core Documentation**: [`packages/ltx-core/README.md`](packages/ltx-core/README.md)
- **HuggingFace Model**: https://huggingface.co/Lightricks/LTX-2
- **Discord Community**: https://discord.gg/ltxplatform

---

## Support

If you encounter issues:

1. Check the troubleshooting section above
2. Review the error messages carefully
3. Join the Discord community for help
4. Open an issue on GitHub with detailed error logs

---

**Your Mac Studio with 64GB RAM is fully configured for LTX-2!** 🚀
