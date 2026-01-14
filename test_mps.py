#!/usr/bin/env python3
"""
Test script for LTX-2 on Apple Silicon (MPS)
This script runs a simple text-to-video generation to verify the setup.
"""

import logging
import sys
from pathlib import Path

import torch

# Check if MPS is available
print("=" * 60)
print("LTX-2 Apple Silicon (MPS) Test Script")
print("=" * 60)
print()

print("Checking PyTorch and MPS availability...")
print(f"PyTorch version: {torch.__version__}")
print(f"MPS available: {torch.backends.mps.is_available()}")
print(f"MPS built: {torch.backends.mps.is_built()}")
print()

if not torch.backends.mps.is_available():
    print("ERROR: MPS is not available on this system!")
    print("Make sure you're running on Apple Silicon Mac with macOS 12.3+")
    sys.exit(1)

print("✓ MPS is available and ready!")
print()

# Import LTX-2 components
print("Importing LTX-2 components...")
try:
    from ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP, LoraPathStrengthAndSDOps
    from ltx_pipelines.ti2vid_two_stages import TI2VidTwoStagesPipeline
    print("✓ Successfully imported LTX-2 components")
except ImportError as e:
    print(f"ERROR: Failed to import LTX-2 components: {e}")
    print("Make sure you've run 'uv sync --frozen' to install dependencies")
    sys.exit(1)

print()

# Check model paths
MODELS_DIR = Path("./models")

# Check for either BF16 or FP8 model
CHECKPOINT_PATH_BF16 = MODELS_DIR / "ltx-2-19b-dev.safetensors"
CHECKPOINT_PATH_FP8 = MODELS_DIR / "ltx-2-19b-dev-fp8.safetensors"

if CHECKPOINT_PATH_BF16.exists():
    CHECKPOINT_PATH = CHECKPOINT_PATH_BF16
    USE_FP8 = False
    print(f"Using BF16 model: {CHECKPOINT_PATH}")
elif CHECKPOINT_PATH_FP8.exists():
    CHECKPOINT_PATH = CHECKPOINT_PATH_FP8
    USE_FP8 = True
    print(f"Using FP8 model: {CHECKPOINT_PATH}")
else:
    print("ERROR: No LTX-2 checkpoint found!")
    print(f"  Expected either: {CHECKPOINT_PATH_BF16}")
    print(f"              or: {CHECKPOINT_PATH_FP8}")
    print("Please run './download_models.sh' to download the models")
    sys.exit(1)

UPSAMPLER_PATH = MODELS_DIR / "ltx-2-spatial-upscaler-x2-1.0.safetensors"
DISTILLED_LORA_PATH = MODELS_DIR / "ltx-2-19b-distilled-lora-384.safetensors"
GEMMA_ROOT = MODELS_DIR / "gemma-3-12b-it"

print()
print("Checking model files...")
missing_files = []
for path, name in [
    (UPSAMPLER_PATH, "Spatial Upsampler"),
    (DISTILLED_LORA_PATH, "Distilled LoRA"),
    (GEMMA_ROOT, "Gemma Text Encoder Directory"),
]:
    if path.exists():
        print(f"✓ Found: {name}")
    else:
        print(f"✗ Missing: {name} at {path}")
        missing_files.append(name)

# Check specific Gemma files
if GEMMA_ROOT.exists():
    gemma_files = [
        "config.json",
        "generation_config.json",
        "model.safetensors.index.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "preprocessor_config.json"
    ]

    for file in gemma_files:
        file_path = GEMMA_ROOT / file
        if file_path.exists():
            print(f"✓ Found Gemma file: {file}")
        else:
            print(f"✗ Missing Gemma file: {file}")
            missing_files.append(f"Gemma {file}")

if missing_files:
    print()
    print("ERROR: Missing required model files!")
    print("Please run './download_models.sh' to download the missing files")
    sys.exit(1)

print()
print("=" * 60)
print("Starting test generation...")
print("=" * 60)
print()

# Configure logging
logging.getLogger().setLevel(logging.INFO)

# Initialize pipeline
print("Initializing pipeline (this may take a few minutes)...")
distilled_lora = [
    LoraPathStrengthAndSDOps(
        str(DISTILLED_LORA_PATH),
        0.8,
        LTXV_LORA_COMFY_RENAMING_MAP
    ),
]

try:
    pipeline = TI2VidTwoStagesPipeline(
        checkpoint_path=str(CHECKPOINT_PATH),
        distilled_lora=distilled_lora,
        spatial_upsampler_path=str(UPSAMPLER_PATH),
        gemma_root=str(GEMMA_ROOT),
        loras=[],
        fp8transformer=USE_FP8,  # Only use FP8 if we have the FP8 model
    )
    print("✓ Pipeline initialized successfully!")
except Exception as e:
    print(f"ERROR: Failed to initialize pipeline: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()

# Test parameters (small for quick test)
TEST_PARAMS = {
    "prompt": "A serene mountain landscape with a flowing river, cinematic lighting",
    "negative_prompt": "blurry, low quality, distorted",
    "seed": 42,
    "height": 512,  # Small resolution for quick test
    "width": 768,
    "num_frames": 33,  # Short video for quick test
    "frame_rate": 25.0,
    "num_inference_steps": 20,  # Reduced steps for quick test
    "cfg_guidance_scale": 3.0,
    "images": [],  # No image conditioning for this test
}

print("Test parameters:")
for key, value in TEST_PARAMS.items():
    print(f"  {key}: {value}")
print()

# Run generation
OUTPUT_PATH = Path("./test_output.mp4")
print(f"Generating video (this will take several minutes on MPS)...")
print(f"Output will be saved to: {OUTPUT_PATH}")
print()

try:
    video, audio = pipeline(**TEST_PARAMS)

    # Save output
    from ltx_pipelines.utils.constants import AUDIO_SAMPLE_RATE
    from ltx_pipelines.utils.media_io import encode_video
    from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number

    tiling_config = TilingConfig.default()
    video_chunks_number = get_video_chunks_number(TEST_PARAMS["num_frames"], tiling_config)

    encode_video(
        video=video,
        fps=TEST_PARAMS["frame_rate"],
        audio=audio,
        audio_sample_rate=AUDIO_SAMPLE_RATE,
        output_path=str(OUTPUT_PATH),
        video_chunks_number=video_chunks_number,
    )

    print()
    print("=" * 60)
    print("✓ SUCCESS! Test completed successfully!")
    print("=" * 60)
    print()
    print(f"Output video saved to: {OUTPUT_PATH}")
    print()
    print("Your LTX-2 setup is working correctly on Apple Silicon!")
    print("You can now use the full pipeline with higher quality settings.")

except Exception as e:
    print()
    print("=" * 60)
    print("✗ ERROR: Test failed!")
    print("=" * 60)
    print()
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
