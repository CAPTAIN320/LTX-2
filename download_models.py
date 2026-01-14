#!/usr/bin/env python3
"""
Python script to download LTX-2 models using huggingface_hub.
This is more robust than curl and handles authentication/redirects better.
"""

import os
import sys
from pathlib import Path
from huggingface_hub import hf_hub_download, snapshot_download

# Configuration
MODELS_DIR = Path("./models")
GEMMA_DIR = MODELS_DIR / "gemma-3-12b-it"

def download_ltx2_models():
    print("=" * 60)
    print("Downloading LTX-2 Models")
    print("=" * 60)

    MODELS_DIR.mkdir(exist_ok=True)

    # 1. Choose Model Version
    print("\nChoose LTX-2 Model Version:")
    print("1) BF16 (Recommended for Apple Silicon/MPS) - ~40GB")
    print("2) FP8 (Smaller, but requires Triton/CUDA for LoRA) - ~20GB")

    choice = input("Enter choice [1-2] (default: 1): ").strip()

    if choice == "2":
        filename = "ltx-2-19b-dev-fp8.safetensors"
        print(f"\nSelected: FP8 Model ({filename})")
    else:
        filename = "ltx-2-19b-dev.safetensors"
        print(f"\nSelected: BF16 Model ({filename})")

    print(f"\nDownloading {filename}...")
    try:
        hf_hub_download(
            repo_id="Lightricks/LTX-2",
            filename=filename,
            local_dir=MODELS_DIR,
            local_dir_use_symlinks=False
        )
        print("✓ Download complete")
    except Exception as e:
        print(f"Error downloading {filename}: {e}")
        sys.exit(1)

    # 2. Spatial Upscaler
    print("\nDownloading Spatial Upscaler...")
    try:
        hf_hub_download(
            repo_id="Lightricks/LTX-2",
            filename="ltx-2-spatial-upscaler-x2-1.0.safetensors",
            local_dir=MODELS_DIR,
            local_dir_use_symlinks=False
        )
        print("✓ Download complete")
    except Exception as e:
        print(f"Error downloading upscaler: {e}")

    # 3. Distilled LoRA
    print("\nDownloading Distilled LoRA...")
    try:
        hf_hub_download(
            repo_id="Lightricks/LTX-2",
            filename="ltx-2-19b-distilled-lora-384.safetensors",
            local_dir=MODELS_DIR,
            local_dir_use_symlinks=False
        )
        print("✓ Download complete")
    except Exception as e:
        print(f"Error downloading LoRA: {e}")

def download_gemma_models():
    print("\n" + "=" * 60)
    print("Downloading Gemma 3 Text Encoder")
    print("=" * 60)

    GEMMA_DIR.mkdir(parents=True, exist_ok=True)

    repo_id = "google/gemma-3-12b-it-qat-q4_0-unquantized"

    # List of files to download
    files = [
        "config.json",
        "generation_config.json",
        "model.safetensors.index.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "preprocessor_config.json",
    ]

    # Download config files
    print("\nDownloading config files...")
    for file in files:
        print(f"  - {file}")
        try:
            hf_hub_download(
                repo_id=repo_id,
                filename=file,
                local_dir=GEMMA_DIR,
                local_dir_use_symlinks=False
            )
        except Exception as e:
            print(f"  ! Error downloading {file}: {e}")
            if "401" in str(e) or "403" in str(e):
                print("\n" + "!" * 60)
                print("ERROR: Authentication failed (401/403 Forbidden)")
                print("!" * 60)
                print(f"\nThe model '{repo_id}' is a gated model.")
                print("You need to accept the license agreement and authenticate to download it.")
                print("\nSTEPS TO FIX:")
                print("1. Go to: https://huggingface.co/google/gemma-3-12b-it-qat-q4_0-unquantized")
                print("2. Accept the license agreement/terms of use on the page.")
                print("3. Create an access token at: https://huggingface.co/settings/tokens")
                print("4. Run the following command in your terminal:")
                print("   huggingface-cli login")
                print("5. Paste your token when prompted.")
                print("\nAfter logging in, please delete the failed download directory:")
                print(f"   rm -rf {GEMMA_DIR}")
                print("Then run this script again.")
                sys.exit(1)

    # Download model shards
    print("\nDownloading model shards (this may take a while)...")
    for i in range(1, 6):
        filename = f"model-{i:05d}-of-00005.safetensors"
        print(f"  - {filename}")
        try:
            hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=GEMMA_DIR,
                local_dir_use_symlinks=False
            )
        except Exception as e:
            print(f"  ! Error downloading {filename}: {e}")

def main():
    print("LTX-2 Model Downloader")
    print("This script uses huggingface_hub to reliably download models.")

    try:
        import huggingface_hub
    except ImportError:
        print("Error: huggingface_hub not found.")
        print("Please run: pip install huggingface_hub")
        sys.exit(1)

    download_ltx2_models()
    download_gemma_models()

    print("\n" + "=" * 60)
    print("All downloads finished!")
    print("=" * 60)

if __name__ == "__main__":
    main()
