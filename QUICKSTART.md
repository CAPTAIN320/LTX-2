# Quick Start Guide - LTX-2 on Apple Silicon

Get up and running with LTX-2 on your Mac Studio in 3 simple steps!

## Step 1: Install Dependencies (5 minutes)

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install project dependencies
uv sync --frozen

# Activate the virtual environment
source .venv/bin/activate
```

## Step 2: Download Models (30-60 minutes)

**Important**: You must authenticate with HuggingFace first to download the Gemma 3 model.

1. Accept terms at: https://huggingface.co/google/gemma-3-12b-it-qat-q4_0-unquantized
2. Create token at: https://huggingface.co/settings/tokens
3. Login:
   ```bash
   huggingface-cli login
   ```

Then run the download script:

```bash
# Download all required models (~40-50GB)
./download_models.sh
```

**Note**: This will download approximately 50GB of models. Make sure you have:

- Stable internet connection
- At least 60GB free disk space
- Time for a coffee break ☕

## Step 3: Test Your Setup (5-15 minutes)

```bash
# Run the test script
python test_mps.py
```

This will generate a short test video to verify everything works!

## What's Next?

Once the test completes successfully, you can:

### Generate Your First Video

```bash
python -m ltx_pipelines.ti2vid_two_stages \
    --checkpoint-path ./models/ltx-2-19b-dev-fp8.safetensors \
    --distilled-lora ./models/ltx-2-19b-distilled-lora-384.safetensors 0.8 \
    --spatial-upsampler-path ./models/ltx-2-spatial-upscaler-x2-1.0.safetensors \
    --gemma-root ./models/gemma-3-12b-it \
    --prompt "Your creative prompt here" \
    --output-path my_video.mp4 \
    --enable-fp8
```

### Read the Full Documentation

For detailed usage, examples, and troubleshooting, see:

- **[SETUP_APPLE_SILICON.md](SETUP_APPLE_SILICON.md)** - Complete setup guide
- **[README.md](README.md)** - Main project documentation
- **[packages/ltx-pipelines/README.md](packages/ltx-pipelines/README.md)** - Pipeline documentation

## Need Help?

- Check [SETUP_APPLE_SILICON.md](SETUP_APPLE_SILICON.md) for troubleshooting
- Join the [Discord community](https://discord.gg/ltxplatform)
- Visit the [HuggingFace model page](https://huggingface.co/Lightricks/LTX-2)

---

**Your Mac Studio with 64GB RAM is perfect for running LTX-2!** 🚀
