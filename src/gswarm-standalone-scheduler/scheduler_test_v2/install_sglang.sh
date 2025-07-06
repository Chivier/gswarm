#!/bin/bash
# Install SGLang with all required dependencies

echo "Installing SGLang and dependencies..."

# Install missing dependency from error
pip install orjson

# Install SGLang with all features
pip install "sglang[all]"

# Install FlashInfer for better performance (optional but recommended)
# Choose the right version based on your CUDA version
# For CUDA 12.1:
pip install flashinfer -i https://flashinfer.ai/whl/cu121/torch2.4/

# For CUDA 11.8:
# pip install flashinfer -i https://flashinfer.ai/whl/cu118/torch2.4/

echo "Installation complete!"
echo "You can verify with: python -c 'import sglang; print(sglang.__version__)'"