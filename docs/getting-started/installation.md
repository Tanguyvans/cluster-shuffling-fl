# Installation Guide

## Prerequisites

- **Python**: 3.8 or higher
- **Operating System**: Linux, macOS, or Windows (WSL recommended)
- **Hardware**:
  - CPU: Multi-core processor
  - RAM: 8GB minimum, 16GB recommended
  - GPU: Optional (CUDA-compatible for faster training)
  - Disk: 10GB free space (for datasets and results)

## Quick Install

### 1. Clone the Repository

```bash
git clone https://github.com/Tanguyvans/cluster-shuffling-fl.git
cd cluster-shuffling-fl
```

### 2. Install Dependencies

```bash
pip3 install -r requirements.txt
```

This will install all required packages including:
- PyTorch (deep learning framework)
- Flower (federated learning library)
- Opacus (differential privacy)
- NumPy, scikit-learn (data processing)
- Cryptography (SMPC)

### 3. Verify Installation

```bash
python3 test_gradient_pruning.py
```

If all tests pass, you're ready to go! ✅

## Detailed Installation

### Option 1: Virtual Environment (Recommended)

```bash
# Create virtual environment
python3 -m venv env

# Activate (Linux/macOS)
source env/bin/activate

# Activate (Windows)
env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Option 2: Conda Environment

```bash
# Create conda environment
conda create -n cluster-fl python=3.10

# Activate
conda activate cluster-fl

# Install PyTorch with CUDA (if available)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install other dependencies
pip install -r requirements.txt
```

### Option 3: Docker (Coming Soon)

Docker support is planned for easier deployment.

## GPU Support

### CUDA Installation

For GPU acceleration, install CUDA-enabled PyTorch:

```bash
# For CUDA 11.8
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Verify GPU availability
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Apple Silicon (M1/M2)

For Mac with Apple Silicon:

```bash
# PyTorch with MPS (Metal Performance Shaders) support
pip3 install torch torchvision torchaudio

# Verify MPS availability
python3 -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}')"
```

## Dataset Setup

Datasets are automatically downloaded on first run, but you can pre-download:

### CIFAR-10/100

```bash
python3 -c "from torchvision import datasets; datasets.CIFAR10(root='./dataset/cifar10', download=True)"
```

### FFHQ-128

FFHQ requires manual download:

1. Visit [FFHQ Dataset](https://github.com/NVlabs/ffhq-dataset)
2. Download FFHQ-128 thumbnails
3. Extract to `dataset/ffhq_dataset/`

**Structure:**
```
dataset/ffhq_dataset/
├── 00000.png
├── 00001.png
└── ...
```

### Caltech256

```bash
# Download will start automatically on first use
# Or manually prepare in dataset/caltech256/
```

## Verify Setup

Run the test suite:

```bash
# Test gradient pruning
python3 test_gradient_pruning.py

# Quick FL training test (single round)
python3 -c "from config import settings; settings['n_rounds']=1; exec(open('main.py').read())"
```

## Common Issues

### Issue: `ModuleNotFoundError: No module named 'torch'`

**Solution**: Ensure PyTorch is installed
```bash
pip3 install torch
```

### Issue: `CUDA out of memory`

**Solution**: Reduce batch size in config.py
```python
"batch_size": 16  # Reduce from 32
```

### Issue: NumPy version conflict

**Solution**: Downgrade NumPy
```bash
pip3 install "numpy<2.0"
```

### Issue: Permission denied on dataset directory

**Solution**: Create dataset directory with write permissions
```bash
mkdir -p dataset/cifar10
chmod 755 dataset
```

## Next Steps

✅ Installation complete!

Continue to:
- [Quickstart Guide](quickstart.md) - Run your first FL experiment
- [Configuration Guide](configuration.md) - Understand config.py settings

## Uninstallation

To remove the environment:

```bash
# Deactivate virtual environment
deactivate

# Remove environment
rm -rf env/

# Or for conda
conda deactivate
conda env remove -n cluster-fl
```

## Support

If you encounter issues:
1. Check [Troubleshooting](../reference/troubleshooting.md)
2. Search existing [GitHub Issues](https://github.com/Tanguyvans/cluster-shuffling-fl/issues)
3. Create a new issue with:
   - Error message
   - Python version (`python3 --version`)
   - Operating system
   - Installation method
