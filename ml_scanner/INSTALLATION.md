# Installation Guide

## Prerequisites

- Python 3.8 or higher
- pip package manager

## Installation Steps

### 1. Install PyTorch

PyTorch installation varies by platform. Visit https://pytorch.org/get-started/locally/ to get the correct installation command for your system.

**For CPU-only (Windows/Linux/Mac):**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

**For CUDA 11.8 (Windows/Linux):**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**For CUDA 12.1 (Windows/Linux):**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**For Mac (Apple Silicon):**
```bash
pip install torch torchvision torchaudio
```

### 2. Install Other Dependencies

After installing PyTorch, install the remaining dependencies:

```bash
pip install transformers tokenizers shap lime pyyaml requests numpy prometheus-client hypothesis pytest pytest-cov
```

Or install from the main requirements.txt:

```bash
pip install -r requirements.txt
```

### 3. Verify Installation

Run the verification script:

```bash
python -c "import ml_scanner; print('ML Scanner installed successfully')"
```

## Troubleshooting

### PyTorch Installation Issues

If you encounter issues installing PyTorch:

1. Check your Python version: `python --version`
2. Ensure pip is up to date: `pip install --upgrade pip`
3. Visit https://pytorch.org/get-started/locally/ for platform-specific instructions
4. For Windows, you may need to install Visual C++ Redistributable

### Import Errors

If you get import errors:

1. Ensure you're in the correct Python environment
2. Verify all dependencies are installed: `pip list`
3. Check for version conflicts: `pip check`

### Memory Issues

If you encounter memory issues during installation:

1. Install packages one at a time
2. Use `--no-cache-dir` flag: `pip install --no-cache-dir <package>`
3. Close other applications to free up memory

## Development Installation

For development, install additional dependencies:

```bash
pip install pytest pytest-cov hypothesis black flake8 mypy
```

## Docker Installation (Recommended for Production)

For production deployments, we recommend using Docker:

```bash
docker build -t ml-scanner .
docker run -v $(pwd)/model_registry:/app/model_registry ml-scanner
```

See the main project documentation for Docker setup instructions.
