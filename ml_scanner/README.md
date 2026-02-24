# ML-Enhanced Secret Scanner

Machine learning-based secret and PII detection using CodeBERT transformers, complementing the existing regex-based pattern matching system.

## Overview

This package provides:

- **Hybrid Detection**: Combines regex-based and ML-based detection for higher accuracy
- **Explainability**: Interpretable explanations for ML detections using SHAP/LIME
- **Real-Time Performance**: Sub-10-second scan times for typical commits
- **Resilience**: Gracefully degrades to regex-only mode when ML components fail
- **Continuous Improvement**: Supports automated retraining and model versioning

## Project Structure

```
ml_scanner/
├── __init__.py           # Package initialization
├── models.py             # Core data models
├── exceptions.py         # Error codes and exceptions
├── config.py             # Configuration management
├── logger.py             # Structured JSON logging
├── requirements.txt      # Python dependencies
├── config.example.yaml   # Example configuration file
└── README.md            # This file
```

## Installation

1. Install dependencies:

```bash
pip install -r ml_scanner/requirements.txt
```

2. Create configuration file:

```bash
cp ml_scanner/config.example.yaml ml_scanner/config.yaml
# Edit config.yaml with your settings
```

## Configuration

The system uses YAML or JSON configuration files. Key configuration options:

- **Confidence Thresholds**: Control when to block vs. warn on detections
- **Timeouts**: Set maximum execution times for various operations
- **Detection Categories**: Enable/disable specific secret types
- **Explainer Backend**: Choose between SHAP or LIME for explanations
- **Device**: Select CPU, GPU, or auto-detection for inference

See `config.example.yaml` for full configuration options.

## Core Data Models

### Finding
Unified detection format for both regex and ML-based detections.

### Detection
Raw ML model output before conversion to Finding format.

### Explanation
Interpretability information showing which tokens influenced detection.

### ModelMetadata
Version, performance metrics, and lineage for trained models.

### ScanResult
Aggregated results from scanning operations.

### Dataset
Training dataset with metadata and provenance tracking.

## Error Handling

The system uses standardized error codes for consistent error handling:

- **1000-1999**: Training Pipeline Errors
- **2000-2999**: Inference Engine Errors
- **3000-3999**: Model Registry Errors
- **4000-4999**: Explainer Errors
- **5000-5999**: Alert Manager Errors
- **6000-6999**: Git Hook Errors
- **7000-7999**: Configuration Errors

All exceptions inherit from `MLScannerError` and include error codes and context.

## Logging

The system uses structured JSON logging for better observability:

```python
from ml_scanner.logger import get_logger

logger = get_logger(__name__)
logger.info("Operation completed", context={"files_scanned": 10})
logger.error("Operation failed", error_code=2001, context={"reason": "timeout"})
```

Log format:
```json
{
  "timestamp": "2024-01-15T10:30:45.123Z",
  "level": "ERROR",
  "component": "ml_scanner.inference",
  "message": "Failed to load model",
  "error_code": 2001,
  "context": {"model_version": "1.2.3"},
  "trace_id": "abc123"
}
```

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=ml_scanner

# Run property tests
pytest tests/property/
```

### Code Style

Follow PEP 8 guidelines. Use type hints for all function signatures.

## License

See main project LICENSE file.
