# PerformanceValidator Guide

## Overview

The `PerformanceValidator` class evaluates trained CodeBERT models against performance thresholds to ensure they meet quality standards before deployment. It computes precision, recall, F1-score, and confusion matrix metrics, and validates that models achieve minimum thresholds (90% precision, 85% recall).

## Requirements Validated

- **Requirement 1.6**: Model performance validation with minimum 90% precision and 85% recall
- **Requirement 1.8**: Performance report generation with precision, recall, F1-score, and confusion matrix

## Usage

### Basic Usage

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from ml_scanner.performance_validator import PerformanceValidator
from ml_scanner.models import Dataset

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")

# Create validator
validator = PerformanceValidator(tokenizer, device="cuda")  # or "cpu"

# Load your trained model
model = AutoModelForSequenceClassification.from_pretrained("path/to/model")

# Prepare test dataset
test_dataset = Dataset(
    texts=["api_key = 'AKIAIOSFODNN7EXAMPLE'", "username = 'john'"],
    labels=[1, 0],  # 1 = secret, 0 = not secret
    metadata={"source": "test"},
    source="test",
    version="1.0.0",
    checksum="abc123"
)

# Evaluate model
metrics = validator.evaluate(model, test_dataset, batch_size=16)

print(f"Precision: {metrics['precision']:.4f}")
print(f"Recall: {metrics['recall']:.4f}")
print(f"F1-Score: {metrics['f1_score']:.4f}")
print(f"Confusion Matrix: {metrics['confusion_matrix']}")
```

### Checking Performance Thresholds

```python
# Check if model meets default thresholds (90% precision, 85% recall)
if validator.meets_threshold(metrics):
    print("✓ Model meets quality thresholds")
else:
    print("✗ Model does not meet quality thresholds")

# Check with custom thresholds
if validator.meets_threshold(metrics, min_precision=0.85, min_recall=0.80):
    print("✓ Model meets custom thresholds")
```

### Generating Performance Reports

```python
# Generate a formatted performance report
report = validator.generate_report(
    metrics,
    model_version="1.2.3",
    dataset_info="GitHub secrets test set (1000 examples)"
)

print(report)
```

Example output:

```
╔══════════════════════════════════════════════════════════════╗
║              MODEL PERFORMANCE REPORT                        ║
╠══════════════════════════════════════════════════════════════╣
║ Model Version:    1.2.3                                      ║
║ Test Dataset:     GitHub secrets test set (1000 examples)   ║
╠══════════════════════════════════════════════════════════════╣
║                    PERFORMANCE METRICS                       ║
╠══════════════════════════════════════════════════════════════╣
║ Precision:        95.00%  ✓ PASS                            ║
║ Recall:           90.00%  ✓ PASS                            ║
║ F1-Score:         92.44%                                     ║
║ Accuracy:         92.50%                                     ║
╠══════════════════════════════════════════════════════════════╣
║                    CONFUSION MATRIX                          ║
╠══════════════════════════════════════════════════════════════╣
║                          Predicted                           ║
║                    Negative    Positive                      ║
║  Actual  Negative       80          5                        ║
║          Positive       10        105                        ║
╠══════════════════════════════════════════════════════════════╣
║ True Negatives:      80                                      ║
║ False Positives:      5                                      ║
║ False Negatives:     10                                      ║
║ True Positives:     105                                      ║
║ Total Samples:      200                                      ║
╚══════════════════════════════════════════════════════════════╝
```

## Integration with Training Pipeline

The `PerformanceValidator` is designed to work seamlessly with the `ModelTrainer`:

```python
from ml_scanner.model_trainer import ModelTrainer
from ml_scanner.performance_validator import PerformanceValidator
from ml_scanner.dataset_loader import DatasetLoader
from ml_scanner.data_preprocessor import DataPreprocessor

# Load and prepare datasets
loader = DatasetLoader()
dataset = loader.load_github_datasets(["https://github.com/dxa4481/truffleHogRegexes"])

preprocessor = DataPreprocessor()
train_dataset, val_dataset, test_dataset = preprocessor.split_dataset(dataset)

# Train model
trainer = ModelTrainer()
model = trainer.fine_tune(model, train_dataset, val_dataset, epochs=3)

# Validate performance
validator = PerformanceValidator(trainer.tokenizer)
metrics = validator.evaluate(model, test_dataset)

# Check thresholds before saving
if validator.meets_threshold(metrics):
    # Generate and log report
    report = validator.generate_report(metrics, model_version="1.0.0")
    print(report)
    
    # Save model
    trainer.save_model(
        model,
        version="1.0.0",
        metadata={
            "performance_metrics": metrics,
            "dataset_version": "1.0.0"
        }
    )
else:
    print("Model does not meet quality thresholds. Retraining required.")
```

## API Reference

### `PerformanceValidator.__init__(tokenizer, device=None)`

Initialize the performance validator.

**Parameters:**
- `tokenizer` (AutoTokenizer): Tokenizer for encoding test data
- `device` (str, optional): Device to use ('cpu', 'cuda', or None for auto-detect)

### `PerformanceValidator.evaluate(model, test_dataset, batch_size=16)`

Evaluate model performance on test set.

**Parameters:**
- `model` (PreTrainedModel): Trained model to evaluate
- `test_dataset` (Dataset): Test dataset for evaluation
- `batch_size` (int): Batch size for evaluation (default 16)

**Returns:**
- `dict`: Dictionary with keys:
  - `precision` (float): Precision score (0.0-1.0)
  - `recall` (float): Recall score (0.0-1.0)
  - `f1_score` (float): F1 score (0.0-1.0)
  - `confusion_matrix` (list): 2x2 confusion matrix as nested list

### `PerformanceValidator.meets_threshold(metrics, min_precision=0.90, min_recall=0.85)`

Check if model meets minimum performance thresholds.

**Parameters:**
- `metrics` (dict): Dictionary containing 'precision' and 'recall' keys
- `min_precision` (float): Minimum required precision (default 0.90)
- `min_recall` (float): Minimum required recall (default 0.85)

**Returns:**
- `bool`: True if model meets both thresholds, False otherwise

### `PerformanceValidator.generate_report(metrics, model_version="unknown", dataset_info="unknown")`

Generate a human-readable performance report.

**Parameters:**
- `metrics` (dict): Dictionary containing evaluation metrics
- `model_version` (str): Version string for the model
- `dataset_info` (str): Information about the test dataset

**Returns:**
- `str`: Formatted performance report

## Performance Thresholds

The default performance thresholds are:
- **Precision**: ≥ 90% (minimize false positives)
- **Recall**: ≥ 85% (minimize false negatives)

These thresholds ensure that:
1. At least 90% of detected secrets are true positives (high precision)
2. At least 85% of actual secrets are detected (high recall)

You can customize these thresholds based on your organization's requirements using the `min_precision` and `min_recall` parameters in `meets_threshold()`.

## Device Selection

The validator automatically detects and uses GPU if available:

```python
# Auto-detect (uses GPU if available)
validator = PerformanceValidator(tokenizer)

# Force CPU
validator = PerformanceValidator(tokenizer, device="cpu")

# Force GPU
validator = PerformanceValidator(tokenizer, device="cuda")
```

## Error Handling

The validator handles common error scenarios:

- **Empty datasets**: Returns zero metrics without crashing
- **Missing metric keys**: Defaults to 0.0 for missing values
- **Zero confusion matrix**: Handles division by zero gracefully

## Best Practices

1. **Always validate before deployment**: Use `meets_threshold()` to ensure models meet quality standards
2. **Generate reports for audit**: Save performance reports for model versioning and compliance
3. **Use consistent test sets**: Maintain a fixed test set for comparing model versions
4. **Monitor performance over time**: Track metrics across model versions to detect degradation
5. **Adjust thresholds based on use case**: Balance precision and recall based on your security requirements

## See Also

- [ModelTrainer Guide](MODEL_TRAINER_GUIDE.md) - Training CodeBERT models
- [Design Document](../.kiro/specs/ml-codebert-secret-detection/design.md) - System architecture
- [Requirements Document](../.kiro/specs/ml-codebert-secret-detection/requirements.md) - Detailed requirements
