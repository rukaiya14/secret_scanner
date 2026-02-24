# ModelTrainer Implementation Guide

## Overview

The `ModelTrainer` class handles fine-tuning of CodeBERT models for secret detection. It implements the complete training workflow including model initialization, training loop execution with validation, and model persistence with metadata.

## Features

### 1. Fine-Tuning
- **Method**: `fine_tune(model, train_dataset, val_dataset, epochs=3, learning_rate=2e-5, ...)`
- Fine-tunes pre-trained CodeBERT models on secret detection datasets
- Default configuration: 3 epochs, learning rate 2e-5 (as per requirements)
- Supports both CPU and GPU training with automatic device detection
- Implements AdamW optimizer with linear warmup scheduler

### 2. Training Loop
- Executes training epochs with progress tracking
- Performs validation after each epoch
- Calculates comprehensive metrics: precision, recall, F1-score, confusion matrix
- Tracks best model based on validation loss

### 3. Model Persistence
- **Method**: `save_model(model, version, metadata, registry_path)`
- Saves trained models to Model Registry with semantic versioning
- Stores model weights, tokenizer, and comprehensive metadata
- Metadata includes: training date, dataset version, performance metrics, environment, training config, and lineage

## Usage Example

```python
from ml_scanner.model_trainer import ModelTrainer
from ml_scanner.models import Dataset
from transformers import AutoModelForSequenceClassification

# Initialize trainer
trainer = ModelTrainer(model_name="microsoft/codebert-base")

# Load pre-trained model
model = AutoModelForSequenceClassification.from_pretrained(
    "microsoft/codebert-base",
    num_labels=2
)

# Prepare datasets (train and validation)
train_dataset = Dataset(
    texts=["api_key = 'AKIA...'", "username = 'john'"],
    labels=[1, 0],
    metadata={},
    source="github",
    version="1.0.0",
    checksum="abc123"
)

val_dataset = Dataset(
    texts=["password = 'secret'", "config = {}"],
    labels=[1, 0],
    metadata={},
    source="github",
    version="1.0.0",
    checksum="def456"
)

# Fine-tune model
trained_model = trainer.fine_tune(
    model=model,
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    epochs=3,
    learning_rate=2e-5,
    batch_size=16,
    warmup_steps=500
)

# Save model to registry
metadata = {
    "dataset_version": "1.0.0",
    "performance_metrics": {
        "precision": 0.92,
        "recall": 0.88,
        "f1_score": 0.90
    },
    "environment": "STAGING",
    "training_config": {
        "epochs": 3,
        "learning_rate": 2e-5,
        "batch_size": 16
    },
    "lineage": {
        "dataset_source": "github",
        "training_date": "2024-01-15"
    }
}

model_path = trainer.save_model(
    model=trained_model,
    version="1.0.0",
    metadata=metadata,
    registry_path="model_registry/models"
)

print(f"Model saved to: {model_path}")
```

## Requirements Validation

This implementation validates the following requirements:

- **Requirement 1.5**: Fine-tunes CodeBERT model on training dataset with 3 epochs and learning rate 2e-5
- **Requirement 1.7**: Saves trained model artifacts to Model Registry with version metadata
- **Requirement 3.2**: Loads CodeBERT model from Model Registry for inference

## Architecture

### Training Flow
1. Initialize model and move to appropriate device (CPU/GPU)
2. Prepare PyTorch DataLoaders from Dataset objects
3. Setup AdamW optimizer and linear warmup scheduler
4. Execute training loop:
   - Forward pass through model
   - Calculate loss
   - Backward pass and parameter updates
   - Learning rate scheduling
5. Validate after each epoch:
   - Calculate precision, recall, F1-score
   - Generate confusion matrix
   - Track best model checkpoint
6. Return fine-tuned model

### Model Persistence
1. Create version directory in model registry
2. Save model weights using `save_pretrained()`
3. Save tokenizer configuration
4. Create ModelMetadata object with all tracking information
5. Serialize metadata to JSON
6. Return path to saved model

## Dependencies

- `torch`: PyTorch for model training
- `transformers`: Hugging Face Transformers for CodeBERT
- `scikit-learn`: Metrics calculation (precision, recall, F1, confusion matrix)
- `tqdm`: Progress bars for training visualization

## Error Handling

The ModelTrainer includes comprehensive error handling:

- **TrainingError**: Raised when training fails (OOM, convergence issues, etc.)
- All errors include context information for debugging
- Graceful handling of device selection (CPU fallback if GPU unavailable)

## Testing

Unit tests are provided in `tests/test_model_trainer.py`:

- Initialization tests
- Device selection tests
- DataLoader preparation tests
- Fine-tuning tests (with small model/dataset)
- Model saving tests
- Metrics calculation tests

Run tests with:
```bash
pytest tests/test_model_trainer.py -v
```

## Future Enhancements

Potential improvements for future iterations:

1. **Checkpoint Management**: Save intermediate checkpoints during training
2. **Early Stopping**: Stop training if validation loss doesn't improve
3. **Mixed Precision Training**: Use FP16 for faster training on compatible GPUs
4. **Distributed Training**: Support multi-GPU training
5. **Hyperparameter Tuning**: Integrate with hyperparameter optimization frameworks
6. **Model Quantization**: Support INT8 quantization for deployment
7. **Training Resumption**: Resume training from saved checkpoints

## Notes

- The implementation uses a simple training loop suitable for the secret detection task
- For production use, consider adding more sophisticated training strategies
- Model registry path defaults to `model_registry/models` but can be customized
- All models are saved with semantic versioning (MAJOR.MINOR.PATCH format)
