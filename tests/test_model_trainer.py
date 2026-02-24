"""
Unit tests for ModelTrainer class.

Tests the fine-tuning, training loop, and model saving functionality.
"""

import os
import pytest
import torch
from transformers import AutoModelForSequenceClassification

from ml_scanner.model_trainer import ModelTrainer
from ml_scanner.models import Dataset
from ml_scanner.exceptions import TrainingError


@pytest.fixture
def sample_dataset():
    """Create a small sample dataset for testing."""
    texts = [
        'api_key = "AKIAIOSFODNN7EXAMPLE"',
        'password = "super_secret_123"',
        'const token = "ghp_1234567890abcdefghijklmnopqrstuvwxyz"',
        'username = "john_doe"',
        'config = {"debug": true}',
        'email = "test@example.com"',
        'SSN = "123-45-6789"',
        'phone = "(555) 123-4567"',
    ]
    labels = [1, 1, 1, 0, 0, 1, 1, 1]  # 1 = secret/PII, 0 = not secret
    
    return Dataset(
        texts=texts,
        labels=labels,
        metadata={"test": True},
        source="test",
        version="1.0.0",
        checksum="test_checksum"
    )


@pytest.fixture
def model_trainer():
    """Create ModelTrainer instance."""
    return ModelTrainer(model_name="microsoft/codebert-base")


def test_model_trainer_initialization():
    """Test ModelTrainer initialization."""
    trainer = ModelTrainer(model_name="microsoft/codebert-base")
    
    assert trainer.model_name == "microsoft/codebert-base"
    assert trainer.tokenizer is not None
    assert trainer.device is not None


def test_model_trainer_device_selection():
    """Test device selection (CPU/CUDA)."""
    # Test explicit CPU
    trainer_cpu = ModelTrainer(device="cpu")
    assert trainer_cpu.device == torch.device("cpu")
    
    # Test auto-detection
    trainer_auto = ModelTrainer()
    assert trainer_auto.device in [torch.device("cpu"), torch.device("cuda")]


def test_prepare_dataloader(model_trainer, sample_dataset):
    """Test dataloader preparation."""
    dataloader = model_trainer._prepare_dataloader(
        sample_dataset,
        batch_size=2,
        shuffle=False
    )
    
    assert dataloader is not None
    assert len(dataloader) > 0
    
    # Check batch structure
    batch = next(iter(dataloader))
    assert len(batch) == 3  # input_ids, attention_mask, labels
    assert batch[0].shape[0] <= 2  # batch size


def test_fine_tune_small_model(model_trainer, sample_dataset, tmp_path):
    """Test fine-tuning with a very small model and dataset."""
    # Use a tiny model for testing
    model = AutoModelForSequenceClassification.from_pretrained(
        "microsoft/codebert-base",
        num_labels=2
    )
    
    # Split dataset into train and val
    train_size = 6
    train_dataset = Dataset(
        texts=sample_dataset.texts[:train_size],
        labels=sample_dataset.labels[:train_size],
        metadata=sample_dataset.metadata,
        source=sample_dataset.source,
        version=sample_dataset.version,
        checksum="train_checksum"
    )
    
    val_dataset = Dataset(
        texts=sample_dataset.texts[train_size:],
        labels=sample_dataset.labels[train_size:],
        metadata=sample_dataset.metadata,
        source=sample_dataset.source,
        version=sample_dataset.version,
        checksum="val_checksum"
    )
    
    # Fine-tune with minimal epochs
    trained_model = model_trainer.fine_tune(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        epochs=1,  # Just 1 epoch for testing
        learning_rate=2e-5,
        batch_size=2,
        warmup_steps=0
    )
    
    assert trained_model is not None
    assert isinstance(trained_model, torch.nn.Module)


def test_save_model(model_trainer, tmp_path):
    """Test model saving to registry."""
    # Create a simple model
    model = AutoModelForSequenceClassification.from_pretrained(
        "microsoft/codebert-base",
        num_labels=2
    )
    
    # Prepare metadata
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
            "learning_rate": 2e-5
        },
        "lineage": {
            "dataset_source": "github",
            "training_date": "2024-01-15"
        }
    }
    
    # Save model
    registry_path = str(tmp_path / "model_registry" / "models")
    model_dir = model_trainer.save_model(
        model=model,
        version="1.0.0",
        metadata=metadata,
        registry_path=registry_path
    )
    
    # Verify model was saved
    assert os.path.exists(model_dir)
    assert os.path.exists(os.path.join(model_dir, "config.json"))
    assert os.path.exists(os.path.join(model_dir, "pytorch_model.bin"))
    assert os.path.exists(os.path.join(model_dir, "metadata.json"))
    
    # Verify metadata content
    import json
    with open(os.path.join(model_dir, "metadata.json"), 'r') as f:
        saved_metadata = json.load(f)
    
    assert saved_metadata["version"] == "1.0.0"
    assert saved_metadata["dataset_version"] == "1.0.0"
    assert saved_metadata["environment"] == "STAGING"
    assert "training_date" in saved_metadata


def test_save_model_creates_directory(model_trainer, tmp_path):
    """Test that save_model creates directory if it doesn't exist."""
    model = AutoModelForSequenceClassification.from_pretrained(
        "microsoft/codebert-base",
        num_labels=2
    )
    
    metadata = {
        "dataset_version": "1.0.0",
        "performance_metrics": {},
        "environment": "STAGING",
        "training_config": {},
        "lineage": {}
    }
    
    # Use non-existent directory
    registry_path = str(tmp_path / "new_registry" / "models")
    model_dir = model_trainer.save_model(
        model=model,
        version="1.0.0",
        metadata=metadata,
        registry_path=registry_path
    )
    
    assert os.path.exists(model_dir)


def test_train_epoch_metrics(model_trainer, sample_dataset):
    """Test that training epoch returns valid metrics."""
    model = AutoModelForSequenceClassification.from_pretrained(
        "microsoft/codebert-base",
        num_labels=2
    )
    model = model.to(model_trainer.device)
    
    dataloader = model_trainer._prepare_dataloader(sample_dataset, batch_size=2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
    
    metrics = model_trainer._train_epoch(
        model=model,
        train_loader=dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=0
    )
    
    assert metrics is not None
    assert metrics.loss >= 0.0
    assert metrics.epoch == 0


def test_validate_epoch_metrics(model_trainer, sample_dataset):
    """Test that validation epoch returns valid metrics."""
    model = AutoModelForSequenceClassification.from_pretrained(
        "microsoft/codebert-base",
        num_labels=2
    )
    model = model.to(model_trainer.device)
    
    dataloader = model_trainer._prepare_dataloader(sample_dataset, batch_size=2)
    
    metrics = model_trainer._validate_epoch(
        model=model,
        val_loader=dataloader,
        epoch=0
    )
    
    assert metrics is not None
    assert metrics.loss >= 0.0
    assert 0.0 <= metrics.precision <= 1.0
    assert 0.0 <= metrics.recall <= 1.0
    assert 0.0 <= metrics.f1_score <= 1.0
    assert len(metrics.confusion_matrix) == 2
    assert len(metrics.confusion_matrix[0]) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
