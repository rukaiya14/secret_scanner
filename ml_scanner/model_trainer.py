"""
ModelTrainer for ML-Enhanced Secret Scanner.

This module handles fine-tuning of CodeBERT models for secret detection,
including training loop execution, validation, and model persistence.
"""

import logging
import os
from typing import Optional, Dict
from datetime import datetime

import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    AdamW,
    get_linear_schedule_with_warmup,
    PreTrainedModel
)
from tqdm import tqdm

from ml_scanner.models import Dataset, TrainingMetrics, ModelMetadata
from ml_scanner.exceptions import TrainingError


logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Fine-tunes CodeBERT model on secret detection datasets.
    
    Handles the complete training workflow including:
    - Model initialization and fine-tuning
    - Training loop with validation
    - Model persistence with metadata
    
    Validates: Requirements 1.5, 1.7, 3.2
    """
    
    def __init__(
        self,
        model_name: str = "microsoft/codebert-base",
        device: Optional[str] = None
    ):
        """
        Initialize ModelTrainer.
        
        Args:
            model_name: Name of the pre-trained CodeBERT model
            device: Device to use for training ('cpu', 'cuda', or None for auto-detect)
        """
        self.model_name = model_name
        
        # Auto-detect device if not specified
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        logger.info(f"Initialized ModelTrainer with model={model_name}, device={self.device}")
        
        # Initialize tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        except Exception as e:
            error_msg = f"Failed to load tokenizer {model_name}: {str(e)}"
            logger.error(error_msg)
            raise TrainingError(error_msg, {"model_name": model_name}) from e
    
    def fine_tune(
        self,
        model: PreTrainedModel,
        train_dataset: Dataset,
        val_dataset: Dataset,
        epochs: int = 3,
        learning_rate: float = 2e-5,
        batch_size: int = 16,
        warmup_steps: int = 500
    ) -> PreTrainedModel:
        """
        Fine-tune CodeBERT model on training dataset.
        
        Args:
            model: Pre-trained model to fine-tune
            train_dataset: Training dataset
            val_dataset: Validation dataset
            epochs: Number of training epochs (default 3)
            learning_rate: Learning rate (default 2e-5)
            batch_size: Batch size for training (default 16)
            warmup_steps: Number of warmup steps for learning rate scheduler
            
        Returns:
            Fine-tuned model
            
        Raises:
            TrainingError: If training fails
            
        Validates: Requirements 1.5
        """
        logger.info(
            f"Starting fine-tuning: epochs={epochs}, lr={learning_rate}, "
            f"batch_size={batch_size}, warmup_steps={warmup_steps}"
        )
        
        try:
            # Move model to device
            model = model.to(self.device)
            
            # Prepare data loaders
            train_loader = self._prepare_dataloader(train_dataset, batch_size, shuffle=True)
            val_loader = self._prepare_dataloader(val_dataset, batch_size, shuffle=False)
            
            # Setup optimizer and scheduler
            optimizer = AdamW(model.parameters(), lr=learning_rate)
            
            total_steps = len(train_loader) * epochs
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps
            )
            
            # Training loop
            best_val_loss = float('inf')
            training_history = []
            
            for epoch in range(epochs):
                logger.info(f"Epoch {epoch + 1}/{epochs}")
                
                # Training phase
                train_metrics = self._train_epoch(
                    model, train_loader, optimizer, scheduler, epoch
                )
                
                # Validation phase
                val_metrics = self._validate_epoch(model, val_loader, epoch)
                
                # Log metrics
                logger.info(
                    f"Epoch {epoch + 1} - "
                    f"Train Loss: {train_metrics.loss:.4f}, "
                    f"Val Loss: {val_metrics.loss:.4f}, "
                    f"Val Precision: {val_metrics.precision:.4f}, "
                    f"Val Recall: {val_metrics.recall:.4f}, "
                    f"Val F1: {val_metrics.f1_score:.4f}"
                )
                
                training_history.append({
                    "epoch": epoch + 1,
                    "train_loss": train_metrics.loss,
                    "val_loss": val_metrics.loss,
                    "val_precision": val_metrics.precision,
                    "val_recall": val_metrics.recall,
                    "val_f1": val_metrics.f1_score
                })
                
                # Save best model checkpoint
                if val_metrics.loss < best_val_loss:
                    best_val_loss = val_metrics.loss
                    logger.info(f"New best validation loss: {best_val_loss:.4f}")
            
            logger.info("Fine-tuning completed successfully")
            return model
            
        except Exception as e:
            error_msg = f"Fine-tuning failed: {str(e)}"
            logger.error(error_msg)
            raise TrainingError(error_msg, {
                "epochs": epochs,
                "learning_rate": learning_rate,
                "batch_size": batch_size
            }) from e
    
    def save_model(
        self,
        model: PreTrainedModel,
        version: str,
        metadata: Dict,
        registry_path: str = "model_registry/models"
    ) -> str:
        """
        Save trained model to Model Registry with metadata.
        
        Args:
            model: Trained model to save
            version: Semantic version string (e.g., "1.0.0")
            metadata: Model metadata dictionary
            registry_path: Base path for model registry
            
        Returns:
            Path to saved model directory
            
        Raises:
            TrainingError: If model saving fails
            
        Validates: Requirements 1.7, 3.2
        """
        logger.info(f"Saving model version {version} to registry")
        
        try:
            # Create version directory
            model_dir = os.path.join(registry_path, version)
            os.makedirs(model_dir, exist_ok=True)
            
            # Save model and tokenizer
            model.save_pretrained(model_dir)
            self.tokenizer.save_pretrained(model_dir)
            
            # Create ModelMetadata object
            model_metadata = ModelMetadata(
                version=version,
                training_date=datetime.now(),
                dataset_version=metadata.get("dataset_version", "unknown"),
                performance_metrics=metadata.get("performance_metrics", {}),
                environment=metadata.get("environment", "STAGING"),
                training_config=metadata.get("training_config", {}),
                lineage=metadata.get("lineage", {})
            )
            
            # Save metadata as JSON
            import json
            metadata_path = os.path.join(model_dir, "metadata.json")
            with open(metadata_path, 'w') as f:
                json.dump({
                    "version": model_metadata.version,
                    "training_date": model_metadata.training_date.isoformat(),
                    "dataset_version": model_metadata.dataset_version,
                    "performance_metrics": model_metadata.performance_metrics,
                    "environment": model_metadata.environment,
                    "training_config": model_metadata.training_config,
                    "lineage": model_metadata.lineage
                }, f, indent=2)
            
            logger.info(f"Model saved successfully to {model_dir}")
            return model_dir
            
        except Exception as e:
            error_msg = f"Failed to save model: {str(e)}"
            logger.error(error_msg)
            raise TrainingError(error_msg, {
                "version": version,
                "registry_path": registry_path
            }) from e
    
    def _prepare_dataloader(
        self,
        dataset: Dataset,
        batch_size: int,
        shuffle: bool = True
    ) -> DataLoader:
        """
        Prepare PyTorch DataLoader from Dataset.
        
        Args:
            dataset: Dataset to prepare
            batch_size: Batch size
            shuffle: Whether to shuffle data
            
        Returns:
            DataLoader for the dataset
        """
        # Tokenize texts
        encodings = self.tokenizer(
            dataset.texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        
        # Create tensor dataset
        labels = torch.tensor(dataset.labels, dtype=torch.long)
        tensor_dataset = TensorDataset(
            encodings['input_ids'],
            encodings['attention_mask'],
            labels
        )
        
        # Create dataloader
        dataloader = DataLoader(
            tensor_dataset,
            batch_size=batch_size,
            shuffle=shuffle
        )
        
        return dataloader
    
    def _train_epoch(
        self,
        model: PreTrainedModel,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LambdaLR,
        epoch: int
    ) -> TrainingMetrics:
        """
        Execute one training epoch.
        
        Args:
            model: Model to train
            train_loader: Training data loader
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            epoch: Current epoch number
            
        Returns:
            Training metrics for the epoch
        """
        model.train()
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch + 1}")
        
        for batch in progress_bar:
            # Move batch to device
            input_ids = batch[0].to(self.device)
            attention_mask = batch[1].to(self.device)
            labels = batch[2].to(self.device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            total_loss += loss.item()
            num_batches += 1
            
            # Backward pass
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            # Update progress bar
            progress_bar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        return TrainingMetrics(
            precision=0.0,  # Not calculated during training
            recall=0.0,
            f1_score=0.0,
            confusion_matrix=[[0, 0], [0, 0]],
            loss=avg_loss,
            epoch=epoch
        )
    
    def _validate_epoch(
        self,
        model: PreTrainedModel,
        val_loader: DataLoader,
        epoch: int
    ) -> TrainingMetrics:
        """
        Execute validation on validation set.
        
        Args:
            model: Model to validate
            val_loader: Validation data loader
            epoch: Current epoch number
            
        Returns:
            Validation metrics
        """
        model.eval()
        total_loss = 0.0
        num_batches = 0
        
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Validation Epoch {epoch + 1}"):
                # Move batch to device
                input_ids = batch[0].to(self.device)
                attention_mask = batch[1].to(self.device)
                labels = batch[2].to(self.device)
                
                # Forward pass
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                total_loss += loss.item()
                num_batches += 1
                
                # Get predictions
                logits = outputs.logits
                predictions = torch.argmax(logits, dim=-1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        # Calculate metrics
        from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
        
        precision = precision_score(all_labels, all_predictions, average='binary', zero_division=0)
        recall = recall_score(all_labels, all_predictions, average='binary', zero_division=0)
        f1 = f1_score(all_labels, all_predictions, average='binary', zero_division=0)
        conf_matrix = confusion_matrix(all_labels, all_predictions).tolist()
        
        return TrainingMetrics(
            precision=precision,
            recall=recall,
            f1_score=f1,
            confusion_matrix=conf_matrix,
            loss=avg_loss,
            epoch=epoch
        )
