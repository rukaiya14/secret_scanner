"""
PerformanceValidator for ML-Enhanced Secret Scanner.

This module handles evaluation of trained models against performance thresholds
to ensure they meet quality standards before deployment.
"""

import logging
from typing import Dict

import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import PreTrainedModel, AutoTokenizer
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
from tqdm import tqdm

from ml_scanner.models import Dataset, TrainingMetrics
from ml_scanner.exceptions import PerformanceThresholdError


logger = logging.getLogger(__name__)


class PerformanceValidator:
    """
    Evaluates trained models against performance thresholds.
    
    Validates model performance on test datasets and ensures models meet
    minimum quality standards (90% precision, 85% recall) before deployment.
    
    Validates: Requirements 1.6, 1.8
    """
    
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        device: str = None
    ):
        """
        Initialize PerformanceValidator.
        
        Args:
            tokenizer: Tokenizer for encoding test data
            device: Device to use for evaluation ('cpu', 'cuda', or None for auto-detect)
        """
        self.tokenizer = tokenizer
        
        # Auto-detect device if not specified
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        logger.info(f"Initialized PerformanceValidator with device={self.device}")
    
    def evaluate(
        self,
        model: PreTrainedModel,
        test_dataset: Dataset,
        batch_size: int = 16
    ) -> Dict[str, any]:
        """
        Evaluate model performance on test set.
        
        Computes precision, recall, F1-score, and confusion matrix for the model
        on the provided test dataset.
        
        Args:
            model: Trained model to evaluate
            test_dataset: Test dataset for evaluation
            batch_size: Batch size for evaluation (default 16)
            
        Returns:
            Dictionary with keys:
                - precision (float): Precision score
                - recall (float): Recall score
                - f1_score (float): F1 score
                - confusion_matrix (list): 2x2 confusion matrix as nested list
                
        Validates: Requirements 1.6, 1.8
        """
        logger.info(f"Evaluating model on test set with {len(test_dataset.texts)} examples")
        
        # Move model to device and set to evaluation mode
        model = model.to(self.device)
        model.eval()
        
        # Prepare test data loader
        test_loader = self._prepare_dataloader(test_dataset, batch_size)
        
        # Collect predictions and labels
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating"):
                # Move batch to device
                input_ids = batch[0].to(self.device)
                attention_mask = batch[1].to(self.device)
                labels = batch[2].to(self.device)
                
                # Forward pass
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                # Get predictions
                logits = outputs.logits
                predictions = torch.argmax(logits, dim=-1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        precision = precision_score(
            all_labels,
            all_predictions,
            average='binary',
            zero_division=0
        )
        recall = recall_score(
            all_labels,
            all_predictions,
            average='binary',
            zero_division=0
        )
        f1 = f1_score(
            all_labels,
            all_predictions,
            average='binary',
            zero_division=0
        )
        conf_matrix = confusion_matrix(all_labels, all_predictions).tolist()
        
        metrics = {
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "confusion_matrix": conf_matrix
        }
        
        logger.info(
            f"Evaluation complete - "
            f"Precision: {precision:.4f}, "
            f"Recall: {recall:.4f}, "
            f"F1: {f1:.4f}"
        )
        
        return metrics
    
    def meets_threshold(
        self,
        metrics: Dict[str, any],
        min_precision: float = 0.90,
        min_recall: float = 0.85
    ) -> bool:
        """
        Check if model meets minimum performance thresholds.
        
        Validates that the model achieves at least the minimum required
        precision and recall scores.
        
        Args:
            metrics: Dictionary containing 'precision' and 'recall' keys
            min_precision: Minimum required precision (default 0.90)
            min_recall: Minimum required recall (default 0.85)
            
        Returns:
            True if model meets both thresholds, False otherwise
            
        Validates: Requirements 1.6
        """
        precision = metrics.get("precision", 0.0)
        recall = metrics.get("recall", 0.0)
        
        meets_precision = precision >= min_precision
        meets_recall = recall >= min_recall
        
        if meets_precision and meets_recall:
            logger.info(
                f"Model meets thresholds - "
                f"Precision: {precision:.4f} >= {min_precision:.2f}, "
                f"Recall: {recall:.4f} >= {min_recall:.2f}"
            )
            return True
        else:
            logger.warning(
                f"Model does not meet thresholds - "
                f"Precision: {precision:.4f} {'✓' if meets_precision else '✗'} {min_precision:.2f}, "
                f"Recall: {recall:.4f} {'✓' if meets_recall else '✗'} {min_recall:.2f}"
            )
            return False
    
    def generate_report(
        self,
        metrics: Dict[str, any],
        model_version: str = "unknown",
        dataset_info: str = "unknown"
    ) -> str:
        """
        Generate a human-readable performance report.
        
        Creates a formatted report with all performance metrics including
        precision, recall, F1-score, and confusion matrix visualization.
        
        Args:
            metrics: Dictionary containing evaluation metrics
            model_version: Version string for the model
            dataset_info: Information about the test dataset
            
        Returns:
            Formatted performance report as string
            
        Validates: Requirements 1.8
        """
        precision = metrics.get("precision", 0.0)
        recall = metrics.get("recall", 0.0)
        f1 = metrics.get("f1_score", 0.0)
        conf_matrix = metrics.get("confusion_matrix", [[0, 0], [0, 0]])
        
        # Calculate additional metrics from confusion matrix
        tn, fp, fn, tp = 0, 0, 0, 0
        if len(conf_matrix) == 2 and len(conf_matrix[0]) == 2:
            tn, fp = conf_matrix[0]
            fn, tp = conf_matrix[1]
        
        total = tn + fp + fn + tp
        accuracy = (tp + tn) / total if total > 0 else 0.0
        
        report = f"""
╔══════════════════════════════════════════════════════════════╗
║              MODEL PERFORMANCE REPORT                        ║
╠══════════════════════════════════════════════════════════════╣
║ Model Version:    {model_version:<42} ║
║ Test Dataset:     {dataset_info:<42} ║
╠══════════════════════════════════════════════════════════════╣
║                    PERFORMANCE METRICS                       ║
╠══════════════════════════════════════════════════════════════╣
║ Precision:        {precision:>6.2%}  {'✓ PASS' if precision >= 0.90 else '✗ FAIL':<35} ║
║ Recall:           {recall:>6.2%}  {'✓ PASS' if recall >= 0.85 else '✗ FAIL':<35} ║
║ F1-Score:         {f1:>6.2%}                                    ║
║ Accuracy:         {accuracy:>6.2%}                                    ║
╠══════════════════════════════════════════════════════════════╣
║                    CONFUSION MATRIX                          ║
╠══════════════════════════════════════════════════════════════╣
║                          Predicted                           ║
║                    Negative    Positive                      ║
║  Actual  Negative    {tn:>5}       {fp:>5}                        ║
║          Positive    {fn:>5}       {tp:>5}                        ║
╠══════════════════════════════════════════════════════════════╣
║ True Negatives:   {tn:>5}                                      ║
║ False Positives:  {fp:>5}                                      ║
║ False Negatives:  {fn:>5}                                      ║
║ True Positives:   {tp:>5}                                      ║
║ Total Samples:    {total:>5}                                      ║
╚══════════════════════════════════════════════════════════════╝
"""
        
        logger.info(f"Generated performance report for model {model_version}")
        return report
    
    def _prepare_dataloader(
        self,
        dataset: Dataset,
        batch_size: int
    ) -> DataLoader:
        """
        Prepare PyTorch DataLoader from Dataset.
        
        Args:
            dataset: Dataset to prepare
            batch_size: Batch size
            
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
        
        # Create dataloader (no shuffling for evaluation)
        dataloader = DataLoader(
            tensor_dataset,
            batch_size=batch_size,
            shuffle=False
        )
        
        return dataloader
