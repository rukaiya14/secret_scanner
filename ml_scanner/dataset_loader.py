"""
DatasetLoader for ML-Enhanced Secret Scanner.

This module handles downloading and validating datasets from GitHub and Kaggle
for training the CodeBERT model.
"""

import hashlib
import logging
import os
import requests
from typing import List, Tuple
from urllib.parse import urlparse

from ml_scanner.models import Dataset
from ml_scanner.exceptions import DatasetLoadError


logger = logging.getLogger(__name__)


class DatasetLoader:
    """
    Loads and validates datasets from GitHub and Kaggle sources.
    
    Handles dataset downloading, checksum validation, and splitting
    into train/validation/test sets.
    """
    
    def __init__(self, cache_dir: str = ".cache/datasets"):
        """
        Initialize DatasetLoader.
        
        Args:
            cache_dir: Directory to cache downloaded datasets
        """
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    
    def load_github_datasets(self, repo_urls: List[str]) -> Dataset:
        """
        Load labeled secret examples from GitHub repositories.
        
        Args:
            repo_urls: List of GitHub repository URLs containing datasets
            
        Returns:
            Dataset with combined examples from all repositories
            
        Raises:
            DatasetLoadError: If dataset loading fails
            
        Validates: Requirements 1.1, 2.1
        """
        all_texts = []
        all_labels = []
        metadata = {
            "sources": repo_urls,
            "num_repos": len(repo_urls)
        }
        
        for repo_url in repo_urls:
            try:
                logger.info(f"Loading GitHub dataset from {repo_url}")
                texts, labels = self._download_github_dataset(repo_url)
                all_texts.extend(texts)
                all_labels.extend(labels)
                logger.info(f"Loaded {len(texts)} examples from {repo_url}")
            except Exception as e:
                error_msg = f"Failed to load dataset from {repo_url}: {str(e)}"
                logger.error(error_msg)
                raise DatasetLoadError(error_msg) from e
        
        if not all_texts:
            raise DatasetLoadError("No data loaded from GitHub repositories")
        
        # Calculate checksum for the combined dataset
        checksum = self._calculate_checksum(all_texts, all_labels)
        
        return Dataset(
            texts=all_texts,
            labels=all_labels,
            metadata=metadata,
            source="github",
            version="1.0.0",
            checksum=checksum
        )
    
    def load_kaggle_datasets(self, dataset_ids: List[str]) -> Dataset:
        """
        Load labeled PII examples from Kaggle.
        
        Args:
            dataset_ids: List of Kaggle dataset identifiers
            
        Returns:
            Dataset with combined examples from all Kaggle datasets
            
        Raises:
            DatasetLoadError: If dataset loading fails
            
        Validates: Requirements 1.2, 2.2
        """
        all_texts = []
        all_labels = []
        metadata = {
            "sources": dataset_ids,
            "num_datasets": len(dataset_ids)
        }
        
        for dataset_id in dataset_ids:
            try:
                logger.info(f"Loading Kaggle dataset {dataset_id}")
                texts, labels = self._download_kaggle_dataset(dataset_id)
                all_texts.extend(texts)
                all_labels.extend(labels)
                logger.info(f"Loaded {len(texts)} examples from {dataset_id}")
            except Exception as e:
                error_msg = f"Failed to load Kaggle dataset {dataset_id}: {str(e)}"
                logger.error(error_msg)
                raise DatasetLoadError(error_msg) from e
        
        if not all_texts:
            raise DatasetLoadError("No data loaded from Kaggle datasets")
        
        # Calculate checksum for the combined dataset
        checksum = self._calculate_checksum(all_texts, all_labels)
        
        return Dataset(
            texts=all_texts,
            labels=all_labels,
            metadata=metadata,
            source="kaggle",
            version="1.0.0",
            checksum=checksum
        )
    
    def validate_checksums(self, dataset: Dataset) -> bool:
        """
        Verify dataset integrity with checksums.
        
        Args:
            dataset: Dataset to validate
            
        Returns:
            True if checksum is valid, False otherwise
            
        Validates: Requirements 2.3
        """
        calculated_checksum = self._calculate_checksum(dataset.texts, dataset.labels)
        is_valid = calculated_checksum == dataset.checksum
        
        if is_valid:
            logger.info(f"Dataset checksum validation passed: {calculated_checksum}")
        else:
            logger.warning(
                f"Dataset checksum validation failed. "
                f"Expected: {dataset.checksum}, Got: {calculated_checksum}"
            )
        
        return is_valid
    
    def split_dataset(
        self,
        dataset: Dataset,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15
    ) -> Tuple[Dataset, Dataset, Dataset]:
        """
        Split dataset into training, validation, and test sets.
        
        Args:
            dataset: Dataset to split
            train_ratio: Proportion for training set (default 0.7)
            val_ratio: Proportion for validation set (default 0.15)
            test_ratio: Proportion for test set (default 0.15)
            
        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
            
        Raises:
            ValueError: If ratios don't sum to 1.0
            
        Validates: Requirements 2.4
        """
        # Validate ratios
        total_ratio = train_ratio + val_ratio + test_ratio
        if abs(total_ratio - 1.0) > 0.001:
            raise ValueError(
                f"Ratios must sum to 1.0, got {total_ratio} "
                f"(train={train_ratio}, val={val_ratio}, test={test_ratio})"
            )
        
        total_size = len(dataset.texts)
        train_size = int(total_size * train_ratio)
        val_size = int(total_size * val_ratio)
        # test_size gets the remainder to ensure all examples are used
        
        logger.info(
            f"Splitting dataset of size {total_size} into "
            f"train={train_size}, val={val_size}, test={total_size - train_size - val_size}"
        )
        
        # Split the data
        train_texts = dataset.texts[:train_size]
        train_labels = dataset.labels[:train_size]
        
        val_texts = dataset.texts[train_size:train_size + val_size]
        val_labels = dataset.labels[train_size:train_size + val_size]
        
        test_texts = dataset.texts[train_size + val_size:]
        test_labels = dataset.labels[train_size + val_size:]
        
        # Create metadata for each split
        train_metadata = {**dataset.metadata, "split": "train", "size": len(train_texts)}
        val_metadata = {**dataset.metadata, "split": "validation", "size": len(val_texts)}
        test_metadata = {**dataset.metadata, "split": "test", "size": len(test_texts)}
        
        # Calculate checksums for each split
        train_checksum = self._calculate_checksum(train_texts, train_labels)
        val_checksum = self._calculate_checksum(val_texts, val_labels)
        test_checksum = self._calculate_checksum(test_texts, test_labels)
        
        train_dataset = Dataset(
            texts=train_texts,
            labels=train_labels,
            metadata=train_metadata,
            source=dataset.source,
            version=dataset.version,
            checksum=train_checksum
        )
        
        val_dataset = Dataset(
            texts=val_texts,
            labels=val_labels,
            metadata=val_metadata,
            source=dataset.source,
            version=dataset.version,
            checksum=val_checksum
        )
        
        test_dataset = Dataset(
            texts=test_texts,
            labels=test_labels,
            metadata=test_metadata,
            source=dataset.source,
            version=dataset.version,
            checksum=test_checksum
        )
        
        return train_dataset, val_dataset, test_dataset
    
    def _download_github_dataset(self, repo_url: str) -> Tuple[List[str], List[int]]:
        """
        Download dataset from a GitHub repository.
        
        Args:
            repo_url: GitHub repository URL
            
        Returns:
            Tuple of (texts, labels)
            
        Note: This is a placeholder implementation. In production, this would:
        - Clone the repository or use GitHub API
        - Parse dataset files (JSON, CSV, etc.)
        - Extract labeled examples
        """
        # For now, return mock data
        # In production, implement actual GitHub dataset downloading
        logger.warning(f"Using mock data for GitHub dataset: {repo_url}")
        
        # Mock data: simple examples of secrets and non-secrets
        texts = [
            'api_key = "AKIAIOSFODNN7EXAMPLE"',
            'password = "super_secret_123"',
            'const token = "ghp_1234567890abcdefghijklmnopqrstuvwxyz"',
            'username = "john_doe"',
            'config = {"debug": true}'
        ]
        labels = [1, 1, 1, 0, 0]  # 1 = secret, 0 = not secret
        
        return texts, labels
    
    def _download_kaggle_dataset(self, dataset_id: str) -> Tuple[List[str], List[int]]:
        """
        Download dataset from Kaggle.
        
        Args:
            dataset_id: Kaggle dataset identifier
            
        Returns:
            Tuple of (texts, labels)
            
        Note: This is a placeholder implementation. In production, this would:
        - Use Kaggle API to download dataset
        - Parse dataset files
        - Extract labeled PII examples
        """
        # For now, return mock data
        # In production, implement actual Kaggle dataset downloading
        logger.warning(f"Using mock data for Kaggle dataset: {dataset_id}")
        
        # Mock data: simple examples of PII and non-PII
        texts = [
            'email: john.doe@example.com',
            'SSN: 123-45-6789',
            'Phone: (555) 123-4567',
            'The weather is nice today',
            'Product ID: ABC-123'
        ]
        labels = [1, 1, 1, 0, 0]  # 1 = PII, 0 = not PII
        
        return texts, labels
    
    def _calculate_checksum(self, texts: List[str], labels: List[int]) -> str:
        """
        Calculate SHA256 checksum for dataset integrity verification.
        
        Args:
            texts: List of text examples
            labels: List of labels
            
        Returns:
            Hexadecimal checksum string
        """
        # Combine texts and labels into a single string for hashing
        combined = "".join(texts) + "".join(str(label) for label in labels)
        checksum = hashlib.sha256(combined.encode('utf-8')).hexdigest()
        return checksum
