"""
DataPreprocessor for ML-Enhanced Secret Scanner.

This module handles tokenization, class balancing, and data augmentation
for training the CodeBERT model.
"""

import logging
import random
from typing import List
from collections import Counter

from transformers import AutoTokenizer, BatchEncoding

from ml_scanner.models import Dataset
from ml_scanner.exceptions import PreprocessingError


logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Preprocesses training data for CodeBERT model.
    
    Handles tokenization using CodeBERT tokenizer, class balancing to 
    maintain max 3:1 ratio, and data augmentation for minority classes.
    
    Validates: Requirements 1.4, 2.5, 2.6, 3.1
    """
    
    def __init__(self, model_name: str = "microsoft/codebert-base"):
        """
        Initialize DataPreprocessor with CodeBERT tokenizer.
        
        Args:
            model_name: Name of the CodeBERT model to use for tokenization
        """
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            logger.info(f"Initialized DataPreprocessor with tokenizer: {model_name}")
        except Exception as e:
            error_msg = f"Failed to load tokenizer {model_name}: {str(e)}"
            logger.error(error_msg)
            raise PreprocessingError(error_msg, {"model_name": model_name}) from e
    
    def tokenize(self, texts: List[str], max_length: int = 512) -> BatchEncoding:
        """
        Tokenize code samples using CodeBERT tokenizer.
        
        Args:
            texts: List of code samples to tokenize
            max_length: Maximum sequence length (default 512 tokens)
            
        Returns:
            BatchEncoding with tokenized inputs
            
        Raises:
            PreprocessingError: If tokenization fails
            
        Validates: Requirements 1.4, 3.1
        """
        if not texts:
            raise PreprocessingError("Cannot tokenize empty text list")
        
        if max_length > 512:
            logger.warning(f"max_length {max_length} exceeds recommended 512, using 512")
            max_length = 512
        
        try:
            logger.info(f"Tokenizing {len(texts)} texts with max_length={max_length}")
            
            # Tokenize with padding and truncation
            encoded = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            )
            
            logger.info(f"Tokenization complete. Shape: {encoded['input_ids'].shape}")
            return encoded
            
        except Exception as e:
            error_msg = f"Tokenization failed: {str(e)}"
            logger.error(error_msg)
            raise PreprocessingError(error_msg, {
                "num_texts": len(texts),
                "max_length": max_length
            }) from e
    
    def balance_classes(self, dataset: Dataset, max_ratio: float = 3.0) -> Dataset:
        """
        Balance classes to prevent bias toward non-secret examples.
        
        Undersamples the majority class to maintain a maximum class ratio.
        
        Args:
            dataset: Dataset to balance
            max_ratio: Maximum allowed ratio between majority and minority classes
            
        Returns:
            Balanced dataset
            
        Raises:
            PreprocessingError: If balancing fails
            
        Validates: Requirements 2.5
        """
        if not dataset.texts or not dataset.labels:
            raise PreprocessingError("Cannot balance empty dataset")
        
        if len(dataset.texts) != len(dataset.labels):
            raise PreprocessingError(
                f"Mismatched texts ({len(dataset.texts)}) and labels ({len(dataset.labels)})"
            )
        
        try:
            # Count class distribution
            label_counts = Counter(dataset.labels)
            logger.info(f"Original class distribution: {dict(label_counts)}")
            
            if len(label_counts) < 2:
                logger.info("Dataset has only one class, no balancing needed")
                return dataset
            
            # Find majority and minority classes
            majority_class = max(label_counts, key=label_counts.get)
            minority_class = min(label_counts, key=label_counts.get)
            
            majority_count = label_counts[majority_class]
            minority_count = label_counts[minority_class]
            
            # Calculate current ratio
            current_ratio = majority_count / minority_count if minority_count > 0 else float('inf')
            logger.info(f"Current class ratio: {current_ratio:.2f}:1")
            
            # Check if balancing is needed
            if current_ratio <= max_ratio:
                logger.info(f"Class ratio {current_ratio:.2f}:1 is within max_ratio {max_ratio}:1")
                return dataset
            
            # Calculate target majority count
            target_majority_count = int(minority_count * max_ratio)
            logger.info(
                f"Balancing: reducing majority class from {majority_count} to {target_majority_count}"
            )
            
            # Separate examples by class
            majority_indices = [i for i, label in enumerate(dataset.labels) if label == majority_class]
            minority_indices = [i for i, label in enumerate(dataset.labels) if label == minority_class]
            
            # Randomly sample from majority class
            random.shuffle(majority_indices)
            sampled_majority_indices = majority_indices[:target_majority_count]
            
            # Combine indices and sort to maintain some order
            balanced_indices = sorted(sampled_majority_indices + minority_indices)
            
            # Create balanced dataset
            balanced_texts = [dataset.texts[i] for i in balanced_indices]
            balanced_labels = [dataset.labels[i] for i in balanced_indices]
            
            # Update metadata
            balanced_metadata = {
                **dataset.metadata,
                "balanced": True,
                "original_size": len(dataset.texts),
                "balanced_size": len(balanced_texts),
                "max_ratio": max_ratio
            }
            
            # Calculate new checksum
            import hashlib
            combined = "".join(balanced_texts) + "".join(str(label) for label in balanced_labels)
            balanced_checksum = hashlib.sha256(combined.encode('utf-8')).hexdigest()
            
            balanced_dataset = Dataset(
                texts=balanced_texts,
                labels=balanced_labels,
                metadata=balanced_metadata,
                source=dataset.source,
                version=dataset.version,
                checksum=balanced_checksum
            )
            
            # Log final distribution
            final_counts = Counter(balanced_labels)
            logger.info(f"Balanced class distribution: {dict(final_counts)}")
            
            return balanced_dataset
            
        except Exception as e:
            error_msg = f"Class balancing failed: {str(e)}"
            logger.error(error_msg)
            raise PreprocessingError(error_msg, {
                "dataset_size": len(dataset.texts),
                "max_ratio": max_ratio
            }) from e
    
    def augment_data(self, dataset: Dataset) -> Dataset:
        """
        Generate synthetic examples for minority classes.
        
        Applies data augmentation techniques to increase minority class examples
        when class imbalance exceeds 3:1 ratio.
        
        Args:
            dataset: Dataset to augment
            
        Returns:
            Augmented dataset
            
        Raises:
            PreprocessingError: If augmentation fails
            
        Validates: Requirements 2.6
        """
        if not dataset.texts or not dataset.labels:
            raise PreprocessingError("Cannot augment empty dataset")
        
        if len(dataset.texts) != len(dataset.labels):
            raise PreprocessingError(
                f"Mismatched texts ({len(dataset.texts)}) and labels ({len(dataset.labels)})"
            )
        
        try:
            # Count class distribution
            label_counts = Counter(dataset.labels)
            logger.info(f"Class distribution before augmentation: {dict(label_counts)}")
            
            if len(label_counts) < 2:
                logger.info("Dataset has only one class, no augmentation needed")
                return dataset
            
            # Find majority and minority classes
            majority_class = max(label_counts, key=label_counts.get)
            minority_class = min(label_counts, key=label_counts.get)
            
            majority_count = label_counts[majority_class]
            minority_count = label_counts[minority_class]
            
            # Calculate current ratio
            current_ratio = majority_count / minority_count if minority_count > 0 else float('inf')
            logger.info(f"Current class ratio: {current_ratio:.2f}:1")
            
            # Check if augmentation is needed (ratio > 3:1)
            if current_ratio <= 3.0:
                logger.info(f"Class ratio {current_ratio:.2f}:1 does not exceed 3:1, no augmentation needed")
                return dataset
            
            # Calculate how many minority examples to generate
            target_minority_count = int(majority_count / 3.0)
            num_to_generate = target_minority_count - minority_count
            logger.info(f"Augmenting: generating {num_to_generate} minority class examples")
            
            # Get minority class examples
            minority_texts = [text for text, label in zip(dataset.texts, dataset.labels) 
                            if label == minority_class]
            
            # Generate augmented examples using simple techniques
            augmented_texts = []
            augmented_labels = []
            
            for _ in range(num_to_generate):
                # Randomly select a minority example to augment
                base_text = random.choice(minority_texts)
                
                # Apply augmentation technique (simple character-level perturbations)
                augmented_text = self._augment_text(base_text)
                augmented_texts.append(augmented_text)
                augmented_labels.append(minority_class)
            
            # Combine original and augmented data
            combined_texts = dataset.texts + augmented_texts
            combined_labels = dataset.labels + augmented_labels
            
            # Update metadata
            augmented_metadata = {
                **dataset.metadata,
                "augmented": True,
                "original_size": len(dataset.texts),
                "augmented_size": len(combined_texts),
                "num_generated": num_to_generate
            }
            
            # Calculate new checksum
            import hashlib
            combined_str = "".join(combined_texts) + "".join(str(label) for label in combined_labels)
            augmented_checksum = hashlib.sha256(combined_str.encode('utf-8')).hexdigest()
            
            augmented_dataset = Dataset(
                texts=combined_texts,
                labels=combined_labels,
                metadata=augmented_metadata,
                source=dataset.source,
                version=dataset.version,
                checksum=augmented_checksum
            )
            
            # Log final distribution
            final_counts = Counter(combined_labels)
            logger.info(f"Class distribution after augmentation: {dict(final_counts)}")
            
            return augmented_dataset
            
        except Exception as e:
            error_msg = f"Data augmentation failed: {str(e)}"
            logger.error(error_msg)
            raise PreprocessingError(error_msg, {
                "dataset_size": len(dataset.texts)
            }) from e
    
    def _augment_text(self, text: str) -> str:
        """
        Apply simple augmentation to a text sample.
        
        Uses character-level perturbations like:
        - Adding whitespace
        - Changing quotes
        - Minor formatting changes
        
        Args:
            text: Original text to augment
            
        Returns:
            Augmented text
        """
        # Apply random augmentation technique
        technique = random.choice(['whitespace', 'quotes', 'case'])
        
        if technique == 'whitespace':
            # Add or remove whitespace
            if random.random() > 0.5:
                # Add extra spaces
                text = text.replace(' ', '  ')
            else:
                # Remove some spaces
                text = text.replace('  ', ' ')
        
        elif technique == 'quotes':
            # Change quote style
            if '"' in text:
                text = text.replace('"', "'")
            elif "'" in text:
                text = text.replace("'", '"')
        
        elif technique == 'case':
            # Change case of variable names (simple heuristic)
            words = text.split()
            if words:
                idx = random.randint(0, len(words) - 1)
                word = words[idx]
                if word.isidentifier():
                    # Toggle between snake_case and camelCase style
                    if '_' in word:
                        words[idx] = word.replace('_', '')
                    else:
                        # Insert underscore at random position
                        if len(word) > 2:
                            pos = random.randint(1, len(word) - 1)
                            words[idx] = word[:pos] + '_' + word[pos:]
                text = ' '.join(words)
        
        return text