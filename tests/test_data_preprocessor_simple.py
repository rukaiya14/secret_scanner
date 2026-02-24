"""
Simple unit tests for DataPreprocessor class without requiring transformers.

Tests class balancing and data augmentation functionality.
"""

import pytest
from collections import Counter
from unittest.mock import Mock, MagicMock, patch
import sys

from ml_scanner.models import Dataset
from ml_scanner.exceptions import PreprocessingError


# Mock transformers before importing DataPreprocessor
sys.modules['transformers'] = MagicMock()


class TestDataPreprocessorSimple:
    """Test suite for DataPreprocessor class (without tokenizer)."""
    
    def test_balance_classes_no_balancing_needed(self):
        """Test class balancing when ratio is already acceptable."""
        from ml_scanner.data_preprocessor import DataPreprocessor
        
        with patch('ml_scanner.data_preprocessor.AutoTokenizer'):
            preprocessor = DataPreprocessor()
        
        # Create dataset with 2:1 ratio (within 3:1 threshold)
        dataset = Dataset(
            texts=['text1', 'text2', 'text3', 'text4'],
            labels=[1, 1, 0, 0],  # 2 of each class
            metadata={},
            source='test',
            version='1.0.0',
            checksum='test'
        )
        
        balanced = preprocessor.balance_classes(dataset, max_ratio=3.0)
        
        # Should return same dataset
        assert len(balanced.texts) == len(dataset.texts)
        assert len(balanced.labels) == len(dataset.labels)
    
    def test_balance_classes_with_imbalance(self):
        """Test class balancing with imbalanced dataset."""
        from ml_scanner.data_preprocessor import DataPreprocessor
        
        with patch('ml_scanner.data_preprocessor.AutoTokenizer'):
            preprocessor = DataPreprocessor()
        
        # Create imbalanced dataset: 10 majority, 2 minority (5:1 ratio)
        dataset = Dataset(
            texts=['text' + str(i) for i in range(12)],
            labels=[0] * 10 + [1] * 2,  # 10 class 0, 2 class 1
            metadata={},
            source='test',
            version='1.0.0',
            checksum='test'
        )
        
        balanced = preprocessor.balance_classes(dataset, max_ratio=3.0)
        
        # Count classes in balanced dataset
        label_counts = Counter(balanced.labels)
        
        # Verify ratio is within threshold
        majority_count = max(label_counts.values())
        minority_count = min(label_counts.values())
        ratio = majority_count / minority_count
        
        assert ratio <= 3.0, f"Ratio {ratio} exceeds max_ratio 3.0"
        
        # Verify minority class is preserved
        assert label_counts[1] == 2
        
        # Verify majority class is reduced
        assert label_counts[0] <= 6  # Should be around 2 * 3 = 6
    
    def test_balance_classes_empty_dataset(self):
        """Test that balancing empty dataset raises error."""
        from ml_scanner.data_preprocessor import DataPreprocessor
        
        with patch('ml_scanner.data_preprocessor.AutoTokenizer'):
            preprocessor = DataPreprocessor()
        
        dataset = Dataset(
            texts=[],
            labels=[],
            metadata={},
            source='test',
            version='1.0.0',
            checksum='test'
        )
        
        with pytest.raises(PreprocessingError, match="Cannot balance empty dataset"):
            preprocessor.balance_classes(dataset)
    
    def test_augment_data_no_augmentation_needed(self):
        """Test data augmentation when ratio is acceptable."""
        from ml_scanner.data_preprocessor import DataPreprocessor
        
        with patch('ml_scanner.data_preprocessor.AutoTokenizer'):
            preprocessor = DataPreprocessor()
        
        # Create dataset with 2:1 ratio (within 3:1 threshold)
        dataset = Dataset(
            texts=['text1', 'text2', 'text3', 'text4'],
            labels=[0, 0, 1, 1],  # 2 of each class
            metadata={},
            source='test',
            version='1.0.0',
            checksum='test'
        )
        
        augmented = preprocessor.augment_data(dataset)
        
        # Should return same dataset (no augmentation needed)
        assert len(augmented.texts) == len(dataset.texts)
        assert len(augmented.labels) == len(dataset.labels)
    
    def test_augment_data_with_imbalance(self):
        """Test data augmentation with imbalanced dataset."""
        from ml_scanner.data_preprocessor import DataPreprocessor
        
        with patch('ml_scanner.data_preprocessor.AutoTokenizer'):
            preprocessor = DataPreprocessor()
        
        # Create imbalanced dataset: 12 majority, 2 minority (6:1 ratio)
        dataset = Dataset(
            texts=['text' + str(i) for i in range(14)],
            labels=[0] * 12 + [1] * 2,  # 12 class 0, 2 class 1
            metadata={},
            source='test',
            version='1.0.0',
            checksum='test'
        )
        
        augmented = preprocessor.augment_data(dataset)
        
        # Count classes in augmented dataset
        label_counts = Counter(augmented.labels)
        
        # Verify minority class has been augmented
        assert label_counts[1] > 2, "Minority class should be augmented"
        
        # Verify ratio is closer to 3:1
        majority_count = label_counts[0]
        minority_count = label_counts[1]
        ratio = majority_count / minority_count
        
        assert ratio <= 3.0, f"Ratio {ratio} should be <= 3.0 after augmentation"
    
    def test_augment_data_empty_dataset(self):
        """Test that augmenting empty dataset raises error."""
        from ml_scanner.data_preprocessor import DataPreprocessor
        
        with patch('ml_scanner.data_preprocessor.AutoTokenizer'):
            preprocessor = DataPreprocessor()
        
        dataset = Dataset(
            texts=[],
            labels=[],
            metadata={},
            source='test',
            version='1.0.0',
            checksum='test'
        )
        
        with pytest.raises(PreprocessingError, match="Cannot augment empty dataset"):
            preprocessor.augment_data(dataset)
    
    def test_augment_text_variations(self):
        """Test that text augmentation produces variations."""
        from ml_scanner.data_preprocessor import DataPreprocessor
        
        with patch('ml_scanner.data_preprocessor.AutoTokenizer'):
            preprocessor = DataPreprocessor()
        
        original_text = 'api_key = "AKIAIOSFODNN7EXAMPLE"'
        
        # Generate multiple augmentations
        augmented_texts = [preprocessor._augment_text(original_text) for _ in range(10)]
        
        # At least some should be different from original
        # (due to randomness, not all may be different)
        different_count = sum(1 for text in augmented_texts if text != original_text)
        
        assert different_count > 0, "At least some augmentations should differ from original"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
