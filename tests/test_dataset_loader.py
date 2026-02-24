"""
Unit tests for DatasetLoader class.

Tests dataset loading, validation, and splitting functionality.
"""

import pytest
from ml_scanner.dataset_loader import DatasetLoader
from ml_scanner.models import Dataset
from ml_scanner.exceptions import DatasetLoadError


class TestDatasetLoader:
    """Test suite for DatasetLoader class."""
    
    def test_load_github_datasets_success(self):
        """Test successful loading of GitHub dataset."""
        loader = DatasetLoader()
        dataset = loader.load_github_datasets([
            "https://github.com/dxa4481/truffleHogRegexes"
        ])
        
        assert dataset is not None
        assert len(dataset.texts) > 0
        assert len(dataset.labels) == len(dataset.texts)
        assert dataset.source == "github"
        assert dataset.checksum is not None
        assert len(dataset.checksum) == 64  # SHA256 hex length
    
    def test_load_kaggle_datasets_success(self):
        """Test successful loading of Kaggle dataset."""
        loader = DatasetLoader()
        dataset = loader.load_kaggle_datasets([
            "lakshmi25npathi/imdb-dataset-of-50k-movie-reviews"
        ])
        
        assert dataset is not None
        assert len(dataset.texts) > 0
        assert len(dataset.labels) == len(dataset.texts)
        assert dataset.source == "kaggle"
        assert dataset.checksum is not None
    
    def test_validate_checksums_valid(self):
        """Test checksum validation with valid dataset."""
        loader = DatasetLoader()
        dataset = loader.load_github_datasets([
            "https://github.com/dxa4481/truffleHogRegexes"
        ])
        
        # Checksum should be valid immediately after loading
        assert loader.validate_checksums(dataset) is True
    
    def test_validate_checksums_invalid(self):
        """Test checksum validation with modified dataset."""
        loader = DatasetLoader()
        dataset = loader.load_github_datasets([
            "https://github.com/dxa4481/truffleHogRegexes"
        ])
        
        # Modify the dataset to invalidate checksum
        dataset.texts.append("new text")
        dataset.labels.append(1)
        
        # Checksum should now be invalid
        assert loader.validate_checksums(dataset) is False
    
    def test_split_dataset_proportions(self):
        """Test dataset splitting with correct proportions."""
        loader = DatasetLoader()
        dataset = loader.load_github_datasets([
            "https://github.com/dxa4481/truffleHogRegexes"
        ])
        
        train, val, test = loader.split_dataset(dataset)
        
        total_size = len(dataset.texts)
        train_size = len(train.texts)
        val_size = len(val.texts)
        test_size = len(test.texts)
        
        # Check that all examples are preserved
        assert train_size + val_size + test_size == total_size
        
        # For small datasets, just verify splits exist and are non-empty
        # For larger datasets (>20 examples), check proportions more strictly
        if total_size > 20:
            assert abs(train_size / total_size - 0.7) < 0.05
            assert abs(val_size / total_size - 0.15) < 0.05
            assert abs(test_size / total_size - 0.15) < 0.05
        else:
            # For small datasets, just ensure reasonable distribution
            assert train_size >= 1
            assert val_size >= 0
            assert test_size >= 0
    
    def test_split_dataset_custom_ratios(self):
        """Test dataset splitting with custom ratios."""
        loader = DatasetLoader()
        dataset = loader.load_github_datasets([
            "https://github.com/dxa4481/truffleHogRegexes"
        ])
        
        train, val, test = loader.split_dataset(
            dataset,
            train_ratio=0.6,
            val_ratio=0.2,
            test_ratio=0.2
        )
        
        total_size = len(dataset.texts)
        train_size = len(train.texts)
        val_size = len(val.texts)
        test_size = len(test.texts)
        
        # Check that all examples are preserved
        assert train_size + val_size + test_size == total_size
        
        # Check approximate proportions (60/20/20)
        assert abs(train_size / total_size - 0.6) < 0.05
        assert abs(val_size / total_size - 0.2) < 0.05
        assert abs(test_size / total_size - 0.2) < 0.05
    
    def test_split_dataset_invalid_ratios(self):
        """Test that invalid ratios raise ValueError."""
        loader = DatasetLoader()
        dataset = loader.load_github_datasets([
            "https://github.com/dxa4481/truffleHogRegexes"
        ])
        
        with pytest.raises(ValueError) as exc_info:
            loader.split_dataset(
                dataset,
                train_ratio=0.5,
                val_ratio=0.3,
                test_ratio=0.3  # Sum is 1.1, not 1.0
            )
        
        assert "must sum to 1.0" in str(exc_info.value)
    
    def test_split_dataset_metadata(self):
        """Test that split datasets have correct metadata."""
        loader = DatasetLoader()
        dataset = loader.load_github_datasets([
            "https://github.com/dxa4481/truffleHogRegexes"
        ])
        
        train, val, test = loader.split_dataset(dataset)
        
        # Check metadata
        assert train.metadata["split"] == "train"
        assert val.metadata["split"] == "validation"
        assert test.metadata["split"] == "test"
        
        assert train.metadata["size"] == len(train.texts)
        assert val.metadata["size"] == len(val.texts)
        assert test.metadata["size"] == len(test.texts)
    
    def test_split_dataset_checksums(self):
        """Test that split datasets have valid checksums."""
        loader = DatasetLoader()
        dataset = loader.load_github_datasets([
            "https://github.com/dxa4481/truffleHogRegexes"
        ])
        
        train, val, test = loader.split_dataset(dataset)
        
        # Each split should have a valid checksum
        assert loader.validate_checksums(train) is True
        assert loader.validate_checksums(val) is True
        assert loader.validate_checksums(test) is True
    
    def test_load_multiple_github_repos(self):
        """Test loading from multiple GitHub repositories."""
        loader = DatasetLoader()
        dataset = loader.load_github_datasets([
            "https://github.com/dxa4481/truffleHogRegexes",
            "https://github.com/awslabs/git-secrets"
        ])
        
        assert dataset is not None
        assert len(dataset.texts) > 0
        assert dataset.metadata["num_repos"] == 2
    
    def test_load_multiple_kaggle_datasets(self):
        """Test loading from multiple Kaggle datasets."""
        loader = DatasetLoader()
        dataset = loader.load_kaggle_datasets([
            "lakshmi25npathi/imdb-dataset-of-50k-movie-reviews",
            "another/dataset"
        ])
        
        assert dataset is not None
        assert len(dataset.texts) > 0
        assert dataset.metadata["num_datasets"] == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
