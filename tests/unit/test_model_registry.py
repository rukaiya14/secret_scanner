"""
Unit tests for ModelRegistry class.

Tests model registration, retrieval, promotion, and fallback mechanisms.
"""

import json
import pytest
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch

from ml_scanner.model_registry import ModelRegistry
from ml_scanner.models import ModelMetadata
from ml_scanner.exceptions import (
    ModelNotFoundError,
    ModelLoadError,
    RegistryError
)


@pytest.fixture
def temp_registry_path():
    """Create a temporary directory for registry testing."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def sample_metadata():
    """Create sample model metadata for testing."""
    return ModelMetadata(
        version="1.0.0",
        training_date=datetime.now(),
        dataset_version="1.0",
        performance_metrics={
            "precision": 0.92,
            "recall": 0.88,
            "f1_score": 0.90
        },
        environment="STAGING",
        training_config={
            "epochs": 3,
            "learning_rate": 2e-5,
            "batch_size": 16
        },
        lineage={
            "github_datasets": ["truffleHogRegexes"],
            "kaggle_datasets": ["imdb-reviews"]
        }
    )


@pytest.fixture
def mock_model():
    """Create a mock PreTrainedModel for testing."""
    model = Mock()
    model.save_pretrained = Mock()
    return model


class TestModelRegistryInitialization:
    """Test ModelRegistry initialization and storage structure."""
    
    def test_initialization_creates_directory_structure(self, temp_registry_path):
        """Test that initialization creates required directories."""
        registry = ModelRegistry(storage_path=temp_registry_path)
        
        assert registry.models_path.exists()
        assert registry.environments_path.exists()
        assert registry.cache_path.exists()
    
    def test_initialization_with_existing_directory(self, temp_registry_path):
        """Test that initialization works with existing directory."""
        # Create registry twice
        registry1 = ModelRegistry(storage_path=temp_registry_path)
        registry2 = ModelRegistry(storage_path=temp_registry_path)
        
        assert registry1.storage_path == registry2.storage_path


class TestModelRegistration:
    """Test model registration functionality."""
    
    def test_register_model_success(self, temp_registry_path, mock_model, sample_metadata):
        """Test successful model registration."""
        registry = ModelRegistry(storage_path=temp_registry_path)
        
        model_id = registry.register_model(
            model=mock_model,
            version="1.0.0",
            metadata=sample_metadata
        )
        
        assert model_id == "1.0.0"
        assert (registry.models_path / "1.0.0").exists()
        assert (registry.models_path / "1.0.0" / "metadata.json").exists()
        
        # Verify model.save_pretrained was called
        mock_model.save_pretrained.assert_called_once()
    
    def test_register_model_invalid_version_format(self, temp_registry_path, mock_model, sample_metadata):
        """Test that invalid version format raises ValueError."""
        registry = ModelRegistry(storage_path=temp_registry_path)
        
        with pytest.raises(ValueError, match="Invalid semantic version format"):
            registry.register_model(
                model=mock_model,
                version="invalid-version",
                metadata=sample_metadata
            )
    
    def test_register_model_missing_metadata_fields(self, temp_registry_path, mock_model):
        """Test that incomplete metadata raises ValueError."""
        registry = ModelRegistry(storage_path=temp_registry_path)
        
        incomplete_metadata = ModelMetadata(
            version="1.0.0",
            training_date=None,  # Missing required field
            dataset_version="1.0",
            performance_metrics={},
            environment="STAGING",
            training_config={},
            lineage={}
        )
        
        with pytest.raises(ValueError, match="Metadata missing required fields"):
            registry.register_model(
                model=mock_model,
                version="1.0.0",
                metadata=incomplete_metadata
            )
    
    def test_register_model_updates_last_known_good(self, temp_registry_path, mock_model, sample_metadata):
        """Test that registration updates last known good cache."""
        registry = ModelRegistry(storage_path=temp_registry_path)
        
        registry.register_model(
            model=mock_model,
            version="1.0.0",
            metadata=sample_metadata
        )
        
        cache_file = registry.cache_path / "last_known_good.json"
        assert cache_file.exists()
        
        with open(cache_file, 'r') as f:
            cache_data = json.load(f)
        
        assert cache_data["version"] == "1.0.0"


class TestModelRetrieval:
    """Test model retrieval functionality."""
    
    def test_get_model_by_version(self, temp_registry_path, mock_model, sample_metadata):
        """Test retrieving model by specific version."""
        registry = ModelRegistry(storage_path=temp_registry_path)
        
        # Register a model
        registry.register_model(
            model=mock_model,
            version="1.0.0",
            metadata=sample_metadata
        )
        
        # Mock AutoModelForSequenceClassification.from_pretrained
        with patch('ml_scanner.model_registry.AutoModelForSequenceClassification.from_pretrained') as mock_load:
            mock_load.return_value = mock_model
            
            loaded_model = registry.get_model(version="1.0.0")
            
            assert loaded_model == mock_model
            mock_load.assert_called_once()
    
    def test_get_model_not_found(self, temp_registry_path):
        """Test that requesting non-existent model raises ModelNotFoundError."""
        registry = ModelRegistry(storage_path=temp_registry_path)
        
        with pytest.raises(ModelNotFoundError, match="Model version 1.0.0 not found"):
            registry.get_model(version="1.0.0")
    
    def test_get_model_by_environment(self, temp_registry_path, mock_model, sample_metadata):
        """Test retrieving model by environment tag."""
        registry = ModelRegistry(storage_path=temp_registry_path)
        
        # Register and promote model
        registry.register_model(
            model=mock_model,
            version="1.0.0",
            metadata=sample_metadata
        )
        registry._set_environment_version("PRODUCTION", "1.0.0")
        
        # Mock model loading
        with patch('ml_scanner.model_registry.AutoModelForSequenceClassification.from_pretrained') as mock_load:
            mock_load.return_value = mock_model
            
            loaded_model = registry.get_model(environment="PRODUCTION")
            
            assert loaded_model == mock_model
    
    def test_get_model_invalid_environment(self, temp_registry_path):
        """Test that invalid environment raises ValueError."""
        registry = ModelRegistry(storage_path=temp_registry_path)
        
        with pytest.raises(ValueError, match="Invalid environment"):
            registry.get_model(environment="INVALID")


class TestModelPromotion:
    """Test model promotion between environments."""
    
    def test_promote_model_success(self, temp_registry_path, mock_model, sample_metadata):
        """Test successful model promotion."""
        registry = ModelRegistry(storage_path=temp_registry_path)
        
        # Register model
        registry.register_model(
            model=mock_model,
            version="1.0.0",
            metadata=sample_metadata
        )
        
        # Promote from STAGING to PRODUCTION
        result = registry.promote_model(
            version="1.0.0",
            from_env="STAGING",
            to_env="PRODUCTION"
        )
        
        assert result is True
        
        # Verify environment file was created
        env_file = registry.environments_path / "production.json"
        assert env_file.exists()
        
        with open(env_file, 'r') as f:
            data = json.load(f)
        
        assert data["version"] == "1.0.0"
    
    def test_promote_model_invalid_environment(self, temp_registry_path, mock_model, sample_metadata):
        """Test that invalid environment raises ValueError."""
        registry = ModelRegistry(storage_path=temp_registry_path)
        
        registry.register_model(
            model=mock_model,
            version="1.0.0",
            metadata=sample_metadata
        )
        
        with pytest.raises(ValueError, match="Invalid source environment"):
            registry.promote_model(
                version="1.0.0",
                from_env="INVALID",
                to_env="PRODUCTION"
            )
    
    def test_promote_model_not_found(self, temp_registry_path):
        """Test that promoting non-existent model raises ModelNotFoundError."""
        registry = ModelRegistry(storage_path=temp_registry_path)
        
        with pytest.raises(ModelNotFoundError, match="Model version 1.0.0 not found"):
            registry.promote_model(
                version="1.0.0",
                from_env="STAGING",
                to_env="PRODUCTION"
            )


class TestModelListing:
    """Test model listing functionality."""
    
    def test_list_models_empty_registry(self, temp_registry_path):
        """Test listing models in empty registry."""
        registry = ModelRegistry(storage_path=temp_registry_path)
        
        models = registry.list_models()
        
        assert models == []
    
    def test_list_models_multiple_versions(self, temp_registry_path, mock_model):
        """Test listing multiple model versions."""
        registry = ModelRegistry(storage_path=temp_registry_path)
        
        # Register multiple versions
        versions = ["1.0.0", "1.0.1", "1.1.0"]
        for version in versions:
            metadata = ModelMetadata(
                version=version,
                training_date=datetime.now(),
                dataset_version="1.0",
                performance_metrics={"f1_score": 0.90},
                environment="STAGING",
                training_config={},
                lineage={}
            )
            registry.register_model(
                model=mock_model,
                version=version,
                metadata=metadata
            )
        
        models = registry.list_models()
        
        assert len(models) == 3
        # Should be sorted by version (newest first)
        assert models[0].version == "1.1.0"
        assert models[1].version == "1.0.1"
        assert models[2].version == "1.0.0"
    
    def test_list_models_with_limit(self, temp_registry_path, mock_model):
        """Test listing models with limit."""
        registry = ModelRegistry(storage_path=temp_registry_path)
        
        # Register multiple versions
        for i in range(5):
            metadata = ModelMetadata(
                version=f"1.0.{i}",
                training_date=datetime.now(),
                dataset_version="1.0",
                performance_metrics={"f1_score": 0.90},
                environment="STAGING",
                training_config={},
                lineage={}
            )
            registry.register_model(
                model=mock_model,
                version=f"1.0.{i}",
                metadata=metadata
            )
        
        models = registry.list_models(limit=2)
        
        assert len(models) == 2


class TestLastKnownGood:
    """Test last known good model fallback."""
    
    def test_get_last_known_good_success(self, temp_registry_path, mock_model, sample_metadata):
        """Test retrieving last known good model."""
        registry = ModelRegistry(storage_path=temp_registry_path)
        
        # Register model
        registry.register_model(
            model=mock_model,
            version="1.0.0",
            metadata=sample_metadata
        )
        
        # Mock model loading
        with patch('ml_scanner.model_registry.AutoModelForSequenceClassification.from_pretrained') as mock_load:
            mock_load.return_value = mock_model
            
            loaded_model = registry.get_last_known_good()
            
            assert loaded_model == mock_model
    
    def test_get_last_known_good_no_cache(self, temp_registry_path, mock_model, sample_metadata):
        """Test fallback to latest version when no cache exists."""
        registry = ModelRegistry(storage_path=temp_registry_path)
        
        # Register model
        registry.register_model(
            model=mock_model,
            version="1.0.0",
            metadata=sample_metadata
        )
        
        # Remove cache file
        cache_file = registry.cache_path / "last_known_good.json"
        cache_file.unlink()
        
        # Mock model loading
        with patch('ml_scanner.model_registry.AutoModelForSequenceClassification.from_pretrained') as mock_load:
            mock_load.return_value = mock_model
            
            loaded_model = registry.get_last_known_good()
            
            assert loaded_model == mock_model
    
    def test_get_last_known_good_empty_registry(self, temp_registry_path):
        """Test that empty registry raises ModelLoadError."""
        registry = ModelRegistry(storage_path=temp_registry_path)
        
        with pytest.raises(ModelLoadError, match="No last known good model available"):
            registry.get_last_known_good()


class TestVersionRetention:
    """Test version retention policy."""
    
    def test_version_retention_enforced(self, temp_registry_path, mock_model):
        """Test that old versions are deleted when exceeding MAX_VERSIONS."""
        registry = ModelRegistry(storage_path=temp_registry_path)
        
        # Register more than MAX_VERSIONS models
        for i in range(7):
            metadata = ModelMetadata(
                version=f"1.0.{i}",
                training_date=datetime.now(),
                dataset_version="1.0",
                performance_metrics={"f1_score": 0.90},
                environment="STAGING",
                training_config={},
                lineage={}
            )
            registry.register_model(
                model=mock_model,
                version=f"1.0.{i}",
                metadata=metadata
            )
        
        # Should only have MAX_VERSIONS (5) models
        models = registry.list_models(limit=None)
        assert len(models) <= registry.MAX_VERSIONS
        
        # Oldest versions should be deleted
        assert not (registry.models_path / "1.0.0").exists()
        assert not (registry.models_path / "1.0.1").exists()
        
        # Newest versions should exist
        assert (registry.models_path / "1.0.6").exists()
        assert (registry.models_path / "1.0.5").exists()


class TestSemanticVersioning:
    """Test semantic versioning validation."""
    
    def test_valid_semver_formats(self, temp_registry_path):
        """Test that valid semantic versions are accepted."""
        registry = ModelRegistry(storage_path=temp_registry_path)
        
        valid_versions = ["1.0.0", "0.1.0", "10.20.30", "1.0.1"]
        
        for version in valid_versions:
            assert registry._is_valid_semver(version)
    
    def test_invalid_semver_formats(self, temp_registry_path):
        """Test that invalid semantic versions are rejected."""
        registry = ModelRegistry(storage_path=temp_registry_path)
        
        invalid_versions = ["1.0", "v1.0.0", "1.0.0-beta", "1.0.0.0", "abc"]
        
        for version in invalid_versions:
            assert not registry._is_valid_semver(version)
    
    def test_semver_parsing(self, temp_registry_path):
        """Test semantic version parsing."""
        registry = ModelRegistry(storage_path=temp_registry_path)
        
        assert registry._parse_semver("1.0.0") == (1, 0, 0)
        assert registry._parse_semver("2.5.10") == (2, 5, 10)
        
        with pytest.raises(ValueError):
            registry._parse_semver("invalid")
