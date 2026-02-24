"""
Validation script for ModelRegistry implementation.

This script demonstrates the ModelRegistry functionality without requiring
PyTorch to be installed. It validates:
- Model registration with semantic versioning
- Storage structure creation
- Metadata management
- Environment promotion
- Version retention
- Last known good fallback
"""

import json
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock

from ml_scanner.model_registry import ModelRegistry
from ml_scanner.models import ModelMetadata


def validate_model_registry():
    """Validate ModelRegistry implementation."""
    print("=" * 70)
    print("ModelRegistry Validation")
    print("=" * 70)
    
    # Create temporary registry
    temp_dir = tempfile.mkdtemp()
    print(f"\n✓ Created temporary registry at: {temp_dir}")
    
    try:
        # Test 1: Initialization
        print("\n[Test 1] Initializing ModelRegistry...")
        registry = ModelRegistry(storage_path=temp_dir)
        
        assert registry.models_path.exists(), "Models directory not created"
        assert registry.environments_path.exists(), "Environments directory not created"
        assert registry.cache_path.exists(), "Cache directory not created"
        print("✓ Storage structure created successfully")
        
        # Test 2: Model Registration
        print("\n[Test 2] Registering model version 1.0.0...")
        mock_model = Mock()
        mock_model.save_pretrained = Mock()
        
        metadata = ModelMetadata(
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
        
        model_id = registry.register_model(
            model=mock_model,
            version="1.0.0",
            metadata=metadata
        )
        
        assert model_id == "1.0.0", "Model ID mismatch"
        assert (registry.models_path / "1.0.0").exists(), "Model directory not created"
        assert (registry.models_path / "1.0.0" / "metadata.json").exists(), "Metadata not saved"
        print("✓ Model registered successfully")
        
        # Test 3: Metadata Validation
        print("\n[Test 3] Validating metadata...")
        metadata_path = registry.models_path / "1.0.0" / "metadata.json"
        with open(metadata_path, 'r') as f:
            saved_metadata = json.load(f)
        
        assert saved_metadata["version"] == "1.0.0", "Version mismatch"
        assert saved_metadata["dataset_version"] == "1.0", "Dataset version mismatch"
        assert saved_metadata["environment"] == "STAGING", "Environment mismatch"
        assert "performance_metrics" in saved_metadata, "Performance metrics missing"
        assert "lineage" in saved_metadata, "Lineage missing"
        print("✓ Metadata validated successfully")
        
        # Test 4: Last Known Good Cache
        print("\n[Test 4] Validating last known good cache...")
        cache_file = registry.cache_path / "last_known_good.json"
        assert cache_file.exists(), "Last known good cache not created"
        
        with open(cache_file, 'r') as f:
            cache_data = json.load(f)
        
        assert cache_data["version"] == "1.0.0", "Cache version mismatch"
        print("✓ Last known good cache validated")
        
        # Test 5: Model Promotion
        print("\n[Test 5] Promoting model to PRODUCTION...")
        result = registry.promote_model(
            version="1.0.0",
            from_env="STAGING",
            to_env="PRODUCTION"
        )
        
        assert result is True, "Promotion failed"
        
        env_file = registry.environments_path / "production.json"
        assert env_file.exists(), "Production environment file not created"
        
        with open(env_file, 'r') as f:
            env_data = json.load(f)
        
        assert env_data["version"] == "1.0.0", "Environment version mismatch"
        print("✓ Model promoted to PRODUCTION successfully")
        
        # Test 6: Multiple Model Versions
        print("\n[Test 6] Registering multiple model versions...")
        versions = ["1.0.1", "1.0.2", "1.1.0"]
        
        for version in versions:
            metadata_v = ModelMetadata(
                version=version,
                training_date=datetime.now(),
                dataset_version="1.0",
                performance_metrics={"f1_score": 0.91},
                environment="STAGING",
                training_config={},
                lineage={}
            )
            registry.register_model(
                model=mock_model,
                version=version,
                metadata=metadata_v
            )
        
        models = registry.list_models()
        assert len(models) == 4, f"Expected 4 models, got {len(models)}"
        assert models[0].version == "1.1.0", "Models not sorted correctly"
        print(f"✓ Registered {len(models)} model versions")
        
        # Test 7: Version Retention
        print("\n[Test 7] Testing version retention policy...")
        # Register more models to exceed MAX_VERSIONS (5)
        for i in range(3, 8):
            metadata_v = ModelMetadata(
                version=f"1.0.{i}",
                training_date=datetime.now(),
                dataset_version="1.0",
                performance_metrics={"f1_score": 0.91},
                environment="STAGING",
                training_config={},
                lineage={}
            )
            registry.register_model(
                model=mock_model,
                version=f"1.0.{i}",
                metadata=metadata_v
            )
        
        models = registry.list_models(limit=None)
        assert len(models) <= registry.MAX_VERSIONS, \
            f"Version retention not enforced: {len(models)} > {registry.MAX_VERSIONS}"
        print(f"✓ Version retention enforced (kept {len(models)} most recent)")
        
        # Test 8: Semantic Versioning Validation
        print("\n[Test 8] Testing semantic versioning validation...")
        
        valid_versions = ["1.0.0", "0.1.0", "10.20.30"]
        for version in valid_versions:
            assert registry._is_valid_semver(version), f"Valid version rejected: {version}"
        
        invalid_versions = ["1.0", "v1.0.0", "1.0.0-beta", "abc"]
        for version in invalid_versions:
            assert not registry._is_valid_semver(version), f"Invalid version accepted: {version}"
        
        print("✓ Semantic versioning validation working correctly")
        
        # Test 9: Storage Structure
        print("\n[Test 9] Validating storage structure...")
        expected_dirs = [
            registry.models_path,
            registry.environments_path,
            registry.cache_path
        ]
        
        for dir_path in expected_dirs:
            assert dir_path.exists(), f"Directory not found: {dir_path}"
        
        # Check that model directories exist
        model_dirs = list(registry.models_path.iterdir())
        assert len(model_dirs) > 0, "No model directories found"
        
        print(f"✓ Storage structure validated ({len(model_dirs)} model versions)")
        
        # Summary
        print("\n" + "=" * 70)
        print("VALIDATION SUMMARY")
        print("=" * 70)
        print("✓ All tests passed successfully!")
        print(f"✓ ModelRegistry implementation is complete and functional")
        print(f"✓ Storage path: {temp_dir}")
        print(f"✓ Model versions: {len(models)}")
        print(f"✓ Requirements validated: 6.1, 6.2, 6.3, 6.4, 6.6, 6.7, 6.8")
        print("=" * 70)
        
        return True
        
    except Exception as e:
        print(f"\n✗ Validation failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Cleanup
        print(f"\nCleaning up temporary directory...")
        shutil.rmtree(temp_dir, ignore_errors=True)
        print("✓ Cleanup complete")


if __name__ == "__main__":
    success = validate_model_registry()
    exit(0 if success else 1)
