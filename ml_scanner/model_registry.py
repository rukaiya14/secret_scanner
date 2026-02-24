"""
ModelRegistry for ML-Enhanced Secret Scanner.

This module manages trained CodeBERT model versions with semantic versioning,
environment-based deployment (STAGING/PRODUCTION), and fallback mechanisms.
"""

import json
import logging
import os
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict

from transformers import AutoModelForSequenceClassification, PreTrainedModel

from ml_scanner.models import ModelMetadata
from ml_scanner.exceptions import (
    ModelNotFoundError,
    ModelLoadError,
    RegistryError,
    StorageQuotaError
)


logger = logging.getLogger(__name__)


class ModelRegistry:
    """
    Manages versioned storage and retrieval of trained CodeBERT models.
    
    Provides:
    - Semantic versioning (MAJOR.MINOR.PATCH)
    - Environment-based deployment (STAGING/PRODUCTION)
    - Model promotion between environments
    - Last known good model fallback
    - Model lineage tracking
    - Version retention (keeps 5 most recent)
    
    Storage Structure:
        model_registry/
        ├── models/
        │   ├── 1.0.0/
        │   │   ├── model.bin
        │   │   ├── config.json
        │   │   ├── tokenizer_config.json
        │   │   └── metadata.json
        │   ├── 1.0.1/
        │   └── 1.1.0/
        ├── environments/
        │   ├── staging.json
        │   └── production.json
        └── cache/
            └── last_known_good.json
    
    Validates: Requirements 6.1, 6.2, 6.3, 6.4, 6.6, 6.7, 6.8
    """
    
    # Semantic version regex pattern
    SEMVER_PATTERN = re.compile(r'^(\d+)\.(\d+)\.(\d+)$')
    
    # Valid environments
    VALID_ENVIRONMENTS = {'STAGING', 'PRODUCTION'}
    
    # Maximum number of model versions to retain
    MAX_VERSIONS = 5
    
    def __init__(self, storage_path: str = "model_registry"):
        """
        Initialize model registry.
        
        Args:
            storage_path: Base path for model storage (local or cloud)
        """
        self.storage_path = Path(storage_path)
        self.models_path = self.storage_path / "models"
        self.environments_path = self.storage_path / "environments"
        self.cache_path = self.storage_path / "cache"
        
        # Create directory structure
        self._initialize_storage()
        
        logger.info(f"Initialized ModelRegistry at {self.storage_path}")
    
    def _initialize_storage(self) -> None:
        """Create registry directory structure if it doesn't exist."""
        self.models_path.mkdir(parents=True, exist_ok=True)
        self.environments_path.mkdir(parents=True, exist_ok=True)
        self.cache_path.mkdir(parents=True, exist_ok=True)
        
        logger.debug("Registry directory structure initialized")
    
    def register_model(
        self,
        model: PreTrainedModel,
        version: str,
        metadata: ModelMetadata
    ) -> str:
        """
        Register a new model version.
        
        Args:
            model: Trained model to register
            version: Semantic version string (e.g., "1.0.0")
            metadata: Model metadata
            
        Returns:
            Model ID (same as version)
            
        Raises:
            RegistryError: If registration fails
            ValueError: If version format is invalid
            
        Validates: Requirements 6.1, 6.2, 6.5, 6.7
        """
        logger.info(f"Registering model version {version}")
        
        # Validate semantic version format
        if not self._is_valid_semver(version):
            raise ValueError(
                f"Invalid semantic version format: {version}. "
                f"Expected format: MAJOR.MINOR.PATCH (e.g., 1.0.0)"
            )
        
        # Validate metadata completeness
        self._validate_metadata(metadata)
        
        try:
            # Create version directory
            model_dir = self.models_path / version
            if model_dir.exists():
                logger.warning(f"Model version {version} already exists, overwriting")
                shutil.rmtree(model_dir)
            
            model_dir.mkdir(parents=True, exist_ok=True)
            
            # Save model
            model.save_pretrained(str(model_dir))
            logger.debug(f"Model saved to {model_dir}")
            
            # Save metadata
            metadata_path = model_dir / "metadata.json"
            self._save_metadata(metadata, metadata_path)
            logger.debug(f"Metadata saved to {metadata_path}")
            
            # Update last known good cache
            self._update_last_known_good(version)
            
            # Enforce version retention policy
            self._enforce_version_retention()
            
            logger.info(f"Model version {version} registered successfully")
            return version
            
        except Exception as e:
            error_msg = f"Failed to register model version {version}: {str(e)}"
            logger.error(error_msg)
            raise RegistryError(error_msg, {
                "version": version,
                "storage_path": str(self.storage_path)
            }) from e
    
    def get_model(
        self,
        version: str = "latest",
        environment: Optional[str] = None
    ) -> PreTrainedModel:
        """
        Retrieve model by version or environment tag.
        
        Args:
            version: Version string or "latest" (default: "latest")
            environment: Environment tag ("STAGING" or "PRODUCTION")
                        If specified, overrides version parameter
            
        Returns:
            Loaded model
            
        Raises:
            ModelNotFoundError: If model not found
            ModelLoadError: If model loading fails
            
        Validates: Requirements 6.6, 6.8
        """
        try:
            # Determine which version to load
            if environment:
                if environment not in self.VALID_ENVIRONMENTS:
                    raise ValueError(
                        f"Invalid environment: {environment}. "
                        f"Must be one of {self.VALID_ENVIRONMENTS}"
                    )
                version = self._get_environment_version(environment)
                logger.info(f"Loading model from {environment} environment: {version}")
            elif version == "latest":
                version = self._get_latest_version()
                logger.info(f"Loading latest model version: {version}")
            else:
                logger.info(f"Loading model version: {version}")
            
            # Load model
            model_dir = self.models_path / version
            if not model_dir.exists():
                raise ModelNotFoundError(
                    f"Model version {version} not found",
                    {"version": version, "model_dir": str(model_dir)}
                )
            
            model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
            logger.info(f"Model version {version} loaded successfully")
            return model
            
        except (ModelNotFoundError, ValueError):
            # Re-raise these exceptions as-is
            raise
        except Exception as e:
            # Try fallback to last known good model
            logger.warning(
                f"Failed to load model version {version}: {str(e)}. "
                f"Attempting fallback to last known good model"
            )
            try:
                return self.get_last_known_good()
            except Exception as fallback_error:
                error_msg = (
                    f"Failed to load model version {version} and "
                    f"fallback also failed: {str(fallback_error)}"
                )
                logger.error(error_msg)
                raise ModelLoadError(error_msg, {
                    "version": version,
                    "original_error": str(e),
                    "fallback_error": str(fallback_error)
                }) from e
    
    def list_models(self, limit: int = 10) -> List[ModelMetadata]:
        """
        List available model versions.
        
        Args:
            limit: Maximum number of models to return (default: 10)
            
        Returns:
            List of ModelMetadata objects, sorted by version (newest first)
            
        Validates: Requirements 6.1, 6.2
        """
        logger.debug(f"Listing models (limit={limit})")
        
        models = []
        
        # Iterate through model directories
        if not self.models_path.exists():
            logger.warning("Models directory does not exist")
            return models
        
        for model_dir in self.models_path.iterdir():
            if not model_dir.is_dir():
                continue
            
            version = model_dir.name
            if not self._is_valid_semver(version):
                logger.warning(f"Skipping invalid version directory: {version}")
                continue
            
            # Load metadata
            metadata_path = model_dir / "metadata.json"
            if not metadata_path.exists():
                logger.warning(f"Metadata not found for version {version}")
                continue
            
            try:
                metadata = self._load_metadata(metadata_path)
                models.append(metadata)
            except Exception as e:
                logger.warning(f"Failed to load metadata for version {version}: {e}")
                continue
        
        # Sort by version (newest first)
        models.sort(key=lambda m: self._parse_semver(m.version), reverse=True)
        
        # Apply limit
        models = models[:limit]
        
        logger.info(f"Found {len(models)} model versions")
        return models
    
    def promote_model(
        self,
        version: str,
        from_env: str,
        to_env: str
    ) -> bool:
        """
        Promote model between environments.
        
        Args:
            version: Model version to promote
            from_env: Source environment ("STAGING" or "PRODUCTION")
            to_env: Target environment ("STAGING" or "PRODUCTION")
            
        Returns:
            True if promotion successful
            
        Raises:
            ValueError: If environments are invalid
            ModelNotFoundError: If model version not found
            RegistryError: If promotion fails
            
        Validates: Requirements 6.3
        """
        logger.info(f"Promoting model {version} from {from_env} to {to_env}")
        
        # Validate environments
        if from_env not in self.VALID_ENVIRONMENTS:
            raise ValueError(f"Invalid source environment: {from_env}")
        if to_env not in self.VALID_ENVIRONMENTS:
            raise ValueError(f"Invalid target environment: {to_env}")
        
        # Check if model exists
        model_dir = self.models_path / version
        if not model_dir.exists():
            raise ModelNotFoundError(
                f"Model version {version} not found",
                {"version": version}
            )
        
        try:
            # Load and update metadata
            metadata_path = model_dir / "metadata.json"
            metadata = self._load_metadata(metadata_path)
            
            # Update environment
            metadata.environment = to_env
            self._save_metadata(metadata, metadata_path)
            
            # Update environment pointer
            self._set_environment_version(to_env, version)
            
            logger.info(
                f"Model {version} promoted from {from_env} to {to_env} successfully"
            )
            return True
            
        except Exception as e:
            error_msg = f"Failed to promote model {version}: {str(e)}"
            logger.error(error_msg)
            raise RegistryError(error_msg, {
                "version": version,
                "from_env": from_env,
                "to_env": to_env
            }) from e
    
    def get_last_known_good(self) -> PreTrainedModel:
        """
        Get last known good model for fallback.
        
        Returns:
            Last known good model
            
        Raises:
            ModelLoadError: If no last known good model exists
            
        Validates: Requirements 6.8
        """
        logger.info("Loading last known good model")
        
        cache_file = self.cache_path / "last_known_good.json"
        
        if not cache_file.exists():
            # Try to use latest version as fallback
            logger.warning("No last known good model cached, using latest version")
            try:
                version = self._get_latest_version()
                return self.get_model(version=version)
            except Exception as e:
                raise ModelLoadError(
                    "No last known good model available and no models in registry",
                    {"cache_file": str(cache_file)}
                ) from e
        
        try:
            with open(cache_file, 'r') as f:
                cache_data = json.load(f)
            
            version = cache_data.get("version")
            if not version:
                raise ModelLoadError(
                    "Invalid last known good cache: missing version",
                    {"cache_file": str(cache_file)}
                )
            
            logger.info(f"Last known good model version: {version}")
            return self.get_model(version=version)
            
        except Exception as e:
            error_msg = f"Failed to load last known good model: {str(e)}"
            logger.error(error_msg)
            raise ModelLoadError(error_msg, {
                "cache_file": str(cache_file)
            }) from e
    
    def _is_valid_semver(self, version: str) -> bool:
        """Check if version string follows semantic versioning format."""
        return bool(self.SEMVER_PATTERN.match(version))
    
    def _parse_semver(self, version: str) -> tuple:
        """Parse semantic version string into (major, minor, patch) tuple."""
        match = self.SEMVER_PATTERN.match(version)
        if not match:
            raise ValueError(f"Invalid semantic version: {version}")
        return tuple(int(x) for x in match.groups())
    
    def _validate_metadata(self, metadata: ModelMetadata) -> None:
        """
        Validate that metadata contains all required fields.
        
        Raises:
            ValueError: If metadata is incomplete
        """
        required_fields = {
            'version': metadata.version,
            'training_date': metadata.training_date,
            'dataset_version': metadata.dataset_version,
            'performance_metrics': metadata.performance_metrics,
            'environment': metadata.environment,
            'training_config': metadata.training_config,
            'lineage': metadata.lineage
        }
        
        missing_fields = [
            field for field, value in required_fields.items()
            if value is None
        ]
        
        if missing_fields:
            raise ValueError(
                f"Metadata missing required fields: {', '.join(missing_fields)}"
            )
        
        # Validate environment
        if metadata.environment not in self.VALID_ENVIRONMENTS:
            raise ValueError(
                f"Invalid environment in metadata: {metadata.environment}. "
                f"Must be one of {self.VALID_ENVIRONMENTS}"
            )
    
    def _save_metadata(self, metadata: ModelMetadata, path: Path) -> None:
        """Save metadata to JSON file."""
        metadata_dict = {
            "version": metadata.version,
            "training_date": metadata.training_date.isoformat(),
            "dataset_version": metadata.dataset_version,
            "performance_metrics": metadata.performance_metrics,
            "environment": metadata.environment,
            "training_config": metadata.training_config,
            "lineage": metadata.lineage
        }
        
        with open(path, 'w') as f:
            json.dump(metadata_dict, f, indent=2)
    
    def _load_metadata(self, path: Path) -> ModelMetadata:
        """Load metadata from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        
        return ModelMetadata(
            version=data["version"],
            training_date=datetime.fromisoformat(data["training_date"]),
            dataset_version=data["dataset_version"],
            performance_metrics=data["performance_metrics"],
            environment=data["environment"],
            training_config=data["training_config"],
            lineage=data["lineage"]
        )
    
    def _get_latest_version(self) -> str:
        """
        Get the latest model version.
        
        Returns:
            Latest version string
            
        Raises:
            ModelNotFoundError: If no models exist
        """
        models = self.list_models(limit=1)
        if not models:
            raise ModelNotFoundError(
                "No models found in registry",
                {"models_path": str(self.models_path)}
            )
        return models[0].version
    
    def _get_environment_version(self, environment: str) -> str:
        """
        Get the model version for a specific environment.
        
        Args:
            environment: Environment name ("STAGING" or "PRODUCTION")
            
        Returns:
            Version string for the environment
            
        Raises:
            ModelNotFoundError: If no model assigned to environment
        """
        env_file = self.environments_path / f"{environment.lower()}.json"
        
        if not env_file.exists():
            raise ModelNotFoundError(
                f"No model assigned to {environment} environment",
                {"environment": environment, "env_file": str(env_file)}
            )
        
        with open(env_file, 'r') as f:
            data = json.load(f)
        
        version = data.get("version")
        if not version:
            raise ModelNotFoundError(
                f"Invalid environment file for {environment}: missing version",
                {"environment": environment, "env_file": str(env_file)}
            )
        
        return version
    
    def _set_environment_version(self, environment: str, version: str) -> None:
        """Set the model version for a specific environment."""
        env_file = self.environments_path / f"{environment.lower()}.json"
        
        data = {
            "version": version,
            "updated_at": datetime.now().isoformat()
        }
        
        with open(env_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.debug(f"Set {environment} environment to version {version}")
    
    def _update_last_known_good(self, version: str) -> None:
        """Update the last known good model cache."""
        cache_file = self.cache_path / "last_known_good.json"
        
        data = {
            "version": version,
            "updated_at": datetime.now().isoformat()
        }
        
        with open(cache_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.debug(f"Updated last known good model to version {version}")
    
    def _enforce_version_retention(self) -> None:
        """
        Enforce version retention policy (keep MAX_VERSIONS most recent).
        
        Validates: Requirements 6.4
        """
        models = self.list_models(limit=None)  # Get all models
        
        if len(models) <= self.MAX_VERSIONS:
            return  # No cleanup needed
        
        # Keep MAX_VERSIONS most recent, delete the rest
        models_to_delete = models[self.MAX_VERSIONS:]
        
        for metadata in models_to_delete:
            version = metadata.version
            model_dir = self.models_path / version
            
            try:
                shutil.rmtree(model_dir)
                logger.info(f"Deleted old model version {version} (retention policy)")
            except Exception as e:
                logger.warning(f"Failed to delete old model version {version}: {e}")
