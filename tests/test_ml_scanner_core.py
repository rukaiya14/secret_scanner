"""
Unit tests for ML scanner core infrastructure.

Tests the basic functionality of data models, configuration, logging,
and exception handling without requiring ML dependencies.
"""

import pytest
import json
import yaml
from pathlib import Path
from datetime import datetime
from ml_scanner.models import (
    Finding, Detection, Explanation, ModelMetadata,
    ScanResult, Dataset, TrainingMetrics, FeedbackEntry
)
from ml_scanner.exceptions import (
    MLScannerError, ErrorCode, DatasetLoadError, ConfigLoadError,
    ConfigValidationError
)
from ml_scanner.config import Config, DEFAULT_CONFIG
from ml_scanner.logger import get_logger, setup_logging


class TestDataModels:
    """Test core data models."""
    
    def test_finding_creation(self):
        """Test Finding dataclass creation."""
        finding = Finding(
            file_path="test.py",
            line_number=10,
            matched_text="AKIA123456",
            secret_type="AWS Key",
            confidence="HIGH",
            entropy=4.5,
            is_comment=False,
            source="REGEX_ONLY"
        )
        
        assert finding.file_path == "test.py"
        assert finding.line_number == 10
        assert finding.confidence == "HIGH"
        assert finding.source == "REGEX_ONLY"
        assert finding.confidence_score is None
    
    def test_finding_with_ml_data(self):
        """Test Finding with ML-specific fields."""
        finding = Finding(
            file_path="test.py",
            line_number=10,
            matched_text="AKIA123456",
            secret_type="AWS Key",
            confidence="HIGH",
            entropy=4.5,
            is_comment=False,
            source="ML_ONLY",
            confidence_score=0.95
        )
        
        assert finding.confidence_score == 0.95
        assert finding.source == "ML_ONLY"
    
    def test_detection_creation(self):
        """Test Detection dataclass creation."""
        detection = Detection(
            text="AKIA123456",
            start_pos=0,
            end_pos=10,
            category="API_KEY",
            confidence_score=0.92
        )
        
        assert detection.category == "API_KEY"
        assert detection.confidence_score == 0.92
        assert detection.token_attributions is None
    
    def test_scan_result_creation(self):
        """Test ScanResult dataclass creation."""
        findings = [
            Finding(
                file_path="test.py",
                line_number=10,
                matched_text="secret",
                secret_type="API Key",
                confidence="HIGH",
                entropy=4.5,
                is_comment=False,
                source="HIGH_CONFIDENCE"
            )
        ]
        
        result = ScanResult(
            findings=findings,
            files_scanned=5,
            high_confidence_count=1,
            low_confidence_count=0,
            execution_time_seconds=2.5,
            ml_available=True,
            fallback_mode=False
        )
        
        assert len(result.findings) == 1
        assert result.files_scanned == 5
        assert result.ml_available is True
        assert result.fallback_mode is False
    
    def test_dataset_creation(self):
        """Test Dataset dataclass creation."""
        dataset = Dataset(
            texts=["code sample 1", "code sample 2"],
            labels=[0, 1],
            metadata={"description": "test dataset"},
            source="github",
            version="1.0.0",
            checksum="abc123"
        )
        
        assert len(dataset.texts) == 2
        assert len(dataset.labels) == 2
        assert dataset.source == "github"
        assert dataset.version == "1.0.0"


class TestExceptions:
    """Test exception classes and error codes."""
    
    def test_error_code_enum(self):
        """Test ErrorCode enum values."""
        assert ErrorCode.DATASET_LOAD_FAILED.value == 1001
        assert ErrorCode.MODEL_LOAD_FAILED.value == 2001
        assert ErrorCode.REGISTRY_UNREACHABLE.value == 3001
        assert ErrorCode.EXPLANATION_TIMEOUT.value == 4001
        assert ErrorCode.SLACK_WEBHOOK_FAILED.value == 5001
        assert ErrorCode.SCANNER_INIT_FAILED.value == 6001
    
    def test_ml_scanner_error_creation(self):
        """Test MLScannerError base exception."""
        error = MLScannerError(
            "Test error",
            ErrorCode.DATASET_LOAD_FAILED,
            context={"url": "https://example.com"}
        )
        
        assert error.message == "Test error"
        assert error.error_code == ErrorCode.DATASET_LOAD_FAILED
        assert error.context["url"] == "https://example.com"
    
    def test_ml_scanner_error_to_dict(self):
        """Test MLScannerError serialization."""
        error = MLScannerError(
            "Test error",
            ErrorCode.MODEL_LOAD_FAILED,
            context={"model_version": "1.0.0"}
        )
        
        error_dict = error.to_dict()
        
        assert error_dict["error_code"] == 2001
        assert error_dict["error_name"] == "MODEL_LOAD_FAILED"
        assert error_dict["message"] == "Test error"
        assert error_dict["context"]["model_version"] == "1.0.0"
    
    def test_dataset_load_error(self):
        """Test DatasetLoadError exception."""
        error = DatasetLoadError(
            "Failed to load dataset",
            context={"url": "https://example.com"}
        )
        
        assert error.error_code == ErrorCode.DATASET_LOAD_FAILED
        assert "Failed to load dataset" in str(error)
    
    def test_config_load_error(self):
        """Test ConfigLoadError exception."""
        error = ConfigLoadError(
            "Config file not found",
            context={"path": "/path/to/config.yaml"}
        )
        
        assert error.error_code == ErrorCode.CONFIG_LOAD_FAILED


class TestConfiguration:
    """Test configuration management."""
    
    def test_default_config_creation(self):
        """Test Config with default values."""
        config = Config()
        
        assert config.get("ml_scanner.batch_size") == 16
        assert config.get("ml_scanner.explainer_backend") == "SHAP"
        assert config.get("ml_scanner.device") == "auto"
        assert config.get("training.epochs") == 3
    
    def test_config_get_nested_value(self):
        """Test getting nested configuration values."""
        config = Config()
        
        blocking = config.get("ml_scanner.confidence_thresholds.blocking")
        assert blocking == 0.90
        
        warning = config.get("ml_scanner.confidence_thresholds.warning")
        assert warning == 0.85
    
    def test_config_get_with_default(self):
        """Test getting non-existent value with default."""
        config = Config()
        
        value = config.get("nonexistent.key", default="default_value")
        assert value == "default_value"
    
    def test_config_set_value(self):
        """Test setting configuration values."""
        config = Config()
        
        config.set("ml_scanner.batch_size", 32)
        assert config.get("ml_scanner.batch_size") == 32
    
    def test_config_validation_invalid_threshold(self):
        """Test config validation with invalid threshold."""
        config = Config()
        config.set("ml_scanner.confidence_thresholds.blocking", 1.5)
        
        with pytest.raises(ConfigValidationError) as exc_info:
            config.validate_config()
        
        assert "must be between 0.0 and 1.0" in str(exc_info.value)
    
    def test_config_validation_invalid_explainer(self, tmp_path):
        """Test config validation with invalid explainer backend."""
        config_file = tmp_path / "config.yaml"
        config_data = {
            "ml_scanner": {
                "explainer_backend": "INVALID"
            }
        }
        
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)
        
        with pytest.raises(ConfigValidationError) as exc_info:
            Config(str(config_file))
        
        assert "must be 'SHAP' or 'LIME'" in str(exc_info.value)
    
    def test_config_validation_invalid_device(self, tmp_path):
        """Test config validation with invalid device."""
        config_file = tmp_path / "config.yaml"
        config_data = {
            "ml_scanner": {
                "device": "invalid_device"
            }
        }
        
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)
        
        with pytest.raises(ConfigValidationError) as exc_info:
            Config(str(config_file))
        
        assert "must be 'auto', 'cpu', or 'cuda'" in str(exc_info.value)
    
    def test_config_to_dict(self):
        """Test converting config to dictionary."""
        config = Config()
        config_dict = config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert "ml_scanner" in config_dict
        assert "training" in config_dict
        assert "model_registry" in config_dict
    
    def test_config_load_yaml(self, tmp_path):
        """Test loading configuration from YAML file."""
        config_file = tmp_path / "config.yaml"
        config_data = {
            "ml_scanner": {
                "batch_size": 32,
                "device": "cpu"
            }
        }
        
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)
        
        config = Config(str(config_file))
        
        assert config.get("ml_scanner.batch_size") == 32
        assert config.get("ml_scanner.device") == "cpu"
        # Default values should still be present
        assert config.get("ml_scanner.explainer_backend") == "SHAP"
    
    def test_config_load_json(self, tmp_path):
        """Test loading configuration from JSON file."""
        config_file = tmp_path / "config.json"
        config_data = {
            "ml_scanner": {
                "batch_size": 64,
                "device": "cuda"
            }
        }
        
        with open(config_file, 'w') as f:
            json.dump(config_data, f)
        
        config = Config(str(config_file))
        
        assert config.get("ml_scanner.batch_size") == 64
        assert config.get("ml_scanner.device") == "cuda"
    
    def test_config_load_nonexistent_file(self):
        """Test loading non-existent config file."""
        with pytest.raises(ConfigLoadError) as exc_info:
            Config("/nonexistent/config.yaml")
        
        assert "not found" in str(exc_info.value)
    
    def test_config_save_yaml(self, tmp_path):
        """Test saving configuration to YAML file."""
        config = Config()
        config.set("ml_scanner.batch_size", 128)
        
        output_file = tmp_path / "output.yaml"
        config.save(str(output_file))
        
        assert output_file.exists()
        
        # Load and verify
        with open(output_file, 'r') as f:
            loaded = yaml.safe_load(f)
        
        assert loaded["ml_scanner"]["batch_size"] == 128
    
    def test_config_save_json(self, tmp_path):
        """Test saving configuration to JSON file."""
        config = Config()
        config.set("ml_scanner.batch_size", 256)
        
        output_file = tmp_path / "output.json"
        config.save(str(output_file))
        
        assert output_file.exists()
        
        # Load and verify
        with open(output_file, 'r') as f:
            loaded = json.load(f)
        
        assert loaded["ml_scanner"]["batch_size"] == 256


class TestLogging:
    """Test logging infrastructure."""
    
    def test_get_logger(self):
        """Test getting a logger instance."""
        logger = get_logger("test_module")
        
        assert logger is not None
        assert hasattr(logger, 'debug')
        assert hasattr(logger, 'info')
        assert hasattr(logger, 'warning')
        assert hasattr(logger, 'error')
        assert hasattr(logger, 'critical')
    
    def test_logger_set_trace_id(self):
        """Test setting trace ID on logger."""
        logger = get_logger("test_module")
        
        trace_id = logger.set_trace_id("test-trace-123")
        assert trace_id == "test-trace-123"
        assert logger.trace_id == "test-trace-123"
    
    def test_logger_auto_generate_trace_id(self):
        """Test auto-generating trace ID."""
        logger = get_logger("test_module")
        
        trace_id = logger.set_trace_id()
        assert trace_id is not None
        assert len(trace_id) > 0
        assert logger.trace_id == trace_id
    
    def test_logger_methods_dont_crash(self):
        """Test that logger methods execute without errors."""
        logger = get_logger("test_module")
        
        # These should not raise exceptions
        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warning message")
        logger.error("Error message", error_code=2001)
        logger.critical("Critical message", error_code=3001)
    
    def test_logger_with_context(self):
        """Test logging with context dictionary."""
        logger = get_logger("test_module")
        
        # Should not raise exception
        logger.info("Operation completed", context={"files": 10, "duration": 2.5})
        logger.error("Operation failed", error_code=2001, context={"reason": "timeout"})


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
