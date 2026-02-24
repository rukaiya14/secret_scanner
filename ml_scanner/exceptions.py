"""
Error codes and exception classes for the ML-Enhanced Secret Scanner.

This module defines standardized error codes and custom exceptions
for consistent error handling throughout the system.
"""

from enum import Enum


class ErrorCode(Enum):
    """Standardized error codes for the ML scanner system."""
    
    # Training Pipeline Errors (1000-1999)
    DATASET_LOAD_FAILED = 1001
    CHECKSUM_VALIDATION_FAILED = 1002
    TRAINING_FAILED = 1003
    PERFORMANCE_THRESHOLD_NOT_MET = 1004
    
    # Inference Engine Errors (2000-2999)
    MODEL_LOAD_FAILED = 2001
    INFERENCE_TIMEOUT = 2002
    GPU_OOM = 2003
    INVALID_INPUT = 2004
    
    # Model Registry Errors (3000-3999)
    REGISTRY_UNREACHABLE = 3001
    MODEL_NOT_FOUND = 3002
    STORAGE_QUOTA_EXCEEDED = 3003
    
    # Explainer Errors (4000-4999)
    EXPLANATION_TIMEOUT = 4001
    EXPLAINER_LIBRARY_ERROR = 4002
    
    # Alert Manager Errors (5000-5999)
    SLACK_WEBHOOK_FAILED = 5001
    MESSAGE_FORMATTING_ERROR = 5002
    
    # Git Hook Errors (6000-6999)
    SCANNER_INIT_FAILED = 6001
    SCAN_TIMEOUT = 6002
    GIT_COMMAND_FAILED = 6003
    
    # Configuration Errors (7000-7999)
    CONFIG_LOAD_FAILED = 7001
    CONFIG_VALIDATION_FAILED = 7002


class MLScannerError(Exception):
    """
    Base exception class for all ML scanner errors.
    
    All custom exceptions in the ML scanner system inherit from this class,
    providing a consistent interface for error handling.
    """
    
    def __init__(self, message: str, error_code: ErrorCode, context: dict = None):
        """
        Initialize ML scanner error.
        
        Args:
            message: Human-readable error message
            error_code: Standardized error code from ErrorCode enum
            context: Additional context information for debugging
        """
        self.message = message
        self.error_code = error_code
        self.context = context or {}
        super().__init__(self.message)
    
    def to_dict(self) -> dict:
        """Convert exception to dictionary for structured logging."""
        return {
            "error_code": self.error_code.value,
            "error_name": self.error_code.name,
            "message": self.message,
            "context": self.context,
        }


class DatasetLoadError(MLScannerError):
    """Exception raised when dataset loading fails."""
    
    def __init__(self, message: str, context: dict = None):
        super().__init__(message, ErrorCode.DATASET_LOAD_FAILED, context)


class ChecksumValidationError(MLScannerError):
    """Exception raised when dataset checksum validation fails."""
    
    def __init__(self, message: str, context: dict = None):
        super().__init__(message, ErrorCode.CHECKSUM_VALIDATION_FAILED, context)


class TrainingError(MLScannerError):
    """Exception raised when model training fails."""
    
    def __init__(self, message: str, context: dict = None):
        super().__init__(message, ErrorCode.TRAINING_FAILED, context)


class PerformanceThresholdError(MLScannerError):
    """Exception raised when model doesn't meet performance thresholds."""
    
    def __init__(self, message: str, context: dict = None):
        super().__init__(message, ErrorCode.PERFORMANCE_THRESHOLD_NOT_MET, context)


class ModelLoadError(MLScannerError):
    """Exception raised when model loading fails."""
    
    def __init__(self, message: str, context: dict = None):
        super().__init__(message, ErrorCode.MODEL_LOAD_FAILED, context)


class InferenceTimeoutError(MLScannerError):
    """Exception raised when inference times out."""
    
    def __init__(self, message: str, context: dict = None):
        super().__init__(message, ErrorCode.INFERENCE_TIMEOUT, context)


class GPUOutOfMemoryError(MLScannerError):
    """Exception raised when GPU runs out of memory."""
    
    def __init__(self, message: str, context: dict = None):
        super().__init__(message, ErrorCode.GPU_OOM, context)


class InvalidInputError(MLScannerError):
    """Exception raised when input is invalid."""
    
    def __init__(self, message: str, context: dict = None):
        super().__init__(message, ErrorCode.INVALID_INPUT, context)


class RegistryError(MLScannerError):
    """Exception raised when model registry operations fail."""
    
    def __init__(self, message: str, context: dict = None):
        super().__init__(message, ErrorCode.REGISTRY_UNREACHABLE, context)


class ModelNotFoundError(MLScannerError):
    """Exception raised when requested model is not found."""
    
    def __init__(self, message: str, context: dict = None):
        super().__init__(message, ErrorCode.MODEL_NOT_FOUND, context)


class StorageQuotaError(MLScannerError):
    """Exception raised when storage quota is exceeded."""
    
    def __init__(self, message: str, context: dict = None):
        super().__init__(message, ErrorCode.STORAGE_QUOTA_EXCEEDED, context)


class ExplanationTimeoutError(MLScannerError):
    """Exception raised when explanation generation times out."""
    
    def __init__(self, message: str, context: dict = None):
        super().__init__(message, ErrorCode.EXPLANATION_TIMEOUT, context)


class ExplainerLibraryError(MLScannerError):
    """Exception raised when SHAP/LIME library encounters an error."""
    
    def __init__(self, message: str, context: dict = None):
        super().__init__(message, ErrorCode.EXPLAINER_LIBRARY_ERROR, context)


class SlackWebhookError(MLScannerError):
    """Exception raised when Slack webhook fails."""
    
    def __init__(self, message: str, context: dict = None):
        super().__init__(message, ErrorCode.SLACK_WEBHOOK_FAILED, context)


class MessageFormattingError(MLScannerError):
    """Exception raised when message formatting fails."""
    
    def __init__(self, message: str, context: dict = None):
        super().__init__(message, ErrorCode.MESSAGE_FORMATTING_ERROR, context)


class ScannerInitError(MLScannerError):
    """Exception raised when scanner initialization fails."""
    
    def __init__(self, message: str, context: dict = None):
        super().__init__(message, ErrorCode.SCANNER_INIT_FAILED, context)


class ScanTimeoutError(MLScannerError):
    """Exception raised when scan times out."""
    
    def __init__(self, message: str, context: dict = None):
        super().__init__(message, ErrorCode.SCAN_TIMEOUT, context)


class GitCommandError(MLScannerError):
    """Exception raised when git command fails."""
    
    def __init__(self, message: str, context: dict = None):
        super().__init__(message, ErrorCode.GIT_COMMAND_FAILED, context)


class ConfigLoadError(MLScannerError):
    """Exception raised when configuration loading fails."""
    
    def __init__(self, message: str, context: dict = None):
        super().__init__(message, ErrorCode.CONFIG_LOAD_FAILED, context)


class ConfigValidationError(MLScannerError):
    """Exception raised when configuration validation fails."""
    
    def __init__(self, message: str, context: dict = None):
        super().__init__(message, ErrorCode.CONFIG_VALIDATION_FAILED, context)


class PreprocessingError(MLScannerError):
    """Exception raised when data preprocessing fails."""
    
    def __init__(self, message: str, context: dict = None):
        super().__init__(message, ErrorCode.DATASET_LOAD_FAILED, context)
