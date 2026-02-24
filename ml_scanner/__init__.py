"""
ML-Enhanced Secret Scanner with CodeBERT

This package provides machine learning-based secret and PII detection capabilities
using CodeBERT transformers, complementing the existing regex-based pattern matching system.
"""

__version__ = "1.0.0"

from ml_scanner.models import (
    Finding,
    Detection,
    Explanation,
    ModelMetadata,
    ScanResult,
    Dataset,
)
from ml_scanner.exceptions import (
    MLScannerError,
    ErrorCode,
)

# Conditionally import components that require heavy dependencies
try:
    from ml_scanner.performance_validator import PerformanceValidator
    _has_performance_validator = True
except ImportError:
    _has_performance_validator = False
    PerformanceValidator = None

try:
    from ml_scanner.code_tokenizer import CodeTokenizer
    _has_code_tokenizer = True
except ImportError:
    _has_code_tokenizer = False
    CodeTokenizer = None

try:
    from ml_scanner.explainer import Explainer, SHAPExplainer, LIMEExplainer
    _has_explainer = True
except ImportError:
    _has_explainer = False
    Explainer = None
    SHAPExplainer = None
    LIMEExplainer = None

__all__ = [
    "Finding",
    "Detection",
    "Explanation",
    "ModelMetadata",
    "ScanResult",
    "Dataset",
    "MLScannerError",
    "ErrorCode",
]

if _has_performance_validator:
    __all__.append("PerformanceValidator")

if _has_code_tokenizer:
    __all__.append("CodeTokenizer")

if _has_explainer:
    __all__.extend(["Explainer", "SHAPExplainer", "LIMEExplainer"])
