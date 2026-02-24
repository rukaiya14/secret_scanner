"""
Core data models for the ML-Enhanced Secret Scanner.

This module defines all data structures used throughout the system,
including findings, detections, explanations, and metadata.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict, Tuple


@dataclass
class Finding:
    """
    Represents a detected secret (unified for regex and ML).
    
    This is the primary output format for both regex-based and ML-based detection,
    providing a consistent interface for downstream consumers.
    """
    file_path: str
    line_number: int
    matched_text: str
    secret_type: str
    confidence: str  # 'HIGH' or 'LOW'
    entropy: float
    is_comment: bool
    source: str  # 'REGEX_ONLY', 'ML_ONLY', 'HIGH_CONFIDENCE'
    confidence_score: Optional[float] = None  # ML confidence (0.0-1.0)
    explanation: Optional['Explanation'] = None  # ML explanation


@dataclass
class Detection:
    """
    ML-specific detection before conversion to Finding.
    
    This represents the raw output from the ML model before it's
    converted to the unified Finding format.
    """
    text: str
    start_pos: int
    end_pos: int
    category: str  # API_KEY, PASSWORD, PII, TOKEN, CERTIFICATE, OTHER
    confidence_score: float  # 0.0 to 1.0
    token_attributions: Optional[List[Tuple[str, float]]] = None


@dataclass
class Explanation:
    """
    Explanation for an ML detection using SHAP or LIME.
    
    Provides interpretability by showing which tokens in the code
    contributed most to the detection decision.
    """
    detection: Detection
    token_attributions: List[Tuple[str, float]]  # (token, attribution_score)
    top_tokens: List[str]  # Top 5 most influential tokens
    formatted_text: str  # Human-readable explanation with highlights


@dataclass
class ModelMetadata:
    """
    Metadata for a trained model version.
    
    Tracks all information needed for model versioning, lineage,
    and performance monitoring.
    """
    version: str  # Semantic version (MAJOR.MINOR.PATCH)
    training_date: datetime
    dataset_version: str
    performance_metrics: Dict[str, float]  # precision, recall, f1_score
    environment: str  # STAGING or PRODUCTION
    training_config: Dict[str, any]
    lineage: Dict[str, any]  # Links to training datasets


@dataclass
class ScanResult:
    """
    Aggregated scan results from hybrid detection.
    
    Provides summary statistics and all findings from a scan operation.
    """
    findings: List[Finding]
    files_scanned: int
    high_confidence_count: int
    low_confidence_count: int
    execution_time_seconds: float
    ml_available: bool
    fallback_mode: bool = False


@dataclass
class Dataset:
    """
    Training dataset with metadata.
    
    Represents a collection of labeled examples for training or evaluation,
    with provenance tracking for reproducibility.
    """
    texts: List[str]
    labels: List[int]
    metadata: Dict[str, any]
    source: str  # 'github' or 'kaggle'
    version: str
    checksum: str


@dataclass
class TrainingMetrics:
    """
    Training performance metrics.
    
    Captures model performance during training for monitoring and comparison.
    """
    precision: float
    recall: float
    f1_score: float
    confusion_matrix: List[List[int]]
    loss: float
    epoch: int


@dataclass
class FeedbackEntry:
    """
    False positive feedback from users.
    
    Stores user feedback on detections to improve model training
    and track false positive rates.
    """
    finding_id: str
    file_path: str
    matched_text: str
    secret_type: str
    detection_source: str  # REGEX_ONLY, ML_ONLY, HIGH_CONFIDENCE
    timestamp: datetime
    user: str
    reason: str
    confidence_score: Optional[float] = None
