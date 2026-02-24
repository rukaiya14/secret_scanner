"""
Unit tests for PerformanceValidator class.

Tests the threshold checking and report generation functionality
of the PerformanceValidator without requiring torch/transformers.
"""

import pytest
from unittest.mock import Mock

from ml_scanner.performance_validator import PerformanceValidator


@pytest.fixture
def mock_tokenizer():
    """Fixture providing a mock tokenizer."""
    tokenizer = Mock()
    tokenizer.return_value = {
        'input_ids': Mock(),
        'attention_mask': Mock()
    }
    return tokenizer


def test_performance_validator_initialization(mock_tokenizer):
    """Test PerformanceValidator initialization."""
    validator = PerformanceValidator(mock_tokenizer)
    
    assert validator.tokenizer is not None
    assert validator.device is not None


def test_performance_validator_initialization_with_device(mock_tokenizer):
    """Test PerformanceValidator initialization with explicit device."""
    validator = PerformanceValidator(mock_tokenizer, device="cpu")
    
    # Device should be set to cpu
    assert str(validator.device) == "cpu"


def test_meets_threshold_with_passing_metrics(mock_tokenizer):
    """Test meets_threshold returns True for metrics above thresholds."""
    validator = PerformanceValidator(mock_tokenizer)
    
    metrics = {
        "precision": 0.95,
        "recall": 0.90,
        "f1_score": 0.92,
        "confusion_matrix": [[80, 5], [10, 105]]
    }
    
    assert validator.meets_threshold(metrics) is True


def test_meets_threshold_with_failing_precision(mock_tokenizer):
    """Test meets_threshold returns False when precision is below threshold."""
    validator = PerformanceValidator(mock_tokenizer)
    
    metrics = {
        "precision": 0.85,  # Below 0.90 threshold
        "recall": 0.90,
        "f1_score": 0.87,
        "confusion_matrix": [[70, 15], [10, 105]]
    }
    
    assert validator.meets_threshold(metrics) is False


def test_meets_threshold_with_failing_recall(mock_tokenizer):
    """Test meets_threshold returns False when recall is below threshold."""
    validator = PerformanceValidator(mock_tokenizer)
    
    metrics = {
        "precision": 0.95,
        "recall": 0.80,  # Below 0.85 threshold
        "f1_score": 0.87,
        "confusion_matrix": [[80, 5], [20, 95]]
    }
    
    assert validator.meets_threshold(metrics) is False


def test_meets_threshold_with_custom_thresholds(mock_tokenizer):
    """Test meets_threshold with custom threshold values."""
    validator = PerformanceValidator(mock_tokenizer)
    
    metrics = {
        "precision": 0.85,
        "recall": 0.80,
        "f1_score": 0.82,
        "confusion_matrix": [[70, 15], [20, 95]]
    }
    
    # Should pass with lower thresholds
    assert validator.meets_threshold(metrics, min_precision=0.80, min_recall=0.75) is True
    
    # Should fail with higher thresholds
    assert validator.meets_threshold(metrics, min_precision=0.90, min_recall=0.85) is False


def test_meets_threshold_at_exact_threshold(mock_tokenizer):
    """Test meets_threshold with metrics exactly at threshold."""
    validator = PerformanceValidator(mock_tokenizer)
    
    metrics = {
        "precision": 0.90,  # Exactly at threshold
        "recall": 0.85,     # Exactly at threshold
        "f1_score": 0.87,
        "confusion_matrix": [[80, 10], [15, 95]]
    }
    
    assert validator.meets_threshold(metrics) is True


def test_generate_report_format(mock_tokenizer):
    """Test that generate_report produces a properly formatted report."""
    validator = PerformanceValidator(mock_tokenizer)
    
    metrics = {
        "precision": 0.95,
        "recall": 0.90,
        "f1_score": 0.92,
        "confusion_matrix": [[80, 5], [10, 105]]
    }
    
    report = validator.generate_report(
        metrics,
        model_version="1.0.0",
        dataset_info="test_dataset"
    )
    
    # Check report contains key information
    assert "MODEL PERFORMANCE REPORT" in report
    assert "1.0.0" in report
    assert "test_dataset" in report
    assert "95.00%" in report  # Precision
    assert "90.00%" in report  # Recall
    assert "92.00%" in report  # F1-score
    assert "CONFUSION MATRIX" in report
    assert "80" in report  # TN
    assert "5" in report   # FP
    assert "10" in report  # FN
    assert "105" in report # TP


def test_generate_report_shows_pass_fail_indicators(mock_tokenizer):
    """Test that generate_report shows PASS/FAIL indicators correctly."""
    validator = PerformanceValidator(mock_tokenizer)
    
    # Passing metrics
    passing_metrics = {
        "precision": 0.95,
        "recall": 0.90,
        "f1_score": 0.92,
        "confusion_matrix": [[80, 5], [10, 105]]
    }
    
    passing_report = validator.generate_report(passing_metrics)
    assert "✓ PASS" in passing_report
    
    # Failing metrics
    failing_metrics = {
        "precision": 0.85,
        "recall": 0.80,
        "f1_score": 0.82,
        "confusion_matrix": [[70, 15], [20, 95]]
    }
    
    failing_report = validator.generate_report(failing_metrics)
    assert "✗ FAIL" in failing_report


def test_meets_threshold_with_missing_metrics(mock_tokenizer):
    """Test meets_threshold handles missing metric keys gracefully."""
    validator = PerformanceValidator(mock_tokenizer)
    
    # Missing precision and recall
    incomplete_metrics = {
        "f1_score": 0.92,
        "confusion_matrix": [[80, 5], [10, 105]]
    }
    
    # Should default to 0.0 for missing metrics and return False
    assert validator.meets_threshold(incomplete_metrics) is False


def test_generate_report_with_zero_confusion_matrix(mock_tokenizer):
    """Test generate_report handles zero confusion matrix."""
    validator = PerformanceValidator(mock_tokenizer)
    
    metrics = {
        "precision": 0.0,
        "recall": 0.0,
        "f1_score": 0.0,
        "confusion_matrix": [[0, 0], [0, 0]]
    }
    
    report = validator.generate_report(metrics)
    
    # Should not crash and should contain the report structure
    assert "MODEL PERFORMANCE REPORT" in report
    assert "0.00%" in report


def test_generate_report_calculates_accuracy(mock_tokenizer):
    """Test that generate_report correctly calculates accuracy."""
    validator = PerformanceValidator(mock_tokenizer)
    
    # TN=80, FP=5, FN=10, TP=105
    # Accuracy = (TP + TN) / (TP + TN + FP + FN) = (105 + 80) / 200 = 0.925
    metrics = {
        "precision": 0.95,
        "recall": 0.90,
        "f1_score": 0.92,
        "confusion_matrix": [[80, 5], [10, 105]]
    }
    
    report = validator.generate_report(metrics)
    
    # Check accuracy is displayed
    assert "92.50%" in report  # Accuracy


def test_meets_threshold_with_edge_case_values(mock_tokenizer):
    """Test meets_threshold with edge case metric values."""
    validator = PerformanceValidator(mock_tokenizer)
    
    # Perfect metrics
    perfect_metrics = {
        "precision": 1.0,
        "recall": 1.0,
        "f1_score": 1.0,
        "confusion_matrix": [[100, 0], [0, 100]]
    }
    assert validator.meets_threshold(perfect_metrics) is True
    
    # Zero metrics
    zero_metrics = {
        "precision": 0.0,
        "recall": 0.0,
        "f1_score": 0.0,
        "confusion_matrix": [[0, 100], [100, 0]]
    }
    assert validator.meets_threshold(zero_metrics) is False
