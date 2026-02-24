"""
Unit tests for hybrid detector components.

Tests ResultMerger, FallbackManager, and HybridDetector classes.
"""

import pytest
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock, patch

from ml_scanner.hybrid_detector import ResultMerger, FallbackManager, HybridDetector
from ml_scanner.models import Finding, ScanResult


class TestResultMerger:
    """Tests for ResultMerger class."""
    
    def test_merge_overlapping_detections(self):
        """Test that overlapping detections are merged as HIGH_CONFIDENCE."""
        # Arrange
        regex_findings = [
            Finding(
                file_path="test.py",
                line_number=10,
                matched_text="AKIA123456789",
                secret_type="AWS Key",
                confidence="HIGH",
                entropy=4.5,
                is_comment=False,
                source="REGEX_ONLY"
            )
        ]
        
        ml_findings = [
            Finding(
                file_path="test.py",
                line_number=10,
                matched_text="AKIA123456789",
                secret_type="AWS Key",
                confidence="HIGH",
                entropy=4.5,
                is_comment=False,
                source="ML_ONLY",
                confidence_score=0.95
            )
        ]
        
        merger = ResultMerger()
        
        # Act
        merged = merger.merge(regex_findings, ml_findings)
        
        # Assert
        assert len(merged) == 1, "Should deduplicate overlapping detections"
        assert merged[0].source == "HIGH_CONFIDENCE", "Should mark as HIGH_CONFIDENCE"
        assert merged[0].confidence_score == 0.95, "Should preserve ML confidence score"
    
    def test_merge_ml_only_above_threshold(self):
        """Test that ML-only detections above threshold are included."""
        # Arrange
        regex_findings = []
        
        ml_findings = [
            Finding(
                file_path="test.py",
                line_number=20,
                matched_text="ghp_1234567890abcdef",
                secret_type="GitHub Token",
                confidence="HIGH",
                entropy=4.2,
                is_comment=False,
                source="ML_ONLY",
                confidence_score=0.90
            )
        ]
        
        merger = ResultMerger()
        
        # Act
        merged = merger.merge(regex_findings, ml_findings)
        
        # Assert
        assert len(merged) == 1, "Should include ML-only detection above threshold"
        assert merged[0].source == "ML_ONLY", "Should mark as ML_ONLY"
        assert merged[0].confidence_score == 0.90
    
    def test_merge_ml_only_below_threshold(self):
        """Test that ML-only detections below threshold are filtered out."""
        # Arrange
        regex_findings = []
        
        ml_findings = [
            Finding(
                file_path="test.py",
                line_number=30,
                matched_text="maybe_secret",
                secret_type="Unknown",
                confidence="LOW",
                entropy=3.0,
                is_comment=False,
                source="ML_ONLY",
                confidence_score=0.70
            )
        ]
        
        merger = ResultMerger()
        
        # Act
        merged = merger.merge(regex_findings, ml_findings)
        
        # Assert
        assert len(merged) == 0, "Should filter out ML-only detection below threshold"
    
    def test_merge_regex_only(self):
        """Test that regex-only detections are included."""
        # Arrange
        regex_findings = [
            Finding(
                file_path="test.py",
                line_number=40,
                matched_text="password123",
                secret_type="Password",
                confidence="HIGH",
                entropy=3.5,
                is_comment=False,
                source="REGEX_ONLY"
            )
        ]
        
        ml_findings = []
        
        merger = ResultMerger()
        
        # Act
        merged = merger.merge(regex_findings, ml_findings)
        
        # Assert
        assert len(merged) == 1, "Should include regex-only detection"
        assert merged[0].source == "REGEX_ONLY", "Should mark as REGEX_ONLY"
    
    def test_merge_uses_higher_confidence_score(self):
        """Test that merge uses higher confidence score when both detect."""
        # Arrange
        regex_findings = [
            Finding(
                file_path="test.py",
                line_number=50,
                matched_text="AKIA987654321",
                secret_type="AWS Key",
                confidence="HIGH",
                entropy=4.8,
                is_comment=False,
                source="REGEX_ONLY"
            )
        ]
        
        ml_findings = [
            Finding(
                file_path="test.py",
                line_number=50,
                matched_text="AKIA987654321",
                secret_type="AWS Key",
                confidence="HIGH",
                entropy=4.8,
                is_comment=False,
                source="ML_ONLY",
                confidence_score=0.95
            )
        ]
        
        merger = ResultMerger()
        
        # Act
        merged = merger.merge(regex_findings, ml_findings)
        
        # Assert
        assert len(merged) == 1
        assert merged[0].source == "HIGH_CONFIDENCE"
        # Should use ML finding since confidence > 0.9
        assert merged[0].confidence_score == 0.95
    
    def test_deduplicate_removes_duplicates(self):
        """Test that deduplicate removes duplicate findings."""
        # Arrange
        findings = [
            Finding(
                file_path="test.py",
                line_number=10,
                matched_text="secret1",
                secret_type="API Key",
                confidence="HIGH",
                entropy=4.0,
                is_comment=False,
                source="REGEX_ONLY"
            ),
            Finding(
                file_path="test.py",
                line_number=10,
                matched_text="secret1",
                secret_type="API Key",
                confidence="HIGH",
                entropy=4.0,
                is_comment=False,
                source="REGEX_ONLY"
            ),
            Finding(
                file_path="test.py",
                line_number=20,
                matched_text="secret2",
                secret_type="API Key",
                confidence="HIGH",
                entropy=4.0,
                is_comment=False,
                source="REGEX_ONLY"
            )
        ]
        
        merger = ResultMerger()
        
        # Act
        deduplicated = merger.deduplicate(findings)
        
        # Assert
        assert len(deduplicated) == 2, "Should remove one duplicate"
        assert deduplicated[0].matched_text == "secret1"
        assert deduplicated[1].matched_text == "secret2"
    
    def test_merge_custom_threshold(self):
        """Test that custom ML-only threshold is respected."""
        # Arrange
        regex_findings = []
        
        ml_findings = [
            Finding(
                file_path="test.py",
                line_number=60,
                matched_text="token123",
                secret_type="Token",
                confidence="HIGH",
                entropy=3.8,
                is_comment=False,
                source="ML_ONLY",
                confidence_score=0.88
            )
        ]
        
        # Use custom threshold of 0.90
        merger = ResultMerger(ml_only_threshold=0.90)
        
        # Act
        merged = merger.merge(regex_findings, ml_findings)
        
        # Assert
        assert len(merged) == 0, "Should filter out detection below custom threshold"