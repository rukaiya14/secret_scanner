"""
Integration test for ResultMerger demonstrating end-to-end functionality.

This test demonstrates the complete merging workflow with realistic scenarios.
"""

import pytest
from ml_scanner.hybrid_detector import ResultMerger
from ml_scanner.models import Finding


def test_result_merger_realistic_scenario():
    """
    Test ResultMerger with a realistic scenario containing:
    - Overlapping detections (both scanners)
    - ML-only detections (above and below threshold)
    - Regex-only detections
    """
    # Arrange: Create realistic findings from both scanners
    regex_findings = [
        # AWS key detected by regex
        Finding(
            file_path="config.py",
            line_number=15,
            matched_text="AKIAIOSFODNN7EXAMPLE",
            secret_type="AWS Access Key",
            confidence="HIGH",
            entropy=4.5,
            is_comment=False,
            source="REGEX_ONLY"
        ),
        # Password detected by regex
        Finding(
            file_path="auth.py",
            line_number=42,
            matched_text="password123",
            secret_type="Password",
            confidence="HIGH",
            entropy=3.2,
            is_comment=False,
            source="REGEX_ONLY"
        ),
        # GitHub token detected by regex
        Finding(
            file_path="deploy.sh",
            line_number=8,
            matched_text="ghp_1234567890abcdefghijklmnopqrstuv",
            secret_type="GitHub Token",
            confidence="HIGH",
            entropy=4.8,
            is_comment=False,
            source="REGEX_ONLY"
        )
    ]
    
    ml_findings = [
        # AWS key also detected by ML (overlapping)
        Finding(
            file_path="config.py",
            line_number=15,
            matched_text="AKIAIOSFODNN7EXAMPLE",
            secret_type="AWS Access Key",
            confidence="HIGH",
            entropy=4.5,
            is_comment=False,
            source="ML_ONLY",
            confidence_score=0.96
        ),
        # GitHub token also detected by ML (overlapping)
        Finding(
            file_path="deploy.sh",
            line_number=8,
            matched_text="ghp_1234567890abcdefghijklmnopqrstuv",
            secret_type="GitHub Token",
            confidence="HIGH",
            entropy=4.8,
            is_comment=False,
            source="ML_ONLY",
            confidence_score=0.92
        ),
        # API key detected only by ML (high confidence)
        Finding(
            file_path="api_client.py",
            line_number=23,
            matched_text="sk_live_51H7xYzABC123DEF456",
            secret_type="Stripe API Key",
            confidence="HIGH",
            entropy=4.6,
            is_comment=False,
            source="ML_ONLY",
            confidence_score=0.89
        ),
        # Potential secret detected only by ML (low confidence - should be filtered)
        Finding(
            file_path="utils.py",
            line_number=67,
            matched_text="maybe_a_secret",
            secret_type="Unknown",
            confidence="LOW",
            entropy=3.0,
            is_comment=False,
            source="ML_ONLY",
            confidence_score=0.72
        )
    ]
    
    # Act: Merge findings
    merger = ResultMerger(ml_only_threshold=0.85)
    merged = merger.merge(regex_findings, ml_findings)
    
    # Assert: Verify correct merging behavior
    # Expected: 2 HIGH_CONFIDENCE (AWS + GitHub) + 1 ML_ONLY (Stripe) + 1 REGEX_ONLY (password) = 4
    assert len(merged) == 4, "Should have 4 findings after merging"
    
    # Check HIGH_CONFIDENCE findings (detected by both)
    high_confidence = [f for f in merged if f.source == "HIGH_CONFIDENCE"]
    assert len(high_confidence) == 2, "Should have 2 HIGH_CONFIDENCE findings"
    
    # Verify AWS key is HIGH_CONFIDENCE with ML confidence score
    aws_key = next(f for f in merged if "AKIA" in f.matched_text)
    assert aws_key.source == "HIGH_CONFIDENCE"
    assert aws_key.confidence_score == 0.96
    
    # Verify GitHub token is HIGH_CONFIDENCE
    github_token = next(f for f in merged if "ghp_" in f.matched_text)
    assert github_token.source == "HIGH_CONFIDENCE"
    assert github_token.confidence_score == 0.92
    
    # Check ML_ONLY findings (above threshold)
    ml_only = [f for f in merged if f.source == "ML_ONLY"]
    assert len(ml_only) == 1, "Should have 1 ML_ONLY finding"
    assert ml_only[0].matched_text == "sk_live_51H7xYzABC123DEF456"
    assert ml_only[0].confidence_score >= 0.85
    
    # Check REGEX_ONLY findings
    regex_only = [f for f in merged if f.source == "REGEX_ONLY"]
    assert len(regex_only) == 1, "Should have 1 REGEX_ONLY finding (password)"
    
    # Verify password is REGEX_ONLY
    password = next(f for f in merged if "password" in f.matched_text)
    assert password.source == "REGEX_ONLY"
    
    # Verify low confidence ML detection was filtered out
    assert not any("maybe_a_secret" in f.matched_text for f in merged), \
        "Low confidence ML-only detection should be filtered out"


def test_result_merger_deduplication():
    """Test that ResultMerger properly deduplicates findings."""
    # Arrange: Create duplicate findings
    findings = [
        Finding(
            file_path="test.py",
            line_number=10,
            matched_text="secret123",
            secret_type="API Key",
            confidence="HIGH",
            entropy=4.0,
            is_comment=False,
            source="REGEX_ONLY"
        ),
        Finding(
            file_path="test.py",
            line_number=10,
            matched_text="secret123",
            secret_type="API Key",
            confidence="HIGH",
            entropy=4.0,
            is_comment=False,
            source="REGEX_ONLY"
        ),
        Finding(
            file_path="test.py",
            line_number=20,
            matched_text="secret456",
            secret_type="API Key",
            confidence="HIGH",
            entropy=4.0,
            is_comment=False,
            source="REGEX_ONLY"
        )
    ]
    
    # Act: Deduplicate
    merger = ResultMerger()
    deduplicated = merger.deduplicate(findings)
    
    # Assert: Verify deduplication
    assert len(deduplicated) == 2, "Should remove duplicate finding"
    assert deduplicated[0].matched_text == "secret123"
    assert deduplicated[1].matched_text == "secret456"


def test_result_merger_confidence_threshold_boundary():
    """Test ResultMerger behavior at confidence threshold boundary."""
    # Arrange: Create ML findings at threshold boundary
    regex_findings = []
    
    ml_findings = [
        # Exactly at threshold (should be included)
        Finding(
            file_path="test1.py",
            line_number=10,
            matched_text="token_at_threshold",
            secret_type="Token",
            confidence="HIGH",
            entropy=4.0,
            is_comment=False,
            source="ML_ONLY",
            confidence_score=0.85
        ),
        # Just below threshold (should be filtered)
        Finding(
            file_path="test2.py",
            line_number=20,
            matched_text="token_below_threshold",
            secret_type="Token",
            confidence="LOW",
            entropy=3.5,
            is_comment=False,
            source="ML_ONLY",
            confidence_score=0.849
        ),
        # Just above threshold (should be included)
        Finding(
            file_path="test3.py",
            line_number=30,
            matched_text="token_above_threshold",
            secret_type="Token",
            confidence="HIGH",
            entropy=4.2,
            is_comment=False,
            source="ML_ONLY",
            confidence_score=0.851
        )
    ]
    
    # Act: Merge with default threshold (0.85)
    merger = ResultMerger()
    merged = merger.merge(regex_findings, ml_findings)
    
    # Assert: Verify threshold behavior
    assert len(merged) == 2, "Should include findings at or above threshold"
    assert any("at_threshold" in f.matched_text for f in merged)
    assert any("above_threshold" in f.matched_text for f in merged)
    assert not any("below_threshold" in f.matched_text for f in merged)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
