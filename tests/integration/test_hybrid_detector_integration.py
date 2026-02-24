"""
Integration tests for HybridDetector.

Tests the complete hybrid detection workflow with both scanners.
"""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, MagicMock

from ml_scanner.hybrid_detector import HybridDetector
from ml_scanner.models import Finding


class TestHybridDetectorIntegration:
    """Integration tests for HybridDetector."""
    
    def test_scan_file_with_both_scanners(self):
        """Test scanning a file with both regex and ML scanners."""
        # Create a temporary test file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write('api_key = "AKIAIOSFODNN7EXAMPLE"\n')
            f.write('password = "super_secret_123"\n')
            test_file = f.name
        
        try:
            # Mock regex scanner
            regex_scanner = Mock()
            regex_scanner.scan_file.return_value = [
                Finding(
                    file_path=test_file,
                    line_number=1,
                    matched_text="AKIAIOSFODNN7EXAMPLE",
                    secret_type="AWS Access Key",
                    confidence="HIGH",
                    entropy=4.5,
                    is_comment=False,
                    source="REGEX_ONLY"
                )
            ]
            
            # Mock ML scanner
            ml_scanner = Mock()
            ml_scanner.is_available.return_value = True
            ml_scanner.scan_file.return_value = [
                Finding(
                    file_path=test_file,
                    line_number=1,
                    matched_text="AKIAIOSFODNN7EXAMPLE",
                    secret_type="AWS Access Key",
                    confidence="HIGH",
                    entropy=4.5,
                    is_comment=False,
                    source="ML_ONLY",
                    confidence_score=0.95
                ),
                Finding(
                    file_path=test_file,
                    line_number=2,
                    matched_text="super_secret_123",
                    secret_type="PASSWORD",
                    confidence="HIGH",
                    entropy=3.8,
                    is_comment=False,
                    source="ML_ONLY",
                    confidence_score=0.88
                )
            ]
            
            # Create HybridDetector
            detector = HybridDetector(
                regex_scanner=regex_scanner,
                ml_scanner=ml_scanner,
                config={"ml_only_threshold": 0.85}
            )
            
            # Scan file
            findings = detector.scan_file(test_file)
            
            # Verify results
            assert len(findings) == 2, "Should have 2 findings"
            
            # First finding should be HIGH_CONFIDENCE (detected by both)
            aws_finding = [f for f in findings if "AKIA" in f.matched_text][0]
            assert aws_finding.source == "HIGH_CONFIDENCE"
            assert aws_finding.confidence_score == 0.95
            
            # Second finding should be ML_ONLY (above threshold)
            password_finding = [f for f in findings if "super_secret" in f.matched_text][0]
            assert password_finding.source == "ML_ONLY"
            assert password_finding.confidence_score == 0.88
            
        finally:
            # Clean up
            os.unlink(test_file)
    
    def test_scan_files_batch(self):
        """Test batch scanning multiple files."""
        # Create temporary test files
        test_files = []
        for i in range(3):
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(f'secret_{i} = "test_secret_{i}"\n')
                test_files.append(f.name)
        
        try:
            # Mock regex scanner
            regex_scanner = Mock()
            regex_scanner.scan_file.return_value = [
                Finding(
                    file_path="test.py",
                    line_number=1,
                    matched_text="test_secret",
                    secret_type="Generic Secret",
                    confidence="HIGH",
                    entropy=4.0,
                    is_comment=False,
                    source="REGEX_ONLY"
                )
            ]
            
            # Mock ML scanner
            ml_scanner = Mock()
            ml_scanner.is_available.return_value = True
            ml_scanner.scan_file.return_value = []
            
            # Create HybridDetector
            detector = HybridDetector(
                regex_scanner=regex_scanner,
                ml_scanner=ml_scanner
            )
            
            # Scan files
            result = detector.scan_files(test_files)
            
            # Verify results
            assert result.files_scanned == 3
            assert len(result.findings) == 3  # One finding per file
            assert result.ml_available is True
            assert result.fallback_mode is False
            assert result.execution_time_seconds > 0
            
        finally:
            # Clean up
            for f in test_files:
                os.unlink(f)
    
    def test_fallback_mode_when_ml_unavailable(self):
        """Test that detector falls back to regex-only when ML is unavailable."""
        # Create a temporary test file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write('api_key = "AKIAIOSFODNN7EXAMPLE"\n')
            test_file = f.name
        
        try:
            # Mock regex scanner
            regex_scanner = Mock()
            regex_scanner.scan_file.return_value = [
                Finding(
                    file_path=test_file,
                    line_number=1,
                    matched_text="AKIAIOSFODNN7EXAMPLE",
                    secret_type="AWS Access Key",
                    confidence="HIGH",
                    entropy=4.5,
                    is_comment=False,
                    source="REGEX_ONLY"
                )
            ]
            
            # Mock ML scanner as unavailable
            ml_scanner = Mock()
            ml_scanner.is_available.return_value = False
            
            # Create HybridDetector
            detector = HybridDetector(
                regex_scanner=regex_scanner,
                ml_scanner=ml_scanner
            )
            
            # Scan file
            findings = detector.scan_file(test_file)
            
            # Verify results
            assert len(findings) == 1
            assert findings[0].source == "REGEX_ONLY"
            assert detector.is_in_fallback_mode() is True
            
        finally:
            # Clean up
            os.unlink(test_file)
    
    def test_resilient_scanning_continues_on_error(self):
        """Test that batch scanning continues even if individual files fail."""
        # Create temporary test files
        test_files = []
        for i in range(3):
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(f'secret_{i} = "test_secret_{i}"\n')
                test_files.append(f.name)
        
        try:
            # Mock regex scanner that fails on second file
            regex_scanner = Mock()
            call_count = [0]
            
            def scan_side_effect(file_path):
                call_count[0] += 1
                if call_count[0] == 2:
                    raise Exception("Simulated scan error")
                return [
                    Finding(
                        file_path=file_path,
                        line_number=1,
                        matched_text="test_secret",
                        secret_type="Generic Secret",
                        confidence="HIGH",
                        entropy=4.0,
                        is_comment=False,
                        source="REGEX_ONLY"
                    )
                ]
            
            regex_scanner.scan_file.side_effect = scan_side_effect
            
            # Mock ML scanner
            ml_scanner = Mock()
            ml_scanner.is_available.return_value = True
            ml_scanner.scan_file.return_value = []
            
            # Create HybridDetector
            detector = HybridDetector(
                regex_scanner=regex_scanner,
                ml_scanner=ml_scanner
            )
            
            # Scan files
            result = detector.scan_files(test_files)
            
            # Verify results - should have 2 findings (file 1 and 3, file 2 failed)
            assert result.files_scanned == 3
            assert len(result.findings) == 2  # Two successful scans
            
        finally:
            # Clean up
            for f in test_files:
                os.unlink(f)
    
    def test_is_in_fallback_mode(self):
        """Test fallback mode checking method."""
        # Mock scanners
        regex_scanner = Mock()
        ml_scanner = Mock()
        ml_scanner.is_available.return_value = True
        
        # Create HybridDetector
        detector = HybridDetector(
            regex_scanner=regex_scanner,
            ml_scanner=ml_scanner
        )
        
        # Initially not in fallback mode
        assert detector.is_in_fallback_mode() is False
        
        # Mark ML as unavailable
        detector.fallback_manager.mark_ml_unavailable("Test failure")
        
        # Should now be in fallback mode
        assert detector.is_in_fallback_mode() is True
        
        # Mark ML as available again
        detector.fallback_manager.mark_ml_available()
        
        # Should exit fallback mode
        assert detector.is_in_fallback_mode() is False
