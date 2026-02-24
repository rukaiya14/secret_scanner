"""
Unit tests for MLScanner class.

Tests the high-level ML scanner interface including file scanning,
text scanning, result formatting, and availability checking.
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from ml_scanner.ml_scanner import MLScanner
from ml_scanner.inference_engine import InferenceEngine
from ml_scanner.models import Detection, Finding
from ml_scanner.exceptions import InferenceTimeoutError, ModelLoadError


@pytest.fixture
def mock_inference_engine():
    """Create a mock inference engine."""
    engine = Mock(spec=InferenceEngine)
    engine.predict = Mock(return_value=[])
    engine.get_metrics = Mock(return_value={
        "total_inferences": 0,
        "avg_latency_ms": 0.0,
    })
    return engine


@pytest.fixture
def ml_scanner(mock_inference_engine):
    """Create an MLScanner instance with mock inference engine."""
    config = {
        "timeout": 5,
        "confidence_threshold": 0.5,
        "enabled_categories": ["API_KEY", "PASSWORD", "PII", "TOKEN", "CERTIFICATE", "OTHER"],
    }
    return MLScanner(config, mock_inference_engine)


class TestMLScannerInitialization:
    """Test MLScanner initialization."""
    
    def test_init_with_default_config(self, mock_inference_engine):
        """Test initialization with default configuration."""
        config = {}
        scanner = MLScanner(config, mock_inference_engine)
        
        assert scanner.default_timeout == 5
        assert scanner.confidence_threshold == 0.5
        assert scanner.is_available() is True
    
    def test_init_with_custom_config(self, mock_inference_engine):
        """Test initialization with custom configuration."""
        config = {
            "timeout": 10,
            "confidence_threshold": 0.8,
            "enabled_categories": ["API_KEY", "PASSWORD"],
        }
        scanner = MLScanner(config, mock_inference_engine)
        
        assert scanner.default_timeout == 10
        assert scanner.confidence_threshold == 0.8
        assert scanner.enabled_categories == ["API_KEY", "PASSWORD"]


class TestScanText:
    """Test text scanning functionality."""
    
    def test_scan_empty_text(self, ml_scanner):
        """Test scanning empty text returns no findings."""
        findings = ml_scanner.scan_text("")
        
        assert findings == []
    
    def test_scan_text_with_detection(self, ml_scanner, mock_inference_engine):
        """Test scanning text with ML detection."""
        # Setup mock detection
        detection = Detection(
            text="api_key = 'AKIAIOSFODNN7EXAMPLE'",
            start_pos=10,
            end_pos=31,
            category="API_KEY",
            confidence_score=0.92,
            token_attributions=None,
        )
        mock_inference_engine.predict.return_value = [detection]
        
        # Scan text
        text = "api_key = 'AKIAIOSFODNN7EXAMPLE'"
        findings = ml_scanner.scan_text(text, "test.py")
        
        # Verify
        assert len(findings) == 1
        assert findings[0].secret_type == "API_KEY"
        assert findings[0].confidence_score == 0.92
        assert findings[0].source == "ML_ONLY"
        assert findings[0].file_path == "test.py"
    
    def test_scan_text_filters_by_confidence(self, ml_scanner, mock_inference_engine):
        """Test that low confidence detections are filtered out."""
        # Setup mock detection with low confidence
        detection = Detection(
            text="test",
            start_pos=0,
            end_pos=4,
            category="API_KEY",
            confidence_score=0.3,  # Below threshold
            token_attributions=None,
        )
        mock_inference_engine.predict.return_value = [detection]
        
        # Scan text
        findings = ml_scanner.scan_text("test", "test.py")
        
        # Should be filtered out
        assert len(findings) == 0
    
    def test_scan_text_filters_by_category(self, ml_scanner, mock_inference_engine):
        """Test that disabled categories are filtered out."""
        # Configure scanner with limited categories
        ml_scanner.enabled_categories = ["API_KEY"]
        
        # Setup mock detection with disabled category
        detection = Detection(
            text="test",
            start_pos=0,
            end_pos=4,
            category="PASSWORD",  # Not in enabled_categories
            confidence_score=0.9,
            token_attributions=None,
        )
        mock_inference_engine.predict.return_value = [detection]
        
        # Scan text
        findings = ml_scanner.scan_text("test", "test.py")
        
        # Should be filtered out
        assert len(findings) == 0
    
    def test_scan_text_handles_inference_timeout(self, ml_scanner, mock_inference_engine):
        """Test handling of inference timeout."""
        mock_inference_engine.predict.side_effect = InferenceTimeoutError(
            "Timeout",
            context={}
        )
        
        with pytest.raises(InferenceTimeoutError):
            ml_scanner.scan_text("test", "test.py")
        
        # Scanner should still be available after timeout
        assert ml_scanner.is_available() is True
    
    def test_scan_text_handles_model_load_error(self, ml_scanner, mock_inference_engine):
        """Test handling of model load error."""
        mock_inference_engine.predict.side_effect = ModelLoadError(
            "Failed to load model",
            context={}
        )
        
        with pytest.raises(ModelLoadError):
            ml_scanner.scan_text("test", "test.py")
        
        # Scanner should be marked unavailable
        assert ml_scanner.is_available() is False
        assert ml_scanner.get_last_error() == "Failed to load model"


class TestScanFile:
    """Test file scanning functionality."""
    
    def test_scan_file_success(self, ml_scanner, mock_inference_engine):
        """Test successful file scanning."""
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.py') as f:
            f.write("api_key = 'AKIAIOSFODNN7EXAMPLE'")
            temp_path = f.name
        
        try:
            # Setup mock detection
            detection = Detection(
                text="api_key = 'AKIAIOSFODNN7EXAMPLE'",
                start_pos=10,
                end_pos=31,
                category="API_KEY",
                confidence_score=0.92,
                token_attributions=None,
            )
            mock_inference_engine.predict.return_value = [detection]
            
            # Scan file
            findings = ml_scanner.scan_file(temp_path)
            
            # Verify
            assert len(findings) == 1
            assert findings[0].secret_type == "API_KEY"
            assert findings[0].file_path == temp_path
        finally:
            # Cleanup
            Path(temp_path).unlink()
    
    def test_scan_file_not_found(self, ml_scanner):
        """Test scanning non-existent file."""
        with pytest.raises(FileNotFoundError):
            ml_scanner.scan_file("nonexistent.py")
    
    def test_scan_file_with_custom_timeout(self, ml_scanner, mock_inference_engine):
        """Test file scanning with custom timeout."""
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.py') as f:
            f.write("test")
            temp_path = f.name
        
        try:
            mock_inference_engine.predict.return_value = []
            
            # Scan with custom timeout
            findings = ml_scanner.scan_file(temp_path, timeout=10)
            
            assert findings == []
        finally:
            Path(temp_path).unlink()
    
    def test_scan_file_handles_encoding_error(self, ml_scanner, mock_inference_engine):
        """Test handling of file encoding errors."""
        # Create file with latin-1 encoding
        with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.py') as f:
            f.write(b"test \xe9")  # Latin-1 encoded character
            temp_path = f.name
        
        try:
            mock_inference_engine.predict.return_value = []
            
            # Should handle encoding gracefully
            findings = ml_scanner.scan_file(temp_path)
            
            assert findings == []
        finally:
            Path(temp_path).unlink()


class TestResultFormatting:
    """Test detection to finding conversion."""
    
    def test_convert_detection_to_finding(self, ml_scanner):
        """Test conversion of Detection to Finding."""
        detection = Detection(
            text="api_key = 'AKIAIOSFODNN7EXAMPLE'",
            start_pos=10,
            end_pos=31,
            category="API_KEY",
            confidence_score=0.92,
            token_attributions=None,
        )
        
        text = "api_key = 'AKIAIOSFODNN7EXAMPLE'"
        findings = ml_scanner._convert_detections_to_findings(
            [detection],
            text,
            "test.py"
        )
        
        assert len(findings) == 1
        finding = findings[0]
        assert finding.file_path == "test.py"
        assert finding.line_number == 1
        assert finding.secret_type == "API_KEY"
        assert finding.confidence == "HIGH"  # 0.92 >= 0.85
        assert finding.confidence_score == 0.92
        assert finding.source == "ML_ONLY"
    
    def test_convert_detection_low_confidence(self, ml_scanner):
        """Test conversion with low confidence detection."""
        detection = Detection(
            text="test",
            start_pos=0,
            end_pos=4,
            category="API_KEY",
            confidence_score=0.7,  # < 0.85
            token_attributions=None,
        )
        
        findings = ml_scanner._convert_detections_to_findings(
            [detection],
            "test",
            "test.py"
        )
        
        assert len(findings) == 1
        assert findings[0].confidence == "LOW"
    
    def test_calculate_line_number(self, ml_scanner):
        """Test line number calculation."""
        detection = Detection(
            text="line1\nline2\napi_key = 'test'",
            start_pos=12,  # Start of line 3
            end_pos=16,
            category="API_KEY",
            confidence_score=0.9,
            token_attributions=None,
        )
        
        text = "line1\nline2\napi_key = 'test'"
        findings = ml_scanner._convert_detections_to_findings(
            [detection],
            text,
            "test.py"
        )
        
        assert findings[0].line_number == 3


class TestAvailability:
    """Test availability checking."""
    
    def test_is_available_initially_true(self, ml_scanner):
        """Test that scanner is initially available."""
        assert ml_scanner.is_available() is True
    
    def test_is_available_after_error(self, ml_scanner, mock_inference_engine):
        """Test availability after error."""
        mock_inference_engine.predict.side_effect = Exception("Test error")
        
        try:
            ml_scanner.scan_text("test")
        except Exception:
            pass
        
        assert ml_scanner.is_available() is False
    
    def test_get_last_error(self, ml_scanner, mock_inference_engine):
        """Test getting last error message."""
        mock_inference_engine.predict.side_effect = Exception("Test error")
        
        try:
            ml_scanner.scan_text("test")
        except Exception:
            pass
        
        assert ml_scanner.get_last_error() == "Test error"
    
    def test_get_stats(self, ml_scanner, mock_inference_engine):
        """Test getting scanner statistics."""
        stats = ml_scanner.get_stats()
        
        assert "available" in stats
        assert "last_error" in stats
        assert "inference_metrics" in stats
        assert stats["available"] is True


class TestCategoryClassification:
    """Test detection category classification."""
    
    def test_classify_api_key(self, ml_scanner, mock_inference_engine):
        """Test API_KEY category classification."""
        detection = Detection(
            text="AKIAIOSFODNN7EXAMPLE",
            start_pos=0,
            end_pos=20,
            category="API_KEY",
            confidence_score=0.95,
            token_attributions=None,
        )
        mock_inference_engine.predict.return_value = [detection]
        
        findings = ml_scanner.scan_text("AKIAIOSFODNN7EXAMPLE")
        
        assert len(findings) == 1
        assert findings[0].secret_type == "API_KEY"
    
    def test_classify_password(self, ml_scanner, mock_inference_engine):
        """Test PASSWORD category classification."""
        detection = Detection(
            text="password123",
            start_pos=0,
            end_pos=11,
            category="PASSWORD",
            confidence_score=0.88,
            token_attributions=None,
        )
        mock_inference_engine.predict.return_value = [detection]
        
        findings = ml_scanner.scan_text("password123")
        
        assert len(findings) == 1
        assert findings[0].secret_type == "PASSWORD"
    
    def test_classify_multiple_categories(self, ml_scanner, mock_inference_engine):
        """Test classification of multiple categories."""
        detections = [
            Detection(
                text="AKIA123",
                start_pos=0,
                end_pos=7,
                category="API_KEY",
                confidence_score=0.9,
                token_attributions=None,
            ),
            Detection(
                text="pass123",
                start_pos=8,
                end_pos=15,
                category="PASSWORD",
                confidence_score=0.85,
                token_attributions=None,
            ),
        ]
        mock_inference_engine.predict.return_value = detections
        
        findings = ml_scanner.scan_text("AKIA123 pass123")
        
        assert len(findings) == 2
        assert findings[0].secret_type == "API_KEY"
        assert findings[1].secret_type == "PASSWORD"


class TestEntropyCalculation:
    """Test entropy calculation."""
    
    def test_calculate_entropy_empty_string(self, ml_scanner):
        """Test entropy of empty string."""
        entropy = ml_scanner._calculate_entropy("")
        assert entropy == 0.0
    
    def test_calculate_entropy_single_char(self, ml_scanner):
        """Test entropy of single repeated character."""
        entropy = ml_scanner._calculate_entropy("aaaa")
        assert entropy == 0.0
    
    def test_calculate_entropy_varied_chars(self, ml_scanner):
        """Test entropy of varied characters."""
        entropy = ml_scanner._calculate_entropy("abcdefgh")
        assert entropy > 0.0


class TestCommentDetection:
    """Test comment detection."""
    
    def test_is_in_comment_single_line(self, ml_scanner):
        """Test detection of single-line comment."""
        text = "// This is a comment"
        assert ml_scanner._is_in_comment(text, 5) is True
    
    def test_is_in_comment_hash(self, ml_scanner):
        """Test detection of hash comment."""
        text = "# This is a comment"
        assert ml_scanner._is_in_comment(text, 5) is True
    
    def test_is_not_in_comment(self, ml_scanner):
        """Test detection of non-comment."""
        text = "api_key = 'test'"
        assert ml_scanner._is_in_comment(text, 5) is False
