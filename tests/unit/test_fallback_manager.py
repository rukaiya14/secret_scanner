"""
Unit tests for FallbackManager class.

Tests ML Scanner availability tracking and fallback mode management.
"""

import pytest
import time
from datetime import datetime, timedelta
from unittest.mock import patch

from ml_scanner.hybrid_detector import FallbackManager


class TestFallbackManager:
    """Tests for FallbackManager class."""
    
    def test_initialization(self):
        """Test FallbackManager initializes with correct defaults."""
        # Act
        manager = FallbackManager()
        
        # Assert
        assert manager.ml_available is True, "Should start with ML available"
        assert manager.fallback_mode is False, "Should not start in fallback mode"
        assert manager.last_failure_time is None, "Should have no failure time initially"
        assert manager.last_reinit_attempt is None, "Should have no reinit attempt initially"
    
    def test_initialization_custom_interval(self):
        """Test FallbackManager accepts custom reinitialization interval."""
        # Act
        manager = FallbackManager(reinit_interval_seconds=600)
        
        # Assert
        assert manager.reinit_interval == timedelta(seconds=600)
    
    def test_mark_ml_unavailable(self):
        """Test marking ML Scanner as unavailable enters fallback mode."""
        # Arrange
        manager = FallbackManager()
        
        # Act
        manager.mark_ml_unavailable("Model load failed")
        
        # Assert
        assert manager.fallback_mode is True, "Should enter fallback mode"
        assert manager.ml_available is False, "Should mark ML as unavailable"
        assert manager.last_failure_time is not None, "Should record failure time"
    
    def test_mark_ml_unavailable_logs_warning(self):
        """Test that marking ML unavailable logs a warning."""
        # Arrange
        manager = FallbackManager()
        
        # Act & Assert
        with patch('ml_scanner.hybrid_detector.logger') as mock_logger:
            manager.mark_ml_unavailable("Test failure")
            mock_logger.warning.assert_called_once()
            call_args = mock_logger.warning.call_args
            assert "fallback mode" in call_args[0][0].lower()
            assert "Test failure" in call_args[0][0]
    
    def test_mark_ml_available(self):
        """Test marking ML Scanner as available exits fallback mode."""
        # Arrange
        manager = FallbackManager()
        manager.mark_ml_unavailable("Test failure")
        
        # Act
        manager.mark_ml_available()
        
        # Assert
        assert manager.fallback_mode is False, "Should exit fallback mode"
        assert manager.ml_available is True, "Should mark ML as available"
        assert manager.last_failure_time is None, "Should clear failure time"
    
    def test_mark_ml_available_logs_info(self):
        """Test that marking ML available logs an info message."""
        # Arrange
        manager = FallbackManager()
        manager.mark_ml_unavailable("Test failure")
        
        # Act & Assert
        with patch('ml_scanner.hybrid_detector.logger') as mock_logger:
            manager.mark_ml_available()
            mock_logger.info.assert_called_once()
            call_args = mock_logger.info.call_args
            assert "exiting fallback mode" in call_args[0][0].lower()
    
    def test_is_in_fallback_mode_returns_false_initially(self):
        """Test is_in_fallback_mode returns False initially."""
        # Arrange
        manager = FallbackManager()
        
        # Act & Assert
        assert manager.is_in_fallback_mode() is False
    
    def test_is_in_fallback_mode_returns_true_after_failure(self):
        """Test is_in_fallback_mode returns True after ML failure."""
        # Arrange
        manager = FallbackManager()
        manager.mark_ml_unavailable("Test failure")
        
        # Act & Assert
        assert manager.is_in_fallback_mode() is True
    
    def test_should_attempt_reinit_false_when_not_in_fallback(self):
        """Test should_attempt_reinit returns False when not in fallback mode."""
        # Arrange
        manager = FallbackManager()
        
        # Act & Assert
        assert manager.should_attempt_reinit() is False
    
    def test_should_attempt_reinit_true_on_first_attempt(self):
        """Test should_attempt_reinit returns True on first attempt after failure."""
        # Arrange
        manager = FallbackManager()
        manager.mark_ml_unavailable("Test failure")
        
        # Act & Assert
        assert manager.should_attempt_reinit() is True, "Should attempt reinit immediately after failure"
    
    def test_should_attempt_reinit_false_before_interval(self):
        """Test should_attempt_reinit returns False before interval elapses."""
        # Arrange
        manager = FallbackManager(reinit_interval_seconds=300)
        manager.mark_ml_unavailable("Test failure")
        manager.record_reinit_attempt()
        
        # Act & Assert
        assert manager.should_attempt_reinit() is False, "Should not attempt reinit before interval"
    
    def test_should_attempt_reinit_true_after_interval(self):
        """Test should_attempt_reinit returns True after interval elapses."""
        # Arrange
        manager = FallbackManager(reinit_interval_seconds=1)  # 1 second for testing
        manager.mark_ml_unavailable("Test failure")
        manager.record_reinit_attempt()
        
        # Wait for interval to elapse
        time.sleep(1.1)
        
        # Act & Assert
        assert manager.should_attempt_reinit() is True, "Should attempt reinit after interval"
    
    def test_record_reinit_attempt(self):
        """Test record_reinit_attempt updates last attempt time."""
        # Arrange
        manager = FallbackManager()
        manager.mark_ml_unavailable("Test failure")
        
        # Act
        before = datetime.now()
        manager.record_reinit_attempt()
        after = datetime.now()
        
        # Assert
        assert manager.last_reinit_attempt is not None, "Should record reinit attempt time"
        assert before <= manager.last_reinit_attempt <= after, "Should record current time"
    
    def test_get_availability_metrics_when_available(self):
        """Test get_availability_metrics returns correct data when ML is available."""
        # Arrange
        manager = FallbackManager()
        
        # Act
        metrics = manager.get_availability_metrics()
        
        # Assert
        assert metrics["ml_available"] is True
        assert metrics["fallback_mode"] is False
        assert metrics["last_failure_time"] is None
        assert metrics["last_reinit_attempt"] is None
        assert "downtime_seconds" not in metrics
    
    def test_get_availability_metrics_when_in_fallback(self):
        """Test get_availability_metrics returns correct data in fallback mode."""
        # Arrange
        manager = FallbackManager()
        manager.mark_ml_unavailable("Test failure")
        
        # Act
        metrics = manager.get_availability_metrics()
        
        # Assert
        assert metrics["ml_available"] is False
        assert metrics["fallback_mode"] is True
        assert metrics["last_failure_time"] is not None
        assert "downtime_seconds" in metrics
        assert metrics["downtime_seconds"] >= 0
    
    def test_get_availability_metrics_includes_reinit_attempt(self):
        """Test get_availability_metrics includes reinit attempt time."""
        # Arrange
        manager = FallbackManager()
        manager.mark_ml_unavailable("Test failure")
        manager.record_reinit_attempt()
        
        # Act
        metrics = manager.get_availability_metrics()
        
        # Assert
        assert metrics["last_reinit_attempt"] is not None
    
    def test_thread_safety_mark_unavailable(self):
        """Test that mark_ml_unavailable is thread-safe."""
        # Arrange
        manager = FallbackManager()
        
        # Act - simulate concurrent calls
        import threading
        threads = []
        for i in range(10):
            t = threading.Thread(target=manager.mark_ml_unavailable, args=(f"Failure {i}",))
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        # Assert - should be in fallback mode without errors
        assert manager.is_in_fallback_mode() is True
    
    def test_thread_safety_mark_available(self):
        """Test that mark_ml_available is thread-safe."""
        # Arrange
        manager = FallbackManager()
        manager.mark_ml_unavailable("Test failure")
        
        # Act - simulate concurrent calls
        import threading
        threads = []
        for i in range(10):
            t = threading.Thread(target=manager.mark_ml_available)
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        # Assert - should not be in fallback mode without errors
        assert manager.is_in_fallback_mode() is False
    
    def test_reinit_interval_five_minutes_default(self):
        """Test that default reinitialization interval is 5 minutes (300 seconds)."""
        # Arrange & Act
        manager = FallbackManager()
        
        # Assert
        assert manager.reinit_interval == timedelta(seconds=300), "Default should be 5 minutes"
    
    def test_multiple_failures_update_failure_time(self):
        """Test that multiple failures update the failure time."""
        # Arrange
        manager = FallbackManager()
        
        # Act
        manager.mark_ml_unavailable("First failure")
        first_failure_time = manager.last_failure_time
        
        time.sleep(0.1)
        
        manager.mark_ml_available()
        manager.mark_ml_unavailable("Second failure")
        second_failure_time = manager.last_failure_time
        
        # Assert
        assert second_failure_time > first_failure_time, "Should update failure time on new failure"
    
    def test_downtime_calculation(self):
        """Test that downtime is calculated correctly."""
        # Arrange
        manager = FallbackManager()
        manager.mark_ml_unavailable("Test failure")
        
        # Wait a bit
        time.sleep(0.5)
        
        # Act
        metrics = manager.get_availability_metrics()
        
        # Assert
        assert metrics["downtime_seconds"] >= 0.5, "Downtime should be at least 0.5 seconds"
        assert metrics["downtime_seconds"] < 1.0, "Downtime should be less than 1 second"
