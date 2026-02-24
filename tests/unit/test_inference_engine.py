"""
Unit tests for InferenceEngine class.

Tests model loading, prediction, batch inference, device management,
quantization, timeout handling, and metrics tracking.
"""

import pytest
import sys
from unittest.mock import Mock, patch, MagicMock

# Mock torch before importing InferenceEngine
torch_mock = MagicMock()
torch_mock.cuda.is_available.return_value = False
torch_mock.cuda.get_device_name.return_value = "Mock GPU"
torch_mock.tensor = MagicMock
torch_mock.no_grad = MagicMock
torch_mock.softmax = MagicMock
torch_mock.max = MagicMock
torch_mock.nn.Linear = MagicMock
torch_mock.qint8 = MagicMock
torch_mock.quantization.quantize_dynamic = MagicMock
sys.modules['torch'] = torch_mock

from ml_scanner.inference_engine import InferenceEngine
from ml_scanner.model_registry import ModelRegistry
from ml_scanner.models import Detection
from ml_scanner.exceptions import (
    ModelLoadError,
    InferenceTimeoutError,
    InvalidInputError,
    GPUOutOfMemoryError,
)


@pytest.fixture
def mock_registry():
    """Create a mock model registry."""
    registry = Mock(spec=ModelRegistry)
    return registry


@pytest.fixture
def mock_model():
    """Create a mock model."""
    model = MagicMock()
    model.eval = Mock(return_value=model)
    model.to = Mock(return_value=model)
    
    # Mock model output
    mock_logits = MagicMock()
    mock_output = MagicMock()
    mock_output.logits = mock_logits
    model.return_value = mock_output
    
    return model


@pytest.fixture
def mock_tokenizer():
    """Create a mock tokenizer."""
    tokenizer = MagicMock()
    mock_tensor = MagicMock()
    mock_tensor.to = Mock(return_value=mock_tensor)
    tokenizer.return_value = {
        "input_ids": mock_tensor,
        "attention_mask": mock_tensor,
    }
    return tokenizer


class TestInferenceEngineInit:
    """Test InferenceEngine initialization."""
    
    def test_init_with_auto_device_cpu(self, mock_registry):
        """Test initialization with auto device detection (CPU)."""
        with patch("torch.cuda.is_available", return_value=False):
            engine = InferenceEngine(mock_registry, device="auto")
            assert engine.device == "cpu"
    
    def test_init_with_auto_device_cuda(self, mock_registry):
        """Test initialization with auto device detection (CUDA)."""
        with patch("torch.cuda.is_available", return_value=True):
            with patch("torch.cuda.get_device_name", return_value="Tesla V100"):
                engine = InferenceEngine(mock_registry, device="auto")
                assert engine.device == "cuda"
    
    def test_init_with_explicit_cpu(self, mock_registry):
        """Test initialization with explicit CPU device."""
        engine = InferenceEngine(mock_registry, device="cpu")
        assert engine.device == "cpu"
    
    def test_init_with_cuda_unavailable(self, mock_registry):
        """Test initialization falls back to CPU when CUDA unavailable."""
        with patch("torch.cuda.is_available", return_value=False):
            engine = InferenceEngine(mock_registry, device="cuda")
            assert engine.device == "cpu"
    
    def test_init_with_custom_timeout(self, mock_registry):
        """Test initialization with custom timeout."""
        engine = InferenceEngine(mock_registry, timeout=5)
        assert engine.timeout == 5
    
    def test_init_with_quantization_disabled(self, mock_registry):
        """Test initialization with quantization disabled."""
        engine = InferenceEngine(mock_registry, enable_quantization=False)
        assert engine.enable_quantization is False


class TestModelLoading:
    """Test model loading functionality."""
    
    def test_load_model_success(self, mock_registry, mock_model):
        """Test successful model loading."""
        mock_registry.get_model.return_value = mock_model
        
        with patch("ml_scanner.inference_engine.AutoTokenizer") as mock_tokenizer_class:
            mock_tokenizer_class.from_pretrained.return_value = MagicMock()
            
            engine = InferenceEngine(mock_registry, device="cpu")
            model = engine.load_model(version="1.0.0")
            
            assert model is not None
            mock_registry.get_model.assert_called_once_with(version="1.0.0")
            mock_model.to.assert_called_once_with("cpu")
            mock_model.eval.assert_called_once()
    
    def test_load_model_caching(self, mock_registry, mock_model):
        """Test that model is cached and reused."""
        mock_registry.get_model.return_value = mock_model
        
        with patch("ml_scanner.inference_engine.AutoTokenizer") as mock_tokenizer_class:
            mock_tokenizer_class.from_pretrained.return_value = MagicMock()
            
            engine = InferenceEngine(mock_registry, device="cpu")
            
            # Load model twice
            model1 = engine.load_model(version="1.0.0")
            model2 = engine.load_model(version="1.0.0")
            
            # Should only call registry once (cached)
            assert mock_registry.get_model.call_count == 1
            assert model1 is model2
    
    def test_load_model_different_versions(self, mock_registry, mock_model):
        """Test loading different model versions."""
        mock_registry.get_model.return_value = mock_model
        
        with patch("ml_scanner.inference_engine.AutoTokenizer") as mock_tokenizer_class:
            mock_tokenizer_class.from_pretrained.return_value = MagicMock()
            
            engine = InferenceEngine(mock_registry, device="cpu")
            
            # Load different versions
            engine.load_model(version="1.0.0")
            engine.load_model(version="2.0.0")
            
            # Should call registry twice (different versions)
            assert mock_registry.get_model.call_count == 2
    
    def test_load_model_failure(self, mock_registry):
        """Test model loading failure handling."""
        mock_registry.get_model.side_effect = Exception("Registry error")
        
        engine = InferenceEngine(mock_registry, device="cpu")
        
        with pytest.raises(ModelLoadError) as exc_info:
            engine.load_model(version="1.0.0")
        
        assert "Failed to load model" in str(exc_info.value)
    
    def test_load_model_with_quantization(self, mock_registry, mock_model):
        """Test model loading with quantization on CPU."""
        mock_registry.get_model.return_value = mock_model
        
        with patch("ml_scanner.inference_engine.AutoTokenizer") as mock_tokenizer_class:
            mock_tokenizer_class.from_pretrained.return_value = MagicMock()
            with patch("torch.quantization.quantize_dynamic") as mock_quantize:
                mock_quantize.return_value = mock_model
                
                engine = InferenceEngine(
                    mock_registry,
                    device="cpu",
                    enable_quantization=True
                )
                engine.load_model(version="1.0.0")
                
                # Quantization should be called for CPU
                mock_quantize.assert_called_once()


class TestPrediction:
    """Test prediction functionality."""
    
    def test_predict_success(self, mock_registry, mock_model, mock_tokenizer):
        """Test successful prediction."""
        mock_registry.get_model.return_value = mock_model
        
        with patch("ml_scanner.inference_engine.AutoTokenizer") as mock_tokenizer_class:
            mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
            
            engine = InferenceEngine(mock_registry, device="cpu")
            detections = engine.predict("api_key = 'AKIAIOSFODNN7EXAMPLE'")
            
            assert isinstance(detections, list)
            # Check that prediction was called
            assert engine._model_cache is not None
    
    def test_predict_invalid_input_empty(self, mock_registry):
        """Test prediction with empty input."""
        engine = InferenceEngine(mock_registry, device="cpu")
        
        with pytest.raises(InvalidInputError):
            engine.predict("")
    
    def test_predict_invalid_input_type(self, mock_registry):
        """Test prediction with invalid input type."""
        engine = InferenceEngine(mock_registry, device="cpu")
        
        with pytest.raises(InvalidInputError):
            engine.predict(None)
    
    def test_predict_updates_metrics(self, mock_registry, mock_model, mock_tokenizer):
        """Test that prediction updates metrics."""
        mock_registry.get_model.return_value = mock_model
        
        with patch("ml_scanner.inference_engine.AutoTokenizer") as mock_tokenizer_class:
            mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
            
            engine = InferenceEngine(mock_registry, device="cpu")
            engine.predict("test code")
            
            metrics = engine.get_metrics()
            assert metrics["total_inferences"] == 1
            assert metrics["avg_latency_ms"] > 0


class TestBatchPrediction:
    """Test batch prediction functionality."""
    
    def test_predict_batch_success(self, mock_registry, mock_model, mock_tokenizer):
        """Test successful batch prediction."""
        mock_registry.get_model.return_value = mock_model
        
        with patch("ml_scanner.inference_engine.AutoTokenizer") as mock_tokenizer_class:
            mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
            
            engine = InferenceEngine(mock_registry, device="cpu")
            code_samples = [
                "api_key = 'AKIA123'",
                "password = 'secret'",
                "token = 'ghp_abc123'",
            ]
            
            results = engine.predict_batch(code_samples, batch_size=2)
            
            assert isinstance(results, list)
            assert len(results) == len(code_samples)
    
    def test_predict_batch_empty_input(self, mock_registry):
        """Test batch prediction with empty input."""
        engine = InferenceEngine(mock_registry, device="cpu")
        results = engine.predict_batch([])
        assert results == []
    
    def test_predict_batch_custom_batch_size(self, mock_registry, mock_model, mock_tokenizer):
        """Test batch prediction with custom batch size."""
        mock_registry.get_model.return_value = mock_model
        
        with patch("ml_scanner.inference_engine.AutoTokenizer") as mock_tokenizer_class:
            mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
            
            engine = InferenceEngine(mock_registry, device="cpu")
            code_samples = ["code1", "code2", "code3", "code4", "code5"]
            
            results = engine.predict_batch(code_samples, batch_size=2)
            
            assert len(results) == 5


class TestMetrics:
    """Test metrics tracking functionality."""
    
    def test_get_metrics_initial(self, mock_registry):
        """Test getting metrics before any inferences."""
        engine = InferenceEngine(mock_registry, device="cpu")
        metrics = engine.get_metrics()
        
        assert metrics["total_inferences"] == 0
        assert metrics["avg_latency_ms"] == 0.0
        assert metrics["memory_warnings"] == 0
        assert metrics["timeouts"] == 0
    
    def test_get_metrics_after_inference(self, mock_registry, mock_model, mock_tokenizer):
        """Test metrics after performing inference."""
        mock_registry.get_model.return_value = mock_model
        
        with patch("ml_scanner.inference_engine.AutoTokenizer") as mock_tokenizer_class:
            mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
            
            engine = InferenceEngine(mock_registry, device="cpu")
            engine.predict("test code")
            
            metrics = engine.get_metrics()
            assert metrics["total_inferences"] == 1
            assert metrics["avg_latency_ms"] > 0


class TestCacheManagement:
    """Test cache management functionality."""
    
    def test_clear_cache(self, mock_registry, mock_model, mock_tokenizer):
        """Test clearing model cache."""
        mock_registry.get_model.return_value = mock_model
        
        with patch("ml_scanner.inference_engine.AutoTokenizer") as mock_tokenizer_class:
            mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
            
            engine = InferenceEngine(mock_registry, device="cpu")
            engine.load_model(version="1.0.0")
            
            assert engine._model_cache is not None
            
            engine.clear_cache()
            
            assert engine._model_cache is None
            assert engine._tokenizer_cache is None
            assert engine._cached_version is None


class TestMemoryManagement:
    """Test memory management functionality."""
    
    def test_memory_warning_threshold(self, mock_registry, mock_model, mock_tokenizer):
        """Test memory warning when usage exceeds 1GB."""
        mock_registry.get_model.return_value = mock_model
        
        with patch("ml_scanner.inference_engine.AutoTokenizer") as mock_tokenizer_class:
            mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
            
            with patch("psutil.Process") as mock_process:
                # Mock memory usage > 1GB
                mock_process.return_value.memory_info.return_value.rss = 2 * 1024 * 1024 * 1024
                
                engine = InferenceEngine(mock_registry, device="cpu")
                engine.predict("test code")
                
                metrics = engine.get_metrics()
                assert metrics["memory_warnings"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
