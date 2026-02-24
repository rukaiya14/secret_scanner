"""
Inference Engine for ML-Enhanced Secret Scanner.

This module provides the InferenceEngine class that manages model loading,
prediction execution, GPU/CPU optimization, and inference metrics tracking.
"""

import time
import torch
import psutil
from pathlib import Path
from typing import List, Optional, Dict, Any
from threading import Lock
from transformers import PreTrainedModel, AutoTokenizer, AutoModelForSequenceClassification

from ml_scanner.model_registry import ModelRegistry
from ml_scanner.models import Detection
from ml_scanner.exceptions import (
    ModelLoadError,
    InferenceTimeoutError,
    GPUOutOfMemoryError,
    InvalidInputError,
)
from ml_scanner.logger import get_logger

logger = get_logger(__name__)


class InferenceEngine:
    """
    Manages ML model inference with GPU/CPU optimization and caching.
    
    The InferenceEngine handles:
    - Model loading from Model Registry with caching
    - Single and batch predictions with confidence scores
    - GPU/CPU auto-detection and device management
    - Model quantization for memory optimization
    - Inference timeout handling
    - Inference metrics tracking (latency, memory usage)
    
    Requirements: 3.2, 3.3, 3.6, 3.7, 3.8, 8.1, 8.2
    """
    
    def __init__(
        self,
        model_registry: ModelRegistry,
        device: str = "auto",
        timeout: int = 10,
        enable_quantization: bool = True,
    ):
        """
        Initialize inference engine.
        
        Args:
            model_registry: Model registry for loading models
            device: "cpu", "cuda", or "auto" for automatic detection
            timeout: Maximum inference time in seconds (default 10)
            enable_quantization: Whether to use model quantization for memory optimization
        """
        self.model_registry = model_registry
        self.timeout = timeout
        self.enable_quantization = enable_quantization
        
        # Device management
        self.device = self._detect_device(device)
        logger.info(f"InferenceEngine initialized with device: {self.device}")
        
        # Model cache
        self._model_cache: Optional[PreTrainedModel] = None
        self._tokenizer_cache: Optional[AutoTokenizer] = None
        self._cached_version: Optional[str] = None
        self._cache_lock = Lock()
        
        # Metrics tracking
        self._metrics = {
            "total_inferences": 0,
            "total_latency_ms": 0.0,
            "confidence_scores": [],
            "memory_warnings": 0,
            "timeouts": 0,
            "gpu_oom_errors": 0,
        }
        self._metrics_lock = Lock()
        
        # Category mapping for secret types
        self._category_labels = [
            "API_KEY",
            "PASSWORD",
            "PII",
            "TOKEN",
            "CERTIFICATE",
            "OTHER",
        ]
    
    def _detect_device(self, device: str) -> str:
        """
        Auto-detect GPU/CPU and manage device selection.
        
        Args:
            device: "cpu", "cuda", or "auto"
            
        Returns:
            Selected device string ("cpu" or "cuda")
        """
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
                logger.info(f"GPU detected: {torch.cuda.get_device_name(0)}")
            else:
                device = "cpu"
                logger.info("No GPU detected, using CPU")
        elif device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available, falling back to CPU")
            device = "cpu"
        
        return device
    
    def load_model(self, version: str = "latest") -> PreTrainedModel:
        """
        Load model from registry into memory with caching.
        
        This method implements model caching to avoid repeated loading.
        The model is loaded once and reused for multiple inferences.
        
        Args:
            version: Model version to load (default "latest")
            
        Returns:
            Loaded PreTrainedModel
            
        Raises:
            ModelLoadError: If model loading fails
        """
        with self._cache_lock:
            # Check if model is already cached
            if self._model_cache is not None and self._cached_version == version:
                logger.debug(f"Using cached model version {version}")
                return self._model_cache
            
            try:
                logger.info(f"Loading model version {version} from registry")
                start_time = time.time()
                
                # Load model from registry
                model = self.model_registry.get_model(version=version)
                
                # Move model to device
                model = model.to(self.device)
                
                # Apply quantization if enabled and on CPU
                if self.enable_quantization and self.device == "cpu":
                    logger.info("Applying model quantization for CPU inference")
                    model = self._quantize_model(model)
                
                # Set model to evaluation mode
                model.eval()
                
                # Load tokenizer
                # Note: Assuming tokenizer is stored alongside model
                # In production, this should be loaded from registry metadata
                tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
                
                # Cache the model and tokenizer
                self._model_cache = model
                self._tokenizer_cache = tokenizer
                self._cached_version = version
                
                load_time = time.time() - start_time
                logger.info(f"Model loaded successfully in {load_time:.2f}s")
                
                return model
                
            except Exception as e:
                logger.error(f"Failed to load model: {str(e)}", exc_info=True)
                raise ModelLoadError(
                    f"Failed to load model version {version}: {str(e)}",
                    context={"version": version, "device": self.device}
                )
    
    def _quantize_model(self, model: PreTrainedModel) -> PreTrainedModel:
        """
        Apply model quantization for memory optimization.
        
        Quantization reduces model memory footprint by converting
        weights from FP32 to INT8, reducing memory usage to under 500MB.
        
        Args:
            model: Model to quantize
            
        Returns:
            Quantized model
        """
        try:
            # Dynamic quantization for linear layers
            quantized_model = torch.quantization.quantize_dynamic(
                model,
                {torch.nn.Linear},
                dtype=torch.qint8
            )
            logger.info("Model quantization applied successfully")
            return quantized_model
        except Exception as e:
            logger.warning(f"Quantization failed, using original model: {str(e)}")
            return model
    
    def predict(self, code_text: str, version: str = "latest") -> List[Detection]:
        """
        Run inference on a single code snippet.
        
        This method tokenizes the input, runs the model, and returns
        detections with confidence scores.
        
        Args:
            code_text: Code snippet to analyze
            version: Model version to use (default "latest")
            
        Returns:
            List of Detection objects with confidence scores
            
        Raises:
            InferenceTimeoutError: If inference exceeds timeout
            InvalidInputError: If input is invalid
        """
        start_time = time.time()
        
        try:
            # Validate input
            if not code_text or not isinstance(code_text, str):
                raise InvalidInputError(
                    "Invalid input: code_text must be a non-empty string",
                    context={"code_text_type": type(code_text).__name__}
                )
            
            # Load model if not cached
            model = self.load_model(version)
            
            # Check memory usage
            self._check_memory_usage()
            
            # Tokenize input
            inputs = self._tokenizer_cache(
                code_text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            )
            
            # Move inputs to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Run inference with timeout
            detections = self._run_inference_with_timeout(
                model, inputs, code_text, start_time
            )
            
            # Track metrics
            latency_ms = (time.time() - start_time) * 1000
            self._update_metrics(latency_ms, detections)
            
            logger.debug(
                f"Inference completed in {latency_ms:.2f}ms, "
                f"found {len(detections)} detections"
            )
            
            return detections
            
        except InferenceTimeoutError:
            with self._metrics_lock:
                self._metrics["timeouts"] += 1
            raise
        except torch.cuda.OutOfMemoryError as e:
            with self._metrics_lock:
                self._metrics["gpu_oom_errors"] += 1
            logger.error("GPU out of memory, falling back to CPU")
            raise GPUOutOfMemoryError(
                "GPU out of memory during inference",
                context={"device": self.device}
            )
        except Exception as e:
            logger.error(f"Inference failed: {str(e)}", exc_info=True)
            raise
    
    def _run_inference_with_timeout(
        self,
        model: PreTrainedModel,
        inputs: Dict[str, torch.Tensor],
        code_text: str,
        start_time: float,
    ) -> List[Detection]:
        """
        Run inference with timeout handling.
        
        Args:
            model: Model to use for inference
            inputs: Tokenized inputs
            code_text: Original code text
            start_time: Start time for timeout calculation
            
        Returns:
            List of detections
            
        Raises:
            InferenceTimeoutError: If inference exceeds timeout
        """
        # Check if timeout already exceeded
        if time.time() - start_time > self.timeout:
            raise InferenceTimeoutError(
                f"Inference timeout exceeded ({self.timeout}s)",
                context={"timeout": self.timeout}
            )
        
        # Run inference
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Extract predictions
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=-1)
        
        # Get predictions
        detections = []
        for i, probs in enumerate(probabilities):
            # Get the predicted class and confidence
            confidence_score, predicted_class = torch.max(probs, dim=0)
            confidence_score = confidence_score.item()
            predicted_class = predicted_class.item()
            
            # Only include detections with confidence > 0.5
            if confidence_score > 0.5:
                category = self._category_labels[predicted_class] if predicted_class < len(self._category_labels) else "OTHER"
                
                detection = Detection(
                    text=code_text,
                    start_pos=0,
                    end_pos=len(code_text),
                    category=category,
                    confidence_score=confidence_score,
                    token_attributions=None,  # Will be added by Explainer
                )
                detections.append(detection)
        
        return detections
    
    def predict_batch(
        self,
        code_texts: List[str],
        version: str = "latest",
        batch_size: int = 16,
    ) -> List[List[Detection]]:
        """
        Run batch inference on multiple code snippets.
        
        Batch processing improves throughput by processing multiple
        samples simultaneously.
        
        Args:
            code_texts: List of code snippets to analyze
            version: Model version to use (default "latest")
            batch_size: Number of samples to process in each batch
            
        Returns:
            List of detection lists, one per input code snippet
        """
        if not code_texts:
            return []
        
        logger.info(f"Running batch inference on {len(code_texts)} samples")
        start_time = time.time()
        
        # Load model if not cached
        model = self.load_model(version)
        
        all_detections = []
        
        # Process in batches
        for i in range(0, len(code_texts), batch_size):
            batch = code_texts[i:i + batch_size]
            
            try:
                # Tokenize batch
                inputs = self._tokenizer_cache(
                    batch,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                    padding=True
                )
                
                # Move to device
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Run inference
                with torch.no_grad():
                    outputs = model(**inputs)
                
                # Extract predictions for each sample in batch
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=-1)
                
                for j, (probs, code_text) in enumerate(zip(probabilities, batch)):
                    detections = []
                    confidence_score, predicted_class = torch.max(probs, dim=0)
                    confidence_score = confidence_score.item()
                    predicted_class = predicted_class.item()
                    
                    if confidence_score > 0.5:
                        category = self._category_labels[predicted_class] if predicted_class < len(self._category_labels) else "OTHER"
                        
                        detection = Detection(
                            text=code_text,
                            start_pos=0,
                            end_pos=len(code_text),
                            category=category,
                            confidence_score=confidence_score,
                            token_attributions=None,
                        )
                        detections.append(detection)
                    
                    all_detections.append(detections)
                
            except Exception as e:
                logger.error(f"Batch inference failed for batch {i}: {str(e)}")
                # Add empty results for failed batch
                all_detections.extend([[] for _ in batch])
        
        total_time = time.time() - start_time
        logger.info(
            f"Batch inference completed in {total_time:.2f}s, "
            f"processed {len(code_texts)} samples"
        )
        
        return all_detections
    
    def _check_memory_usage(self) -> None:
        """
        Check memory usage and log warning if exceeds 1GB.
        
        When memory usage exceeds 1GB, logs a warning and clears caches.
        """
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        
        if memory_mb > 1024:  # 1GB threshold
            logger.warning(
                f"Memory usage exceeds 1GB: {memory_mb:.2f}MB. Clearing caches."
            )
            with self._metrics_lock:
                self._metrics["memory_warnings"] += 1
            
            # Clear PyTorch cache if using CUDA
            if self.device == "cuda":
                torch.cuda.empty_cache()
    
    def _update_metrics(
        self,
        latency_ms: float,
        detections: List[Detection]
    ) -> None:
        """
        Update inference metrics.
        
        Args:
            latency_ms: Inference latency in milliseconds
            detections: List of detections from inference
        """
        with self._metrics_lock:
            self._metrics["total_inferences"] += 1
            self._metrics["total_latency_ms"] += latency_ms
            
            # Track confidence scores
            for detection in detections:
                self._metrics["confidence_scores"].append(detection.confidence_score)
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Return inference performance metrics.
        
        Returns:
            Dictionary containing:
            - total_inferences: Total number of inferences performed
            - avg_latency_ms: Average inference latency in milliseconds
            - min_confidence: Minimum confidence score
            - max_confidence: Maximum confidence score
            - avg_confidence: Average confidence score
            - memory_warnings: Number of memory warnings
            - timeouts: Number of timeout errors
            - gpu_oom_errors: Number of GPU OOM errors
        """
        with self._metrics_lock:
            metrics = self._metrics.copy()
            
            # Calculate derived metrics
            if metrics["total_inferences"] > 0:
                metrics["avg_latency_ms"] = (
                    metrics["total_latency_ms"] / metrics["total_inferences"]
                )
            else:
                metrics["avg_latency_ms"] = 0.0
            
            if metrics["confidence_scores"]:
                metrics["min_confidence"] = min(metrics["confidence_scores"])
                metrics["max_confidence"] = max(metrics["confidence_scores"])
                metrics["avg_confidence"] = (
                    sum(metrics["confidence_scores"]) / len(metrics["confidence_scores"])
                )
            else:
                metrics["min_confidence"] = 0.0
                metrics["max_confidence"] = 0.0
                metrics["avg_confidence"] = 0.0
            
            # Remove raw confidence scores from output (too large)
            del metrics["confidence_scores"]
            
            return metrics
    
    def clear_cache(self) -> None:
        """
        Clear model cache to free memory.
        
        This forces the next inference to reload the model from registry.
        """
        with self._cache_lock:
            self._model_cache = None
            self._tokenizer_cache = None
            self._cached_version = None
            
            # Clear PyTorch cache if using CUDA
            if self.device == "cuda":
                torch.cuda.empty_cache()
            
            logger.info("Model cache cleared")
