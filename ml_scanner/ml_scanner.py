"""
ML Scanner for ML-Enhanced Secret Scanner.

This module provides the MLScanner class that integrates the Inference Engine
with the existing scanner architecture, providing a high-level interface for
ML-based secret detection.
"""

import time
from pathlib import Path
from typing import List, Optional, Dict, Any

from ml_scanner.inference_engine import InferenceEngine
from ml_scanner.code_tokenizer import CodeTokenizer
from ml_scanner.models import Finding, Detection
from ml_scanner.exceptions import (
    InferenceTimeoutError,
    InvalidInputError,
    ModelLoadError,
)
from ml_scanner.logger import get_logger

logger = get_logger(__name__)


class MLScanner:
    """
    Main interface for ML-based secret detection.
    
    The MLScanner provides:
    - File scanning with timeout handling (5 seconds default)
    - Direct text scanning
    - Result formatting (Detection → Finding conversion)
    - Availability checking
    - Detection category classification
    
    Requirements: 3.1, 3.3, 3.4, 3.5, 3.6
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        inference_engine: InferenceEngine,
    ):
        """
        Initialize ML scanner with configuration.
        
        Args:
            config: Configuration dictionary containing:
                - timeout: Default timeout in seconds (default 5)
                - confidence_threshold: Minimum confidence for detections (default 0.5)
                - enabled_categories: List of enabled detection categories
            inference_engine: InferenceEngine instance for running predictions
        """
        self.config = config
        self.inference_engine = inference_engine
        
        # Extract configuration
        self.default_timeout = config.get("timeout", 5)
        self.confidence_threshold = config.get("confidence_threshold", 0.5)
        self.enabled_categories = config.get("enabled_categories", [
            "API_KEY", "PASSWORD", "PII", "TOKEN", "CERTIFICATE", "OTHER"
        ])
        
        # Initialize tokenizer
        self.tokenizer = CodeTokenizer()
        
        # Track availability
        self._available = True
        self._last_error: Optional[str] = None
        
        logger.info(
            f"MLScanner initialized with timeout={self.default_timeout}s, "
            f"confidence_threshold={self.confidence_threshold}"
        )
    
    def scan_file(
        self,
        file_path: str,
        timeout: Optional[int] = None,
    ) -> List[Finding]:
        """
        Scan a single file for secrets using ML.
        
        This method reads the file, runs ML inference, and converts
        detections to Finding objects with proper formatting.
        
        Args:
            file_path: Path to file to scan
            timeout: Maximum time in seconds (default from config, typically 5)
            
        Returns:
            List of Finding objects with ML detections
            
        Raises:
            FileNotFoundError: If file does not exist
            InferenceTimeoutError: If inference exceeds timeout
        """
        if timeout is None:
            timeout = self.default_timeout
        
        start_time = time.time()
        
        try:
            # Read file content
            path = Path(file_path)
            if not path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            
            # Read file with encoding handling
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    content = f.read()
            except UnicodeDecodeError:
                # Try with latin-1 encoding as fallback
                logger.warning(f"UTF-8 decode failed for {file_path}, trying latin-1")
                with open(path, 'r', encoding='latin-1') as f:
                    content = f.read()
            
            # Check if timeout already exceeded
            elapsed = time.time() - start_time
            if elapsed >= timeout:
                raise InferenceTimeoutError(
                    f"File reading exceeded timeout ({timeout}s)",
                    context={"file_path": file_path, "timeout": timeout}
                )
            
            # Scan the content
            findings = self.scan_text(content, file_path)
            
            # Check total time
            total_time = time.time() - start_time
            if total_time > timeout:
                logger.warning(
                    f"Scan of {file_path} took {total_time:.2f}s, "
                    f"exceeding timeout of {timeout}s"
                )
            
            logger.info(
                f"Scanned {file_path} in {total_time:.2f}s, "
                f"found {len(findings)} detections"
            )
            
            return findings
            
        except InferenceTimeoutError:
            logger.warning(f"Timeout scanning {file_path}")
            self._available = True  # Timeout doesn't mean unavailable
            raise
        except Exception as e:
            logger.error(f"Error scanning {file_path}: {str(e)}", exc_info=True)
            self._last_error = str(e)
            self._available = False
            raise
    
    def scan_text(
        self,
        text: str,
        file_path: str = "",
    ) -> List[Finding]:
        """
        Scan text content for secrets using ML.
        
        This method runs ML inference on the provided text and converts
        detections to Finding objects.
        
        Args:
            text: Text content to scan
            file_path: Optional file path for context (used in Finding objects)
            
        Returns:
            List of Finding objects with ML detections
        """
        if not text:
            logger.debug("Empty text provided, returning no findings")
            return []
        
        try:
            # Run inference
            detections = self.inference_engine.predict(text)
            
            # Convert detections to findings
            findings = self._convert_detections_to_findings(
                detections,
                text,
                file_path
            )
            
            # Filter by enabled categories
            findings = [
                f for f in findings
                if f.secret_type in self.enabled_categories
            ]
            
            # Filter by confidence threshold
            findings = [
                f for f in findings
                if f.confidence_score and f.confidence_score >= self.confidence_threshold
            ]
            
            logger.debug(
                f"Scanned text ({len(text)} chars), "
                f"found {len(findings)} detections"
            )
            
            # Mark as available since inference succeeded
            self._available = True
            
            return findings
            
        except InferenceTimeoutError:
            logger.warning("Inference timeout during text scan")
            self._available = True  # Timeout doesn't mean unavailable
            raise
        except ModelLoadError as e:
            logger.error(f"Model load error: {str(e)}")
            self._last_error = str(e)
            self._available = False
            raise
        except Exception as e:
            logger.error(f"Error during text scan: {str(e)}", exc_info=True)
            self._last_error = str(e)
            self._available = False
            raise
    
    def _convert_detections_to_findings(
        self,
        detections: List[Detection],
        text: str,
        file_path: str,
    ) -> List[Finding]:
        """
        Convert ML detections to Finding objects.
        
        This method transforms the raw Detection objects from the inference
        engine into the unified Finding format used throughout the system.
        
        Args:
            detections: List of Detection objects from inference
            text: Original text that was scanned
            file_path: File path for the findings
            
        Returns:
            List of Finding objects
        """
        findings = []
        
        for detection in detections:
            # Calculate line number from start position
            line_number = text[:detection.start_pos].count('\n') + 1
            
            # Extract the matched text
            matched_text = detection.text[detection.start_pos:detection.end_pos]
            
            # Determine confidence level
            confidence = "HIGH" if detection.confidence_score >= 0.85 else "LOW"
            
            # Calculate entropy (simple approximation)
            entropy = self._calculate_entropy(matched_text)
            
            # Check if detection is in a comment
            is_comment = self._is_in_comment(text, detection.start_pos)
            
            # Create Finding object
            finding = Finding(
                file_path=file_path,
                line_number=line_number,
                matched_text=matched_text,
                secret_type=detection.category,
                confidence=confidence,
                entropy=entropy,
                is_comment=is_comment,
                source="ML_ONLY",
                confidence_score=detection.confidence_score,
                explanation=None,  # Will be added by Explainer if needed
            )
            
            findings.append(finding)
        
        return findings
    
    def _calculate_entropy(self, text: str) -> float:
        """
        Calculate Shannon entropy of text.
        
        This is a simple approximation used for consistency with
        the existing regex scanner.
        
        Args:
            text: Text to calculate entropy for
            
        Returns:
            Entropy value (0.0 to ~8.0 for typical text)
        """
        if not text:
            return 0.0
        
        # Count character frequencies
        char_counts: Dict[str, int] = {}
        for char in text:
            char_counts[char] = char_counts.get(char, 0) + 1
        
        # Calculate entropy
        import math
        entropy = 0.0
        text_len = len(text)
        
        for count in char_counts.values():
            probability = count / text_len
            entropy -= probability * math.log2(probability)
        
        return entropy
    
    def _is_in_comment(self, text: str, position: int) -> bool:
        """
        Check if a position in text is within a comment.
        
        This is a simple heuristic that checks for common comment patterns.
        
        Args:
            text: Full text content
            position: Position to check
            
        Returns:
            True if position appears to be in a comment
        """
        # Get the line containing the position
        line_start = text.rfind('\n', 0, position) + 1
        line_end = text.find('\n', position)
        if line_end == -1:
            line_end = len(text)
        
        line = text[line_start:line_end]
        
        # Check for common comment patterns
        stripped = line.strip()
        if stripped.startswith('//') or stripped.startswith('#'):
            return True
        if stripped.startswith('/*') or stripped.startswith('*'):
            return True
        
        return False
    
    def is_available(self) -> bool:
        """
        Check if ML scanner is operational.
        
        Returns:
            True if ML scanner is available and functional, False otherwise
        """
        return self._available
    
    def get_last_error(self) -> Optional[str]:
        """
        Get the last error message if scanner is unavailable.
        
        Returns:
            Last error message, or None if no error
        """
        return self._last_error
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get scanner statistics.
        
        Returns:
            Dictionary containing:
            - available: Whether scanner is available
            - last_error: Last error message (if any)
            - inference_metrics: Metrics from inference engine
        """
        return {
            "available": self._available,
            "last_error": self._last_error,
            "inference_metrics": self.inference_engine.get_metrics(),
        }
