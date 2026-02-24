"""
Hybrid Detector for combining regex and ML-based secret detection.

This module implements the core hybrid detection system that orchestrates
both regex-based and ML-based scanners, merges their results, and handles
fallback scenarios when ML components are unavailable.
"""

import time
import threading
from typing import List, Optional, Dict, Set, Tuple
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

from .models import Finding, ScanResult
from .logger import get_logger
from .exceptions import ErrorCode, MLScannerError

logger = get_logger(__name__)


class ResultMerger:
    """
    Merges and deduplicates findings from regex and ML scanners.
    
    Implements the merging logic specified in Requirements 4.3-4.8:
    - Deduplicates findings based on file, line, and text
    - Selects higher confidence scores when both scanners detect the same secret
    - Attributes source as HIGH_CONFIDENCE, ML_ONLY, or REGEX_ONLY
    - Applies ML-only confidence threshold of 0.85
    """
    
    def __init__(self, ml_only_threshold: float = 0.85):
        """
        Initialize ResultMerger.
        
        Args:
            ml_only_threshold: Minimum confidence score for ML-only detections (default 0.85)
        """
        self.ml_only_threshold = ml_only_threshold
        logger.debug(f"ResultMerger initialized with ML-only threshold: {ml_only_threshold}")
    
    def merge(self, regex_findings: List[Finding], ml_findings: List[Finding]) -> List[Finding]:
        """
        Merge and deduplicate findings from both scanners.
        
        Logic:
        - If same secret detected by both: Use higher confidence, mark as HIGH_CONFIDENCE
        - If detected by ML only: Mark as ML_ONLY, require confidence >= 0.85
        - If detected by regex only: Mark as REGEX_ONLY
        
        Args:
            regex_findings: Findings from regex scanner
            ml_findings: Findings from ML scanner
            
        Returns:
            Merged and deduplicated list of findings
            
        Validates: Requirements 4.3, 4.4, 4.5, 4.6, 4.7, 4.8
        """
        logger.debug(f"Merging {len(regex_findings)} regex findings with {len(ml_findings)} ML findings")
        
        merged = []
        
        # Create lookup maps for efficient matching
        # Key: (file_path, line_number, matched_text)
        regex_map: Dict[Tuple[str, int, str], Finding] = {
            self._create_key(f): f for f in regex_findings
        }
        
        ml_map: Dict[Tuple[str, int, str], Finding] = {
            self._create_key(f): f for f in ml_findings
        }
        
        # Find overlapping and unique detections
        regex_keys = set(regex_map.keys())
        ml_keys = set(ml_map.keys())
        
        # Both scanners detected (HIGH_CONFIDENCE)
        overlapping_keys = regex_keys & ml_keys
        logger.debug(f"Found {len(overlapping_keys)} overlapping detections")
        
        for key in overlapping_keys:
            regex_finding = regex_map[key]
            ml_finding = ml_map[key]
            
            # Use the finding with higher confidence score
            # ML findings have confidence_score, regex findings don't
            if ml_finding.confidence_score and ml_finding.confidence_score > 0.9:
                # Use ML finding as base but mark as HIGH_CONFIDENCE
                finding = ml_finding
            else:
                # Use regex finding as base
                finding = regex_finding
                # Preserve ML confidence score and explanation
                finding.confidence_score = ml_finding.confidence_score
                finding.explanation = ml_finding.explanation
            
            # Mark as detected by both scanners
            finding.source = "HIGH_CONFIDENCE"
            merged.append(finding)
        
        # ML only (require high confidence threshold)
        ml_only_keys = ml_keys - regex_keys
        logger.debug(f"Found {len(ml_only_keys)} ML-only detections")
        
        for key in ml_only_keys:
            ml_finding = ml_map[key]
            
            # Apply confidence threshold for ML-only detections
            if ml_finding.confidence_score and ml_finding.confidence_score >= self.ml_only_threshold:
                ml_finding.source = "ML_ONLY"
                merged.append(ml_finding)
            else:
                logger.debug(
                    f"Filtered ML-only detection with confidence {ml_finding.confidence_score:.2f} "
                    f"(threshold: {self.ml_only_threshold})"
                )
        
        # Regex only
        regex_only_keys = regex_keys - ml_keys
        logger.debug(f"Found {len(regex_only_keys)} regex-only detections")
        
        for key in regex_only_keys:
            regex_finding = regex_map[key]
            regex_finding.source = "REGEX_ONLY"
            merged.append(regex_finding)
        
        logger.info(
            f"Merge complete: {len(merged)} total findings "
            f"({len(overlapping_keys)} HIGH_CONFIDENCE, "
            f"{len([f for f in merged if f.source == 'ML_ONLY'])} ML_ONLY, "
            f"{len(regex_only_keys)} REGEX_ONLY)"
        )
        
        return merged
    
    def deduplicate(self, findings: List[Finding]) -> List[Finding]:
        """
        Remove duplicate findings based on file, line, and text.
        
        Args:
            findings: List of findings to deduplicate
            
        Returns:
            Deduplicated list of findings
            
        Validates: Requirements 4.3
        """
        seen: Set[Tuple[str, int, str]] = set()
        deduplicated = []
        
        for finding in findings:
            key = self._create_key(finding)
            if key not in seen:
                seen.add(key)
                deduplicated.append(finding)
        
        if len(findings) != len(deduplicated):
            logger.debug(f"Removed {len(findings) - len(deduplicated)} duplicate findings")
        
        return deduplicated
    
    @staticmethod
    def _create_key(finding: Finding) -> Tuple[str, int, str]:
        """
        Create a unique key for a finding based on file, line, and text.
        
        Args:
            finding: Finding to create key for
            
        Returns:
            Tuple of (file_path, line_number, matched_text)
        """
        return (finding.file_path, finding.line_number, finding.matched_text)


class FallbackManager:
    """
    Manages ML Scanner availability and fallback mode.
    
    Tracks ML Scanner state and implements reinitialization attempts
    as specified in Requirements 10.1-10.4.
    """
    
    def __init__(self, reinit_interval_seconds: int = 300):
        """
        Initialize FallbackManager.
        
        Args:
            reinit_interval_seconds: Time between reinitialization attempts (default 300 = 5 minutes)
        """
        self.reinit_interval = timedelta(seconds=reinit_interval_seconds)
        self.ml_available = True
        self.fallback_mode = False
        self.last_failure_time: Optional[datetime] = None
        self.last_reinit_attempt: Optional[datetime] = None
        self._lock = threading.Lock()
        
        logger.debug(f"FallbackManager initialized with reinit interval: {reinit_interval_seconds}s")
    
    def mark_ml_unavailable(self, reason: str) -> None:
        """
        Mark ML Scanner as unavailable and enter fallback mode.
        
        Args:
            reason: Reason for ML Scanner failure
            
        Validates: Requirements 10.1, 10.2
        """
        with self._lock:
            if not self.fallback_mode:
                self.fallback_mode = True
                self.ml_available = False
                self.last_failure_time = datetime.now()
                logger.warning(
                    f"Entering fallback mode: ML Scanner unavailable. Reason: {reason}",
                    context={"error_code": ErrorCode.MODEL_LOAD_FAILED.value}
                )
    
    def mark_ml_available(self) -> None:
        """
        Mark ML Scanner as available and exit fallback mode.
        
        Validates: Requirements 10.1, 10.2
        """
        with self._lock:
            if self.fallback_mode:
                self.fallback_mode = False
                self.ml_available = True
                self.last_failure_time = None
                logger.info("Exiting fallback mode: ML Scanner reinitialized successfully")
    
    def is_in_fallback_mode(self) -> bool:
        """
        Check if system is in fallback mode.
        
        Returns:
            True if in fallback mode, False otherwise
            
        Validates: Requirements 10.2
        """
        with self._lock:
            return self.fallback_mode
    
    def should_attempt_reinit(self) -> bool:
        """
        Check if reinitialization should be attempted.
        
        Reinitialization is attempted every 5 minutes while in fallback mode.
        
        Returns:
            True if reinitialization should be attempted, False otherwise
            
        Validates: Requirements 10.3, 10.4
        """
        with self._lock:
            if not self.fallback_mode:
                return False
            
            now = datetime.now()
            
            # First attempt after failure
            if self.last_reinit_attempt is None:
                return True
            
            # Check if enough time has passed since last attempt
            time_since_last_attempt = now - self.last_reinit_attempt
            return time_since_last_attempt >= self.reinit_interval
    
    def record_reinit_attempt(self) -> None:
        """
        Record that a reinitialization attempt was made.
        
        Validates: Requirements 10.4
        """
        with self._lock:
            self.last_reinit_attempt = datetime.now()
            logger.debug("Recorded ML Scanner reinitialization attempt")
    
    def get_availability_metrics(self) -> Dict[str, any]:
        """
        Get ML Scanner availability metrics.
        
        Returns:
            Dictionary with availability metrics
            
        Validates: Requirements 10.6
        """
        with self._lock:
            metrics = {
                "ml_available": self.ml_available,
                "fallback_mode": self.fallback_mode,
                "last_failure_time": self.last_failure_time.isoformat() if self.last_failure_time else None,
                "last_reinit_attempt": self.last_reinit_attempt.isoformat() if self.last_reinit_attempt else None,
            }
            
            if self.last_failure_time:
                downtime = datetime.now() - self.last_failure_time
                metrics["downtime_seconds"] = downtime.total_seconds()
            
            return metrics


class HybridDetector:
    """
    Orchestrates both regex and ML scanners for hybrid detection.
    
    Implements the hybrid detection system specified in Requirements 4.1-4.3
    and 10.1-10.8, combining regex-based and ML-based detection with
    resilience and fallback capabilities.
    """
    
    def __init__(
        self,
        regex_scanner,
        ml_scanner,
        config: Optional[Dict[str, any]] = None
    ):
        """
        Initialize HybridDetector.
        
        Args:
            regex_scanner: Regex-based scanner instance (FileScanner)
            ml_scanner: ML-based scanner instance (MLScanner)
            config: Configuration dictionary
        """
        self.regex_scanner = regex_scanner
        self.ml_scanner = ml_scanner
        self.config = config or {}
        
        # Initialize components
        ml_only_threshold = self.config.get("ml_only_threshold", 0.85)
        self.result_merger = ResultMerger(ml_only_threshold=ml_only_threshold)
        
        reinit_interval = self.config.get("ml_reinit_interval_seconds", 300)
        self.fallback_manager = FallbackManager(reinit_interval_seconds=reinit_interval)
        
        # Timeout settings
        self.inference_timeout = self.config.get("inference_timeout_seconds", 10)
        
        logger.info(
            f"HybridDetector initialized with ML-only threshold: {ml_only_threshold}, "
            f"inference timeout: {self.inference_timeout}s"
        )
    
    def scan_file(self, file_path: str) -> List[Finding]:
        """
        Scan a single file using both regex and ML scanners.
        
        Invokes both scanners in parallel and merges results. If ML Scanner
        is unavailable or times out, falls back to regex-only results.
        
        Args:
            file_path: Path to file to scan
            
        Returns:
            Merged and deduplicated findings
            
        Validates: Requirements 4.1, 4.2, 4.3, 10.1, 10.2, 10.5, 10.8
        """
        logger.debug(f"Scanning file: {file_path}")
        
        # Attempt ML Scanner reinitialization if needed
        if self.fallback_manager.should_attempt_reinit():
            self._attempt_ml_reinit()
        
        regex_findings = []
        ml_findings = []
        
        # Execute both scanners in parallel
        with ThreadPoolExecutor(max_workers=2) as executor:
            # Submit regex scan
            regex_future = executor.submit(self._scan_with_regex, file_path)
            
            # Submit ML scan only if not in fallback mode
            ml_future = None
            if not self.fallback_manager.is_in_fallback_mode():
                ml_future = executor.submit(self._scan_with_ml, file_path)
            
            # Collect regex results
            try:
                regex_findings = regex_future.result()
            except Exception as e:
                logger.error(f"Regex scan failed for {file_path}: {e}")
                # Continue with empty regex findings
            
            # Collect ML results with timeout
            if ml_future:
                try:
                    ml_findings = ml_future.result(timeout=self.inference_timeout)
                except TimeoutError:
                    logger.warning(
                        f"ML inference timeout ({self.inference_timeout}s) for {file_path}, "
                        "using regex-only results",
                        context={"error_code": ErrorCode.INFERENCE_TIMEOUT.value}
                    )
                    # Use only regex results for this file
                except Exception as e:
                    logger.error(f"ML scan failed for {file_path}: {e}")
                    # Continue with empty ML findings
        
        # Merge results
        merged_findings = self.result_merger.merge(regex_findings, ml_findings)
        
        logger.debug(f"Scan complete for {file_path}: {len(merged_findings)} findings")
        return merged_findings
    
    def scan_files(self, file_paths: List[str]) -> ScanResult:
        """
        Scan multiple files using hybrid detection.
        
        Args:
            file_paths: List of file paths to scan
            
        Returns:
            ScanResult with aggregated findings and metrics
            
        Validates: Requirements 4.1, 4.2, 4.3, 10.8
        """
        logger.info(f"Starting batch scan of {len(file_paths)} files")
        start_time = time.time()
        
        all_findings = []
        
        # Scan each file
        for file_path in file_paths:
            try:
                findings = self.scan_file(file_path)
                all_findings.extend(findings)
            except Exception as e:
                logger.error(f"Failed to scan {file_path}: {e}")
                # Continue with other files (resilient scanning)
        
        # Calculate metrics
        execution_time = time.time() - start_time
        high_confidence_count = len([f for f in all_findings if f.source == "HIGH_CONFIDENCE"])
        low_confidence_count = len(all_findings) - high_confidence_count
        
        result = ScanResult(
            findings=all_findings,
            files_scanned=len(file_paths),
            high_confidence_count=high_confidence_count,
            low_confidence_count=low_confidence_count,
            execution_time_seconds=execution_time,
            ml_available=not self.fallback_manager.is_in_fallback_mode(),
            fallback_mode=self.fallback_manager.is_in_fallback_mode()
        )
        
        logger.info(
            f"Batch scan complete: {len(all_findings)} findings in {execution_time:.2f}s "
            f"({high_confidence_count} HIGH_CONFIDENCE, {low_confidence_count} other)"
        )
        
        return result
    
    def is_in_fallback_mode(self) -> bool:
        """
        Check if hybrid detector is operating in fallback mode.
        
        Returns:
            True if in fallback mode (regex-only), False otherwise
            
        Validates: Requirements 10.2
        """
        return self.fallback_manager.is_in_fallback_mode()
    
    def get_availability_metrics(self) -> Dict[str, any]:
        """
        Get ML Scanner availability metrics.
        
        Returns:
            Dictionary with availability metrics
            
        Validates: Requirements 10.6
        """
        return self.fallback_manager.get_availability_metrics()
    
    def _scan_with_regex(self, file_path: str) -> List[Finding]:
        """
        Scan file with regex scanner.
        
        Args:
            file_path: Path to file to scan
            
        Returns:
            List of findings from regex scanner
        """
        try:
            # Assuming regex_scanner has a scan_file method
            # The actual interface may vary based on existing implementation
            findings = self.regex_scanner.scan_file(file_path)
            logger.debug(f"Regex scan found {len(findings)} findings in {file_path}")
            return findings
        except Exception as e:
            logger.error(f"Regex scan error for {file_path}: {e}")
            return []
    
    def _scan_with_ml(self, file_path: str) -> List[Finding]:
        """
        Scan file with ML scanner.
        
        Args:
            file_path: Path to file to scan
            
        Returns:
            List of findings from ML scanner
        """
        try:
            if not self.ml_scanner.is_available():
                self.fallback_manager.mark_ml_unavailable("ML Scanner not available")
                return []
            
            findings = self.ml_scanner.scan_file(file_path)
            logger.debug(f"ML scan found {len(findings)} findings in {file_path}")
            return findings
        except Exception as e:
            logger.error(f"ML scan error for {file_path}: {e}")
            self.fallback_manager.mark_ml_unavailable(str(e))
            return []
    
    def _attempt_ml_reinit(self) -> None:
        """
        Attempt to reinitialize ML Scanner.
        
        Validates: Requirements 10.3, 10.4
        """
        logger.info("Attempting ML Scanner reinitialization")
        self.fallback_manager.record_reinit_attempt()
        
        try:
            # Check if ML Scanner is now available
            if self.ml_scanner.is_available():
                self.fallback_manager.mark_ml_available()
                logger.info("ML Scanner reinitialization successful")
            else:
                logger.warning("ML Scanner still unavailable after reinitialization attempt")
        except Exception as e:
            logger.error(f"ML Scanner reinitialization failed: {e}")
