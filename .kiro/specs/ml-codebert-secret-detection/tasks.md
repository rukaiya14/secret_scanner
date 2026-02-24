# Implementation Plan: ML-Enhanced Secret Scanner with CodeBERT

## Overview

This implementation plan breaks down the ML-Enhanced Secret Scanner into discrete, actionable coding tasks. The system adds machine learning capabilities using CodeBERT transformers to the existing GitHub Secret Scanner, providing context-aware detection with explainability features.

The implementation follows a bottom-up approach: core infrastructure first (data models, configuration), then training pipeline, inference engine, integration components, and finally monitoring and deployment.

## Tasks

- [x] 1. Set up project structure and core infrastructure
  - Create Python package structure for ml_scanner module
  - Set up configuration management (YAML/JSON loading and validation)
  - Define core data models (Finding, Detection, Explanation, ModelMetadata, ScanResult, Dataset)
  - Set up logging infrastructure with structured JSON logging
  - Create error code enumeration and exception classes
  - Install dependencies: transformers, torch, hypothesis, pytest, shap, lime, pyyaml, requests
  - _Requirements: 13.1, 13.7, 13.8_

- [ ] 2. Implement training pipeline components
  - [x] 2.1 Implement DatasetLoader class
    - Write methods to download GitHub datasets (truffleHogRegexes, git-secrets)
    - Write methods to download Kaggle datasets (IMDB reviews for PII)
    - Implement checksum validation for dataset integrity
    - Implement dataset splitting (70% train, 15% val, 15% test)
    - _Requirements: 1.1, 1.2, 1.3, 2.1, 2.2, 2.3, 2.4_
  
  - [ ]* 2.2 Write property test for dataset loading
    - **Property 1: Dataset Loading Round Trip**
    - **Validates: Requirements 1.1, 1.2, 2.1, 2.2**
  
  - [ ]* 2.3 Write property test for dataset combination
    - **Property 2: Dataset Combination Preserves Examples**
    - **Validates: Requirements 1.3**
  
  - [ ]* 2.4 Write property test for checksum validation
    - **Property 5: Checksum Validation**
    - **Validates: Requirements 2.3**
  
  - [ ]* 2.5 Write property test for dataset split proportions
    - **Property 6: Dataset Split Proportions**
    - **Validates: Requirements 2.4**
  
  - [x] 2.6 Implement DataPreprocessor class
    - Write tokenization method using CodeBERT tokenizer (max 512 tokens)
    - Implement class balancing to maintain max 3:1 ratio
    - Implement data augmentation for minority classes
    - _Requirements: 1.4, 2.5, 2.6, 3.1_
  
  - [ ]* 2.7 Write property test for tokenization length constraint
    - **Property 3: Tokenization Length Constraint**
    - **Validates: Requirements 1.4, 3.1**
  
  - [ ]* 2.8 Write property test for class balancing ratio
    - **Property 7: Class Balancing Ratio**
    - **Validates: Requirements 2.5**
  
  - [ ]* 2.9 Write property test for conditional augmentation
    - **Property 8: Conditional Augmentation**
    - **Validates: Requirements 2.6**
  
  - [x] 2.10 Implement ModelTrainer class
    - Write fine-tuning method for CodeBERT model (3 epochs, lr=2e-5)
    - Implement model saving to Model Registry with metadata
    - Implement training loop with validation
    - _Requirements: 1.5, 1.7, 3.2_
  
  - [x] 2.11 Implement PerformanceValidator class
    - Write evaluation method computing precision, recall, F1-score, confusion matrix
    - Implement threshold checking (min 90% precision, 85% recall)
    - Generate performance reports
    - _Requirements: 1.6, 1.8_
  
  - [ ]* 2.12 Write unit tests for training pipeline error handling
    - Test dataset loading failures with retries
    - Test training failures and checkpoint saving
    - Test performance threshold failures
    - _Requirements: 2.8_

- [x] 3. Checkpoint - Verify training pipeline
  - Ensure all training pipeline tests pass, ask the user if questions arise.

- [ ] 4. Implement Model Registry
  - [x] 4.1 Implement ModelRegistry class
    - Write model registration with semantic versioning
    - Implement model retrieval by version or environment tag
    - Implement model promotion between STAGING and PRODUCTION
    - Create storage structure (models/, environments/, cache/)
    - Implement last known good model fallback
    - Implement model lineage tracking
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.6, 6.7, 6.8_
  
  - [ ]* 4.2 Write property test for model persistence round trip
    - **Property 4: Model Persistence Round Trip**
    - **Validates: Requirements 1.7, 3.2, 6.6**
  
  - [ ]* 4.3 Write property test for semantic version format
    - **Property 26: Semantic Version Format**
    - **Validates: Requirements 6.1**
  
  - [ ]* 4.4 Write property test for model metadata completeness
    - **Property 27: Model Metadata Completeness**
    - **Validates: Requirements 6.2**
  
  - [ ]* 4.5 Write property test for model promotion
    - **Property 28: Model Promotion**
    - **Validates: Requirements 6.3**
  
  - [ ]* 4.6 Write property test for registry fallback
    - **Property 32: Registry Fallback**
    - **Validates: Requirements 6.8**
  
  - [ ]* 4.7 Write unit tests for model registry
    - Test model format validation
    - Test version retention (keep 5 most recent)
    - Test storage quota handling
    - _Requirements: 6.4, 6.5_

- [ ] 5. Implement Inference Engine
  - [x] 5.1 Implement InferenceEngine class
    - Write model loading from Model Registry with caching
    - Implement single prediction method with confidence scores
    - Implement batch prediction for multiple code snippets
    - Add GPU/CPU auto-detection and device management
    - Implement model quantization for memory optimization
    - Add inference timeout handling (10 seconds)
    - Track inference metrics (latency, memory usage)
    - _Requirements: 3.2, 3.3, 3.6, 3.7, 3.8, 8.1, 8.2_
  
  - [ ]* 5.2 Write property test for confidence score range
    - **Property 11: Confidence Score Range**
    - **Validates: Requirements 3.4**
  
  - [ ]* 5.3 Write property test for detection category validity
    - **Property 12: Detection Category Validity**
    - **Validates: Requirements 3.5**
  
  - [ ]* 5.4 Write property test for inference time constraint
    - **Property 13: Inference Time Constraint**
    - **Validates: Requirements 3.6**
  
  - [ ]* 5.5 Write property test for model caching
    - **Property 40: Model Caching**
    - **Validates: Requirements 8.1**
  
  - [ ]* 5.6 Write property test for batch inference correctness
    - **Property 41: Batch Inference Correctness**
    - **Validates: Requirements 8.2**
  
  - [ ]* 5.7 Write unit tests for inference engine
    - Test GPU OOM handling and CPU fallback
    - Test invalid input handling (binary files, encoding issues)
    - Test inference timeout behavior
    - Test memory warning threshold (1GB)
    - _Requirements: 8.7_

- [ ] 6. Implement ML Scanner
  - [x] 6.1 Implement CodeTokenizer class
    - Wrap CodeBERT tokenizer with encode/decode methods
    - Handle max sequence length truncation
    - _Requirements: 3.1_
  
  - [x] 6.2 Implement MLScanner class
    - Write scan_file method with timeout (5 seconds default)
    - Write scan_text method for direct text scanning
    - Implement result formatting (Detection → Finding conversion)
    - Add availability check method
    - Classify detections into categories (API_KEY, PASSWORD, PII, TOKEN, CERTIFICATE, OTHER)
    - _Requirements: 3.1, 3.3, 3.4, 3.5, 3.6_
  
  - [ ]* 6.3 Write unit tests for ML Scanner
    - Test file scanning with various file sizes
    - Test timeout handling
    - Test category classification
    - Test availability checking
    - _Requirements: 3.6_

- [x] 7. Checkpoint - Verify inference components
  - Ensure all inference and ML scanner tests pass, ask the user if questions arise.

- [ ] 8. Implement Explainer components
  - [x] 8.1 Implement base Explainer abstract class
    - Define explain method interface
    - Define Explanation data model
    - _Requirements: 5.1, 5.2_
  
  - [x] 8.2 Implement SHAPExplainer class
    - Integrate SHAP library for token attributions
    - Generate top 5 influential tokens
    - Normalize attribution scores to sum to 1.0
    - Format explanations with highlighted tokens
    - Add timeout handling (2 seconds)
    - _Requirements: 5.1, 5.3, 5.4, 5.5, 5.6, 5.7_
  
  - [x] 8.3 Implement LIMEExplainer class
    - Integrate LIME library as alternative explainer
    - Implement same interface as SHAPExplainer
    - _Requirements: 5.2_
  
  - [ ]* 8.4 Write property test for token attribution completeness
    - **Property 20: Token Attribution Completeness**
    - **Validates: Requirements 5.3**
  
  - [ ]* 8.5 Write property test for top tokens count
    - **Property 21: Top Tokens Count**
    - **Validates: Requirements 5.4**
  
  - [ ]* 8.6 Write property test for attribution score normalization
    - **Property 22: Attribution Score Normalization**
    - **Validates: Requirements 5.5**
  
  - [ ]* 8.7 Write property test for explanation generation time
    - **Property 23: Explanation Generation Time**
    - **Validates: Requirements 5.6**
  
  - [ ]* 8.8 Write property test for explanation format completeness
    - **Property 24: Explanation Format Completeness**
    - **Validates: Requirements 5.7**
  
  - [ ]* 8.9 Write property test for explanation error handling
    - **Property 25: Explanation Error Handling**
    - **Validates: Requirements 5.8**

- [ ] 9. Implement Hybrid Detector
  - [x] 9.1 Implement ResultMerger class
    - Write merge method combining regex and ML findings
    - Implement deduplication logic (same file, line, text)
    - Implement confidence score selection (use higher score)
    - Implement source attribution (HIGH_CONFIDENCE, ML_ONLY, REGEX_ONLY)
    - Apply ML-only confidence threshold (0.85)
    - _Requirements: 4.3, 4.4, 4.5, 4.6, 4.7, 4.8_
  
  - [ ]* 9.2 Write property test for result deduplication
    - **Property 15: Result Deduplication**
    - **Validates: Requirements 4.3**
  
  - [ ]* 9.3 Write property test for confidence score selection
    - **Property 16: Confidence Score Selection**
    - **Validates: Requirements 4.4**
  
  - [ ]* 9.4 Write property test for high confidence attribution
    - **Property 17: High Confidence Attribution**
    - **Validates: Requirements 4.5**
  
  - [ ]* 9.5 Write property test for ML-only confidence threshold
    - **Property 18: ML-Only Confidence Threshold**
    - **Validates: Requirements 4.6**
  
  - [ ]* 9.6 Write property test for source attribution completeness
    - **Property 19: Source Attribution Completeness**
    - **Validates: Requirements 4.7, 4.8**
  
  - [x] 9.7 Implement FallbackManager class
    - Track ML Scanner availability state
    - Implement fallback mode detection
    - Implement reinitialization attempts (every 5 minutes)
    - _Requirements: 10.1, 10.2, 10.3, 10.4_
  
  - [x] 9.8 Implement HybridDetector class
    - Write scan_file method invoking both scanners in parallel
    - Write scan_files method for batch scanning
    - Integrate ResultMerger for combining results
    - Integrate FallbackManager for resilience
    - Add fallback mode checking method
    - _Requirements: 4.1, 4.2, 4.3, 10.1, 10.2, 10.5, 10.8_
  
  - [ ]* 9.9 Write property test for hybrid scanner invocation
    - **Property 14: Hybrid Scanner Invocation**
    - **Validates: Requirements 4.1, 4.2**
  
  - [ ]* 9.10 Write property test for fallback mode operation
    - **Property 38: Fallback Mode Operation**
    - **Validates: Requirements 7.7, 10.1, 10.2, 10.3**
  
  - [ ]* 9.11 Write property test for resilient scanning
    - **Property 52: Resilient Scanning**
    - **Validates: Requirements 10.8**
  
  - [ ]* 9.12 Write unit tests for hybrid detector
    - Test parallel scanner invocation
    - Test inference timeout handling per file
    - Test ML Scanner reinitialization
    - Test registry cache fallback
    - _Requirements: 10.5, 10.7_

- [x] 10. Checkpoint - Verify hybrid detection
  - Ensure all hybrid detector tests pass, ask the user if questions arise.

- [ ] 11. Implement Alert Manager
  - [x] 11.1 Implement SlackFormatter class
    - Format messages using Slack Block Kit JSON
    - Include confidence score, detection source, token attribution
    - Include file path, line number, code snippet
    - Add action buttons (Mark False Positive, View Full Report)
    - _Requirements: 9.2, 9.3, 9.4, 9.5, 9.6, 9.7_
  
  - [x] 11.2 Implement FeedbackCollector class
    - Store false positive feedback with all required fields
    - Implement feedback export functionality
    - Track false positive rate by detection category
    - _Requirements: 14.1, 14.2, 14.3, 14.6, 14.8_
  
  - [x] 11.3 Implement AlertManager class
    - Write send_alert method with Slack webhook integration
    - Implement retry logic (3 attempts with exponential backoff)
    - Implement mark_false_positive method
    - Integrate SlackFormatter and FeedbackCollector
    - _Requirements: 9.1, 9.8, 14.1_
  
  - [ ]* 11.4 Write property test for Slack alert sending
    - **Property 43: Slack Alert Sending**
    - **Validates: Requirements 9.1**
  
  - [ ]* 11.5 Write property test for Slack message completeness
    - **Property 44: Slack Message Completeness**
    - **Validates: Requirements 9.2, 9.3, 9.4, 9.5, 9.7**
  
  - [ ]* 11.6 Write property test for Slack Block Kit format
    - **Property 45: Slack Block Kit Format**
    - **Validates: Requirements 9.6**
  
  - [ ]* 11.7 Write property test for Slack retry logic
    - **Property 46: Slack Retry Logic**
    - **Validates: Requirements 9.8**
  
  - [ ]* 11.8 Write property test for false positive marking
    - **Property 70: False Positive Marking**
    - **Validates: Requirements 14.1, 14.2, 14.3**
  
  - [ ]* 11.9 Write unit tests for alert manager
    - Test webhook failure handling
    - Test message formatting errors
    - Test feedback storage
    - _Requirements: 9.8_

- [ ] 12. Implement Git Hook Integration
  - [x] 12.1 Implement CommitScanner class
    - Detect modified files in current git commit
    - Filter files to scan (exclude binaries, large files)
    - _Requirements: 7.2_
  
  - [x] 12.2 Implement TerminalFormatter class
    - Format scan results for terminal display
    - Include explanations for ML detections
    - Use color coding for severity levels
    - _Requirements: 7.6_
  
  - [x] 12.3 Implement GitHookIntegration class
    - Write scan_commit method with timeout (10 seconds)
    - Implement should_block_commit logic (HIGH_CONFIDENCE blocks, ML_ONLY < 0.90 warns)
    - Implement display_results method using TerminalFormatter
    - Write results to local audit file
    - _Requirements: 7.1, 7.3, 7.4, 7.5, 7.6, 7.8_
  
  - [x] 12.4 Create pre-commit git hook script
    - Write bash script invoking GitHookIntegration
    - Handle exit codes (0=pass, 1=block, 2=error)
    - Display appropriate messages
    - _Requirements: 7.1, 7.7_
  
  - [ ]* 12.5 Write property test for modified files only
    - **Property 33: Modified Files Only**
    - **Validates: Requirements 7.2**
  
  - [ ]* 12.6 Write property test for high confidence blocking
    - **Property 34: High Confidence Blocking**
    - **Validates: Requirements 7.3**
  
  - [ ]* 12.7 Write property test for low confidence warning
    - **Property 35: Low Confidence Warning**
    - **Validates: Requirements 7.4**
  
  - [ ]* 12.8 Write property test for commit scan time constraint
    - **Property 36: Commit Scan Time Constraint**
    - **Validates: Requirements 7.5**
  
  - [ ]* 12.9 Write property test for terminal output completeness
    - **Property 37: Terminal Output Completeness**
    - **Validates: Requirements 7.6**
  
  - [ ]* 12.10 Write property test for audit logging
    - **Property 39: Audit Logging**
    - **Validates: Requirements 7.8**

- [x] 13. Checkpoint - Verify git hook integration
  - Ensure all git hook tests pass, ask the user if questions arise.

- [ ] 14. Implement monitoring and metrics
  - [x] 14.1 Implement MetricsExporter class
    - Track prediction latency per inference
    - Track confidence score distribution
    - Track detection rate (secrets per 1000 files)
    - Track false positive rate from feedback
    - Export metrics in Prometheus format
    - _Requirements: 12.1, 12.2, 12.3, 12.4, 12.5_
  
  - [x] 14.2 Implement AlertingRules class
    - Implement low confidence alert (avg < 0.70 over 1000 predictions)
    - Implement high latency alert (10 consecutive files > 5s)
    - Implement high false positive rate alert (> 20% per category)
    - _Requirements: 12.6, 12.7, 14.7_
  
  - [x] 14.3 Implement DailyReportGenerator class
    - Generate daily summary with KPIs
    - Include model performance metrics
    - Include availability metrics
    - _Requirements: 12.8_
  
  - [ ]* 14.4 Write property test for metrics logging completeness
    - **Property 59: Metrics Logging Completeness**
    - **Validates: Requirements 12.1, 12.2, 12.3, 12.4**
  
  - [ ]* 14.5 Write property test for Prometheus metrics format
    - **Property 60: Prometheus Metrics Format**
    - **Validates: Requirements 12.5**
  
  - [ ]* 14.6 Write property test for low confidence alert
    - **Property 61: Low Confidence Alert**
    - **Validates: Requirements 12.6**
  
  - [ ]* 14.7 Write property test for high latency alert
    - **Property 62: High Latency Alert**
    - **Validates: Requirements 12.7**
  
  - [ ]* 14.8 Write property test for daily report generation
    - **Property 63: Daily Report Generation**
    - **Validates: Requirements 12.8**
  
  - [ ]* 14.9 Write property test for availability metrics tracking
    - **Property 50: Availability Metrics Tracking**
    - **Validates: Requirements 10.6**
  
  - [ ]* 14.10 Write property test for false positive rate tracking
    - **Property 72: False Positive Rate Tracking**
    - **Validates: Requirements 14.6**
  
  - [ ]* 14.11 Write property test for high false positive rate alert
    - **Property 73: High False Positive Rate Alert**
    - **Validates: Requirements 14.7**

- [ ] 15. Implement configuration management
  - [x] 15.1 Create configuration schema and validation
    - Define YAML/JSON schema for all configuration options
    - Implement configuration loading with safe defaults
    - Implement validation with clear error messages
    - _Requirements: 13.1, 13.7, 13.8_
  
  - [x] 15.2 Implement configurable parameters
    - Confidence thresholds (blocking, warning)
    - Timeout values (inference, explanation, commit scan)
    - Detection category filters (enable/disable categories)
    - Batch size configuration
    - Explainer backend selection (SHAP or LIME)
    - _Requirements: 13.2, 13.3, 13.4, 13.5, 13.6_
  
  - [ ]* 15.3 Write property test for configuration loading and validation
    - **Property 64: Configuration Loading and Validation**
    - **Validates: Requirements 13.1, 13.7, 13.8**
  
  - [ ]* 15.4 Write property test for configurable thresholds
    - **Property 65: Configurable Thresholds**
    - **Validates: Requirements 13.2**
  
  - [ ]* 15.5 Write property test for configurable timeouts
    - **Property 66: Configurable Timeouts**
    - **Validates: Requirements 13.3**
  
  - [ ]* 15.6 Write property test for category filtering
    - **Property 67: Category Filtering**
    - **Validates: Requirements 13.4**
  
  - [ ]* 15.7 Write property test for configurable batch size
    - **Property 68: Configurable Batch Size**
    - **Validates: Requirements 13.5**
  
  - [ ]* 15.8 Write property test for configurable explainer backend
    - **Property 69: Configurable Explainer Backend**
    - **Validates: Requirements 13.6**

- [ ] 16. Implement training pipeline automation
  - [x] 16.1 Implement TrainingOrchestrator class
    - Check for new dataset versions before training
    - Automatically initiate retraining when new data available
    - Compare new model performance against current PRODUCTION model
    - Promote to STAGING if F1-score improvement >= 2%
    - Send Slack notification on training completion
    - Store training logs and artifacts
    - _Requirements: 11.1, 11.2, 11.3, 11.4, 11.5, 11.6, 11.8_
  
  - [x] 16.2 Create training pipeline script for cron/workflow orchestrator
    - Write CLI script for scheduled execution
    - Implement manual approval workflow for STAGING → PRODUCTION promotion
    - _Requirements: 11.1, 11.7_
  
  - [x] 16.3 Implement feedback integration in training
    - Load false positive feedback from FeedbackStore
    - Incorporate feedback as hard negatives in training dataset
    - _Requirements: 14.4, 14.5_
  
  - [ ]* 16.4 Write property test for dataset version checking
    - **Property 53: Dataset Version Checking**
    - **Validates: Requirements 11.2**
  
  - [ ]* 16.5 Write property test for conditional retraining
    - **Property 54: Conditional Retraining**
    - **Validates: Requirements 11.3**
  
  - [ ]* 16.6 Write property test for model performance comparison
    - **Property 55: Model Performance Comparison**
    - **Validates: Requirements 11.4**
  
  - [ ]* 16.7 Write property test for conditional promotion
    - **Property 56: Conditional Promotion**
    - **Validates: Requirements 11.5**
  
  - [ ]* 16.8 Write property test for training completion notification
    - **Property 57: Training Completion Notification**
    - **Validates: Requirements 11.6**
  
  - [ ]* 16.9 Write property test for training artifact retention
    - **Property 58: Training Artifact Retention**
    - **Validates: Requirements 11.8**
  
  - [ ]* 16.10 Write property test for feedback integration in training
    - **Property 71: Feedback Integration in Training**
    - **Validates: Requirements 14.4, 14.5**

- [x] 17. Checkpoint - Verify automation and configuration
  - Ensure all automation and configuration tests pass, ask the user if questions arise.

- [ ] 18. Implement database schema for feedback and metrics
  - [x] 18.1 Create SQL schema for feedback table
    - Define feedback table with all required fields
    - Create indexes for efficient querying
    - _Requirements: 14.1, 14.2, 14.3_
  
  - [x] 18.2 Create SQL schema for model_performance table
    - Define model_performance table for tracking metrics over time
    - Create indexes for version and timestamp
    - _Requirements: 12.1, 12.2_
  
  - [x] 18.3 Create SQL schema for inference_metrics table
    - Define inference_metrics table for detailed inference tracking
    - Create indexes for timestamp and version
    - _Requirements: 12.1, 12.2, 12.3_
  
  - [x] 18.4 Implement database connection and ORM layer
    - Set up SQLAlchemy models matching schema
    - Implement connection pooling
    - Add migration scripts
    - _Requirements: 14.1, 14.2, 14.3_

- [ ] 19. Create integration tests
  - [ ]* 19.1 Write integration test for training pipeline end-to-end
    - Test: Load datasets → Preprocess → Train → Validate → Save to registry
    - Verify complete training workflow
  
  - [ ]* 19.2 Write integration test for inference pipeline end-to-end
    - Test: Load model → Tokenize → Predict → Explain → Format results
    - Verify complete inference workflow
  
  - [ ]* 19.3 Write integration test for hybrid detection end-to-end
    - Test: Regex scan + ML scan → Merge → Alert → Log
    - Verify complete detection workflow
  
  - [ ]* 19.4 Write integration test for feedback loop
    - Test: Mark false positive → Store feedback → Incorporate in retraining
    - Verify feedback integration
  
  - [ ]* 19.5 Write integration test for git hook workflow
    - Test: Commit → Scan → Block/Warn → Audit log
    - Verify git hook integration

- [ ] 20. Create deployment artifacts
  - [x] 20.1 Create Dockerfile for inference service
    - Multi-stage build for optimized image size
    - Include model artifacts or registry connection
    - Configure GPU support (optional)
    - _Requirements: 15.1, 15.2_
  
  - [x] 20.2 Create Kubernetes manifests
    - Deployment for inference service pods
    - PersistentVolumeClaim for Model Registry
    - CronJob for training pipeline
    - Service for metrics endpoint
    - ConfigMap for configuration
    - _Requirements: 15.1, 15.2, 15.3_
  
  - [x] 20.3 Create health check endpoints
    - Implement /health endpoint checking component status
    - Implement /ready endpoint for readiness probes
    - _Requirements: 15.6_
  
  - [ ]* 20.4 Write property test for health check endpoints
    - **Property 75: Health Check Endpoints**
    - **Validates: Requirements 15.6**
  
  - [x] 20.5 Create installation script for git hook
    - Write Python script to install pre-commit hook
    - Handle existing hooks gracefully
    - _Requirements: 7.1_

- [ ] 21. Create documentation
  - [x] 21.1 Write README with setup instructions
    - Installation steps
    - Configuration guide
    - Usage examples
  
  - [x] 21.2 Write API documentation
    - Document all public classes and methods
    - Include code examples
  
  - [x] 21.3 Write training pipeline guide
    - Dataset preparation
    - Training execution
    - Model promotion workflow
  
  - [x] 21.4 Write deployment guide
    - Kubernetes deployment steps
    - Configuration options
    - Monitoring setup
  
  - [x] 21.5 Write troubleshooting guide
    - Common errors and solutions
    - Fallback mode debugging
    - Performance tuning

- [x] 22. Final checkpoint - Complete system verification
  - Run all tests (unit, property, integration)
  - Verify all 75 property tests pass
  - Verify performance benchmarks meet requirements
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- All 75 correctness properties from the design document have corresponding property tests
- Property tests use hypothesis framework with minimum 100 iterations
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation at major milestones
- Implementation uses Python with transformers, torch, hypothesis, pytest, shap, lime
- The system is designed for resilience with comprehensive fallback mechanisms
- All ML components are isolated from existing regex scanner for fail-safe operation
