# Design Document: ML-Enhanced Secret Scanner with CodeBERT

## Overview

This design document specifies the architecture and implementation details for enhancing the existing GitHub Secret Scanner with machine learning capabilities using CodeBERT transformers. The enhancement adds intelligent, context-aware secret and PII detection that complements the existing regex-based pattern matching system.

### System Goals

1. **Hybrid Detection**: Combine regex-based and ML-based detection for higher accuracy
2. **Explainability**: Provide interpretable explanations for ML detections using SHAP/LIME
3. **Real-Time Performance**: Maintain sub-10-second scan times for typical commits
4. **Resilience**: Gracefully degrade to regex-only mode when ML components fail
5. **Continuous Improvement**: Support automated retraining and model versioning

### Key Design Principles

- **Separation of Concerns**: ML components are isolated from existing regex scanner
- **Fail-Safe Operation**: System remains functional even if ML components fail
- **Observability**: Comprehensive metrics and logging for production monitoring
- **Reproducibility**: Versioned models and datasets for audit trails

## Architecture

### High-Level System Architecture

```mermaid
graph TB
    subgraph "Git Workflow Integration"
        A[Git Pre-Commit Hook] --> B[Hybrid Detector]
    end
    
    subgraph "Detection Layer"
        B --> C[Regex Scanner]
        B --> D[ML Scanner]
        C --> E[Result Merger]
        D --> E
    end
    
    subgraph "ML Inference Stack"
        D --> F[Inference Engine]
        F --> G[CodeBERT Model]
        F --> H[Model Registry]
        D --> I[Explainer SHAP/LIME]
    end
    
    subgraph "Training Pipeline"
        J[Dataset Loader] --> K[Training Pipeline]
        K --> L[Model Trainer]
        L --> H
        K --> M[Performance Validator]
    end
    
    subgraph "Alerting & Feedback"
        E --> N[Alert Manager]
        N --> O[Slack Webhook]
        N --> P[Feedback Store]
        P --> J
    end
    
    subgraph "Monitoring"
        F --> Q[Metrics Exporter]
        Q --> R[Prometheus/Grafana]
    end
    
    style B fill:#ff9999
    style D fill:#99ccff
    style F fill:#99ccff
    style K fill:#99ff99
```

### Component Interaction Flow

**Inference Flow (Real-Time Scanning)**:
1. Developer commits code → Git hook triggers
2. Hybrid Detector invokes both Regex Scanner and ML Scanner in parallel
3. ML Scanner tokenizes code → Inference Engine loads model → Generates predictions
4. Explainer generates token attributions for ML detections
5. Result Merger deduplicates and ranks findings
6. Alert Manager sends Slack notifications with explanations
7. Metrics Exporter logs performance data

**Training Flow (Periodic Retraining)**:
1. Dataset Loader fetches GitHub/Kaggle datasets
2. Training Pipeline preprocesses and augments data
3. Model Trainer fine-tunes CodeBERT
4. Performance Validator evaluates on test set
5. Model Registry stores versioned artifacts
6. Manual approval promotes model to production

### Deployment Architecture

```mermaid
graph LR
    subgraph "Developer Workstation"
        A[Git Hook Script] --> B[Scanner CLI]
    end
    
    subgraph "Kubernetes Cluster"
        C[Inference Service Pod] --> D[Model Registry PVC]
        E[Training Job CronJob] --> D
        F[Metrics Service] --> G[Prometheus]
    end
    
    subgraph "External Services"
        H[Slack API]
        I[GitHub Datasets]
        J[Kaggle Datasets]
    end
    
    B -.HTTP.-> C
    C -.Webhook.-> H
    E -.Download.-> I
    E -.Download.-> J
    
    style C fill:#99ccff
    style E fill:#99ff99
```

## Components and Interfaces

### 1. Training Pipeline

**Responsibility**: Automate dataset loading, preprocessing, model training, and validation.

**Key Classes**:
- `DatasetLoader`: Downloads and validates GitHub/Kaggle datasets
- `DataPreprocessor`: Tokenizes, balances, and augments training data
- `ModelTrainer`: Fine-tunes CodeBERT on prepared datasets
- `PerformanceValidator`: Evaluates model on test set

**Interfaces**:

```python
class DatasetLoader:
    def load_github_datasets(self, repo_urls: List[str]) -> Dataset:
        """Load labeled secret examples from GitHub repositories."""
        pass
    
    def load_kaggle_datasets(self, dataset_ids: List[str]) -> Dataset:
        """Load labeled PII examples from Kaggle."""
        pass
    
    def validate_checksums(self, dataset: Dataset) -> bool:
        """Verify dataset integrity with checksums."""
        pass
    
    def split_dataset(self, dataset: Dataset, 
                     train_ratio: float = 0.7,
                     val_ratio: float = 0.15,
                     test_ratio: float = 0.15) -> Tuple[Dataset, Dataset, Dataset]:
        """Split dataset into train/val/test sets."""
        pass

class DataPreprocessor:
    def tokenize(self, texts: List[str], max_length: int = 512) -> BatchEncoding:
        """Tokenize code samples using CodeBERT tokenizer."""
        pass
    
    def balance_classes(self, dataset: Dataset, max_ratio: float = 3.0) -> Dataset:
        """Balance classes to prevent bias."""
        pass
    
    def augment_data(self, dataset: Dataset) -> Dataset:
        """Generate synthetic examples for minority classes."""
        pass

class ModelTrainer:
    def fine_tune(self, model: PreTrainedModel, 
                  train_dataset: Dataset,
                  val_dataset: Dataset,
                  epochs: int = 3,
                  learning_rate: float = 2e-5) -> PreTrainedModel:
        """Fine-tune CodeBERT model on training dataset."""
        pass
    
    def save_model(self, model: PreTrainedModel, 
                   version: str,
                   metadata: dict) -> str:
        """Save trained model to Model Registry."""
        pass

class PerformanceValidator:
    def evaluate(self, model: PreTrainedModel, 
                test_dataset: Dataset) -> dict:
        """Evaluate model performance on test set.
        
        Returns:
            dict with keys: precision, recall, f1_score, confusion_matrix
        """
        pass
    
    def meets_threshold(self, metrics: dict,
                       min_precision: float = 0.90,
                       min_recall: float = 0.85) -> bool:
        """Check if model meets minimum performance thresholds."""
        pass
```

**Configuration**:
```yaml
training:
  datasets:
    github_repos:
      - "https://github.com/dxa4481/truffleHogRegexes"
      - "https://github.com/awslabs/git-secrets"
    kaggle_datasets:
      - "lakshmi25npathi/imdb-dataset-of-50k-movie-reviews"
  
  preprocessing:
    max_sequence_length: 512
    class_balance_ratio: 3.0
    augmentation_enabled: true
  
  training_params:
    epochs: 3
    batch_size: 16
    learning_rate: 2e-5
    warmup_steps: 500
  
  validation:
    min_precision: 0.90
    min_recall: 0.85
```

### 2. Inference Engine

**Responsibility**: Execute ML-based secret detection at runtime with GPU/CPU optimization.

**Key Classes**:
- `InferenceEngine`: Manages model loading and prediction execution
- `ModelCache`: Caches loaded models in memory for reuse
- `BatchProcessor`: Handles batch inference for multiple files

**Interfaces**:

```python
class InferenceEngine:
    def __init__(self, model_registry: ModelRegistry, 
                 device: str = "auto"):
        """Initialize inference engine.
        
        Args:
            model_registry: Model registry for loading models
            device: "cpu", "cuda", or "auto" for automatic detection
        """
        pass
    
    def load_model(self, version: str = "latest") -> PreTrainedModel:
        """Load model from registry into memory."""
        pass
    
    def predict(self, code_text: str) -> List[Detection]:
        """Run inference on a single code snippet.
        
        Returns:
            List of Detection objects with confidence scores
        """
        pass
    
    def predict_batch(self, code_texts: List[str]) -> List[List[Detection]]:
        """Run batch inference on multiple code snippets."""
        pass
    
    def get_metrics(self) -> dict:
        """Return inference performance metrics."""
        pass

@dataclass
class Detection:
    """Represents an ML-detected secret."""
    text: str
    start_pos: int
    end_pos: int
    category: str  # API_KEY, PASSWORD, PII, TOKEN, CERTIFICATE, OTHER
    confidence_score: float  # 0.0 to 1.0
    token_attributions: Optional[List[Tuple[str, float]]] = None
```

**Performance Optimizations**:
- Model quantization (FP16 or INT8) to reduce memory footprint
- Connection pooling for Model Registry access
- Batch processing for multiple files
- GPU acceleration when available
- Model caching to avoid repeated loading

### 3. ML Scanner

**Responsibility**: Integrate Inference Engine with existing scanner architecture.

**Key Classes**:
- `MLScanner`: Main interface for ML-based detection
- `CodeTokenizer`: Wraps CodeBERT tokenizer
- `ResultFormatter`: Converts ML detections to Finding objects

**Interfaces**:

```python
class MLScanner:
    def __init__(self, config: dict, inference_engine: InferenceEngine):
        """Initialize ML scanner with configuration."""
        pass
    
    def scan_file(self, file_path: str, timeout: int = 5) -> List[Finding]:
        """Scan a single file for secrets using ML.
        
        Args:
            file_path: Path to file to scan
            timeout: Maximum time in seconds (default 5)
            
        Returns:
            List of Finding objects with ML detections
        """
        pass
    
    def scan_text(self, text: str, file_path: str = "") -> List[Finding]:
        """Scan text content for secrets."""
        pass
    
    def is_available(self) -> bool:
        """Check if ML scanner is operational."""
        pass

class CodeTokenizer:
    def __init__(self, model_name: str = "microsoft/codebert-base"):
        """Initialize CodeBERT tokenizer."""
        pass
    
    def tokenize(self, code: str, max_length: int = 512) -> BatchEncoding:
        """Tokenize code for model input."""
        pass
    
    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs back to text."""
        pass
```

### 4. Hybrid Detector

**Responsibility**: Orchestrate both regex and ML scanners, merge results, and deduplicate.

**Key Classes**:
- `HybridDetector`: Main orchestrator
- `ResultMerger`: Deduplicates and ranks findings
- `FallbackManager`: Handles ML component failures

**Interfaces**:

```python
class HybridDetector:
    def __init__(self, regex_scanner: FileScanner, 
                 ml_scanner: MLScanner,
                 config: dict):
        """Initialize hybrid detector."""
        pass
    
    def scan_file(self, file_path: str) -> List[Finding]:
        """Scan file using both regex and ML scanners.
        
        Returns:
            Merged and deduplicated findings
        """
        pass
    
    def scan_files(self, file_paths: List[str]) -> ScanResult:
        """Scan multiple files."""
        pass
    
    def is_in_fallback_mode(self) -> bool:
        """Check if operating in fallback mode (regex-only)."""
        pass

class ResultMerger:
    def merge(self, regex_findings: List[Finding],
             ml_findings: List[Finding]) -> List[Finding]:
        """Merge and deduplicate findings from both scanners.
        
        Logic:
        - If same secret detected by both: Use higher confidence, mark as HIGH_CONFIDENCE
        - If detected by ML only: Mark as ML_ONLY, require confidence >= 0.85
        - If detected by regex only: Mark as REGEX_ONLY
        """
        pass
    
    def deduplicate(self, findings: List[Finding]) -> List[Finding]:
        """Remove duplicate findings based on file, line, and text."""
        pass
```

**Merging Logic**:

```python
def merge_findings(regex_findings, ml_findings):
    merged = []
    
    # Create lookup for regex findings
    regex_map = {(f.file_path, f.line_number, f.matched_text): f 
                 for f in regex_findings}
    
    # Create lookup for ML findings
    ml_map = {(f.file_path, f.line_number, f.matched_text): f 
              for f in ml_findings}
    
    # Find overlapping detections
    regex_keys = set(regex_map.keys())
    ml_keys = set(ml_map.keys())
    
    # Both scanners detected (HIGH_CONFIDENCE)
    for key in regex_keys & ml_keys:
        regex_f = regex_map[key]
        ml_f = ml_map[key]
        
        # Use higher confidence score
        if ml_f.confidence_score > 0.9:
            finding = ml_f
            finding.source = "HIGH_CONFIDENCE"
        else:
            finding = regex_f
            finding.source = "HIGH_CONFIDENCE"
        
        merged.append(finding)
    
    # ML only (require high confidence)
    for key in ml_keys - regex_keys:
        ml_f = ml_map[key]
        if ml_f.confidence_score >= 0.85:
            ml_f.source = "ML_ONLY"
            merged.append(ml_f)
    
    # Regex only
    for key in regex_keys - ml_keys:
        regex_f = regex_map[key]
        regex_f.source = "REGEX_ONLY"
        merged.append(regex_f)
    
    return merged
```

### 5. Explainer

**Responsibility**: Generate interpretable explanations for ML detections using SHAP or LIME.

**Key Classes**:
- `Explainer`: Main interface for generating explanations
- `SHAPExplainer`: SHAP-based implementation
- `LIMEExplainer`: LIME-based implementation

**Interfaces**:

```python
class Explainer(ABC):
    @abstractmethod
    def explain(self, model: PreTrainedModel,
               text: str,
               prediction: Detection) -> Explanation:
        """Generate explanation for a detection."""
        pass

class SHAPExplainer(Explainer):
    def __init__(self, model: PreTrainedModel, tokenizer: CodeTokenizer):
        """Initialize SHAP explainer."""
        pass
    
    def explain(self, model: PreTrainedModel,
               text: str,
               prediction: Detection) -> Explanation:
        """Generate SHAP-based explanation.
        
        Returns:
            Explanation with token attributions
        """
        pass

class LIMEExplainer(Explainer):
    def __init__(self, model: PreTrainedModel, tokenizer: CodeTokenizer):
        """Initialize LIME explainer."""
        pass
    
    def explain(self, model: PreTrainedModel,
               text: str,
               prediction: Detection) -> Explanation:
        """Generate LIME-based explanation."""
        pass

@dataclass
class Explanation:
    """Represents an explanation for a detection."""
    detection: Detection
    token_attributions: List[Tuple[str, float]]  # (token, attribution_score)
    top_tokens: List[str]  # Top 5 most influential tokens
    formatted_text: str  # Human-readable explanation with highlights
```

**Explanation Format Example**:
```
Detection: API_KEY (confidence: 0.92)
Top influential tokens:
  1. "AKIA" (0.35) ████████████████████
  2. "I44QH8" (0.22) ████████████
  3. "DHBEX" (0.18) ██████████
  4. "AMPLE" (0.15) ████████
  5. "=" (0.10) █████

Highlighted code:
  api_key = "[AKIA]I44QH8[DHBEX][AMPLE]"
            ^^^^^^       ^^^^^  ^^^^^
```

### 6. Model Registry

**Responsibility**: Version and manage trained CodeBERT models with metadata.

**Key Classes**:
- `ModelRegistry`: Main interface for model storage and retrieval
- `ModelMetadata`: Metadata for each model version
- `ModelPromoter`: Handles promotion between environments

**Interfaces**:

```python
class ModelRegistry:
    def __init__(self, storage_path: str):
        """Initialize model registry.
        
        Args:
            storage_path: Base path for model storage (local or cloud)
        """
        pass
    
    def register_model(self, model: PreTrainedModel,
                      version: str,
                      metadata: ModelMetadata) -> str:
        """Register a new model version.
        
        Returns:
            Model ID
        """
        pass
    
    def get_model(self, version: str = "latest",
                 environment: str = "production") -> PreTrainedModel:
        """Retrieve model by version or environment tag."""
        pass
    
    def list_models(self, limit: int = 10) -> List[ModelMetadata]:
        """List available model versions."""
        pass
    
    def promote_model(self, version: str,
                     from_env: str,
                     to_env: str) -> bool:
        """Promote model between environments."""
        pass
    
    def get_last_known_good(self) -> PreTrainedModel:
        """Get last known good model for fallback."""
        pass

@dataclass
class ModelMetadata:
    """Metadata for a model version."""
    version: str  # Semantic version (MAJOR.MINOR.PATCH)
    training_date: datetime
    dataset_version: str
    performance_metrics: dict  # precision, recall, f1_score
    environment: str  # STAGING or PRODUCTION
    training_config: dict
    lineage: dict  # Links to training datasets
```

**Storage Structure**:
```
model_registry/
├── models/
│   ├── 1.0.0/
│   │   ├── model.bin
│   │   ├── config.json
│   │   ├── tokenizer_config.json
│   │   └── metadata.json
│   ├── 1.0.1/
│   └── 1.1.0/
├── environments/
│   ├── staging.json  # Points to current staging model
│   └── production.json  # Points to current production model
└── cache/
    └── last_known_good.json
```

### 7. Alert Manager

**Responsibility**: Send enhanced Slack notifications with ML insights and handle feedback.

**Key Classes**:
- `AlertManager`: Main interface for alerting
- `SlackFormatter`: Formats messages with Block Kit
- `FeedbackCollector`: Collects false positive feedback

**Interfaces**:

```python
class AlertManager:
    def __init__(self, webhook_url: str, feedback_store: FeedbackStore):
        """Initialize alert manager."""
        pass
    
    def send_alert(self, findings: List[Finding],
                  explanations: Dict[str, Explanation],
                  context: dict) -> bool:
        """Send Slack alert with ML insights.
        
        Args:
            findings: Detected secrets
            explanations: Explanations for ML detections
            context: Repo, branch, author info
        """
        pass
    
    def mark_false_positive(self, finding_id: str,
                           user: str,
                           reason: str) -> bool:
        """Mark a detection as false positive."""
        pass

class SlackFormatter:
    def format_message(self, findings: List[Finding],
                      explanations: Dict[str, Explanation],
                      context: dict) -> dict:
        """Format Slack message with Block Kit.
        
        Returns:
            Slack Block Kit JSON payload
        """
        pass
```

**Enhanced Slack Message Format**:
```json
{
  "blocks": [
    {
      "type": "header",
      "text": {"type": "plain_text", "text": "🚨 Secrets Detected"}
    },
    {
      "type": "section",
      "fields": [
        {"type": "mrkdwn", "text": "*Confidence:* HIGH_CONFIDENCE"},
        {"type": "mrkdwn", "text": "*Source:* ML + Regex"},
        {"type": "mrkdwn", "text": "*Score:* 0.92"}
      ]
    },
    {
      "type": "section",
      "text": {
        "type": "mrkdwn",
        "text": "*Explanation:*\nTop tokens: `AKIA` (0.35), `I44QH8` (0.22)"
      }
    },
    {
      "type": "actions",
      "elements": [
        {
          "type": "button",
          "text": {"type": "plain_text", "text": "Mark False Positive"},
          "action_id": "mark_fp"
        },
        {
          "type": "button",
          "text": {"type": "plain_text", "text": "View Full Report"},
          "action_id": "view_report"
        }
      ]
    }
  ]
}
```

### 8. Git Hook Integration

**Responsibility**: Integrate Hybrid Detector into git pre-commit workflow.

**Key Classes**:
- `GitHookIntegration`: Main interface for git hook
- `CommitScanner`: Scans only modified files in commit
- `TerminalFormatter`: Formats output for terminal display

**Interfaces**:

```python
class GitHookIntegration:
    def __init__(self, hybrid_detector: HybridDetector, config: dict):
        """Initialize git hook integration."""
        pass
    
    def scan_commit(self, timeout: int = 10) -> ScanResult:
        """Scan modified files in current commit.
        
        Returns:
            ScanResult with findings
        """
        pass
    
    def should_block_commit(self, result: ScanResult) -> bool:
        """Determine if commit should be blocked.
        
        Logic:
        - Block if HIGH_CONFIDENCE secrets found
        - Warn but allow if ML_ONLY with confidence < 0.90
        """
        pass
    
    def display_results(self, result: ScanResult) -> None:
        """Display scan results in terminal with explanations."""
        pass
```

**Git Hook Script** (`pre-commit`):
```bash
#!/bin/bash
# ML-Enhanced Secret Scanner Pre-Commit Hook

python3 -m ml_scanner.git_hook --mode pre-commit

exit_code=$?

if [ $exit_code -eq 1 ]; then
    echo "❌ Commit blocked: HIGH confidence secrets detected"
    exit 1
elif [ $exit_code -eq 2 ]; then
    echo "⚠️  Warning: ML scanner unavailable, using regex-only mode"
    # Don't block on scanner errors
    exit 0
else
    echo "✅ No secrets detected"
    exit 0
fi
```

## Data Models

### Core Data Structures

```python
@dataclass
class Finding:
    """Represents a detected secret (unified for regex and ML)."""
    file_path: str
    line_number: int
    matched_text: str
    secret_type: str
    confidence: str  # 'HIGH' or 'LOW'
    entropy: float
    is_comment: bool
    source: str  # 'REGEX_ONLY', 'ML_ONLY', 'HIGH_CONFIDENCE'
    confidence_score: Optional[float] = None  # ML confidence (0.0-1.0)
    explanation: Optional[Explanation] = None  # ML explanation

@dataclass
class Detection:
    """ML-specific detection before conversion to Finding."""
    text: str
    start_pos: int
    end_pos: int
    category: str  # API_KEY, PASSWORD, PII, TOKEN, CERTIFICATE, OTHER
    confidence_score: float
    token_attributions: Optional[List[Tuple[str, float]]] = None

@dataclass
class Explanation:
    """Explanation for an ML detection."""
    detection: Detection
    token_attributions: List[Tuple[str, float]]
    top_tokens: List[str]
    formatted_text: str

@dataclass
class ModelMetadata:
    """Metadata for a trained model."""
    version: str
    training_date: datetime
    dataset_version: str
    performance_metrics: dict
    environment: str
    training_config: dict
    lineage: dict

@dataclass
class ScanResult:
    """Aggregated scan results."""
    findings: List[Finding]
    files_scanned: int
    high_confidence_count: int
    low_confidence_count: int
    execution_time_seconds: float
    ml_available: bool
    fallback_mode: bool

@dataclass
class Dataset:
    """Training dataset."""
    texts: List[str]
    labels: List[int]
    metadata: dict
    source: str  # 'github' or 'kaggle'
    version: str
    checksum: str

@dataclass
class TrainingMetrics:
    """Training performance metrics."""
    precision: float
    recall: float
    f1_score: float
    confusion_matrix: List[List[int]]
    loss: float
    epoch: int

@dataclass
class FeedbackEntry:
    """False positive feedback."""
    finding_id: str
    file_path: str
    matched_text: str
    secret_type: str
    timestamp: datetime
    user: str
    reason: str
    detection_source: str
```

### Database Schema (Feedback Store)

```sql
-- Feedback table for false positives
CREATE TABLE feedback (
    id UUID PRIMARY KEY,
    finding_id VARCHAR(255) NOT NULL,
    file_path TEXT NOT NULL,
    matched_text TEXT NOT NULL,
    secret_type VARCHAR(100) NOT NULL,
    detection_source VARCHAR(50) NOT NULL,  -- REGEX_ONLY, ML_ONLY, HIGH_CONFIDENCE
    confidence_score FLOAT,
    timestamp TIMESTAMP NOT NULL,
    user_email VARCHAR(255),
    reason TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_feedback_timestamp ON feedback(timestamp);
CREATE INDEX idx_feedback_secret_type ON feedback(secret_type);
CREATE INDEX idx_feedback_source ON feedback(detection_source);

-- Model performance tracking
CREATE TABLE model_performance (
    id UUID PRIMARY KEY,
    model_version VARCHAR(50) NOT NULL,
    metric_name VARCHAR(100) NOT NULL,
    metric_value FLOAT NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    environment VARCHAR(50) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_performance_version ON model_performance(model_version);
CREATE INDEX idx_performance_timestamp ON model_performance(timestamp);

-- Inference metrics
CREATE TABLE inference_metrics (
    id UUID PRIMARY KEY,
    model_version VARCHAR(50) NOT NULL,
    file_path TEXT,
    inference_time_ms FLOAT NOT NULL,
    confidence_score FLOAT,
    detection_category VARCHAR(100),
    timestamp TIMESTAMP NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_inference_timestamp ON inference_metrics(timestamp);
CREATE INDEX idx_inference_version ON inference_metrics(model_version);
```


## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system—essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*


### Property Reflection

After analyzing all 120 acceptance criteria, I identified several areas of redundancy:

1. **Dataset Loading (1.1, 1.2, 2.1, 2.2)**: These can be combined into a single property about loading datasets from various sources
2. **Tokenization (1.4, 3.1)**: Both test tokenization behavior and can be combined
3. **Model Storage/Retrieval (1.7, 3.2, 6.6)**: These test round-trip model persistence and can be consolidated
4. **Scanner Invocation (4.1, 4.2)**: Both test that scanners are invoked and can be combined
5. **Message Content (9.2, 9.3, 9.4, 9.5, 9.7)**: All test that Slack messages contain required fields - can be one property
6. **Fallback Behavior (10.1, 10.2, 10.3)**: These describe the same fallback mode and can be combined
7. **Metrics Logging (12.1, 12.2, 12.3, 12.4)**: All test that various metrics are logged - can be consolidated
8. **Configuration Loading (13.1, 13.7, 13.8)**: These test configuration validation and can be combined

After consolidation, we have approximately 60 unique testable properties.

### Property 1: Dataset Loading Round Trip

*For any* dataset source (GitHub or Kaggle), loading a dataset and then accessing its contents should preserve all examples and maintain proper structure (texts, labels, metadata).

**Validates: Requirements 1.1, 1.2, 2.1, 2.2**

### Property 2: Dataset Combination Preserves Examples

*For any* set of datasets, combining them into a unified dataset should preserve all examples from all source datasets without loss.

**Validates: Requirements 1.3**

### Property 3: Tokenization Length Constraint

*For any* code sample, tokenizing it with CodeBERT tokenizer should produce output with length ≤ 512 tokens.

**Validates: Requirements 1.4, 3.1**

### Property 4: Model Persistence Round Trip

*For any* trained CodeBERT model, saving it to the Model Registry and then loading it back should produce an equivalent model that generates the same predictions.

**Validates: Requirements 1.7, 3.2, 6.6**

### Property 5: Checksum Validation

*For any* dataset with a valid checksum, validation should pass; for any dataset with an invalid checksum, validation should fail.

**Validates: Requirements 2.3**

### Property 6: Dataset Split Proportions

*For any* dataset, splitting it into train/val/test sets should produce sets with sizes approximately matching the specified ratios (70%/15%/15% with ±2% tolerance).

**Validates: Requirements 2.4**

### Property 7: Class Balancing Ratio

*For any* imbalanced dataset, applying class balancing should produce a dataset where the class ratio does not exceed the specified maximum ratio (3:1).

**Validates: Requirements 2.5**

### Property 8: Conditional Augmentation

*For any* dataset with class imbalance exceeding 3:1, augmentation should be applied; for any dataset with ratio ≤ 3:1, augmentation should not be applied.

**Validates: Requirements 2.6**

### Property 9: Dataset Metadata Completeness

*For any* loaded dataset, its metadata should contain all required fields: source, version, and statistics.

**Validates: Requirements 2.7**

### Property 10: Graceful Error Handling

*For any* invalid dataset source or loading failure, the system should log detailed error messages and exit gracefully without crashing.

**Validates: Requirements 2.8**

### Property 11: Confidence Score Range

*For any* ML detection, the confidence score should be in the valid range [0.0, 1.0].

**Validates: Requirements 3.4**

### Property 12: Detection Category Validity

*For any* ML detection, the category should be one of the valid values: API_KEY, PASSWORD, PII, TOKEN, CERTIFICATE, or OTHER.

**Validates: Requirements 3.5**

### Property 13: Inference Time Constraint

*For any* file up to 10,000 lines, ML inference should complete within 5 seconds.

**Validates: Requirements 3.6**

### Property 14: Hybrid Scanner Invocation

*For any* file scanned by the Hybrid Detector, both the Regex Scanner and ML Scanner should be invoked (unless in fallback mode).

**Validates: Requirements 4.1, 4.2**

### Property 15: Result Deduplication

*For any* set of findings from both scanners, the merged result should contain no duplicate findings (same file, line, and text).

**Validates: Requirements 4.3**

### Property 16: Confidence Score Selection

*For any* secret detected by both scanners, the merged finding should use the higher confidence score of the two.

**Validates: Requirements 4.4**

### Property 17: High Confidence Attribution

*For any* secret detected by both scanners, the merged finding should be flagged as HIGH_CONFIDENCE.

**Validates: Requirements 4.5**

### Property 18: ML-Only Confidence Threshold

*For any* secret detected only by ML Scanner, it should only appear in the final results if its confidence score is ≥ 0.85.

**Validates: Requirements 4.6**

### Property 19: Source Attribution Completeness

*For any* finding in the final results, it should have a valid source attribution: REGEX_ONLY, ML_ONLY, or HIGH_CONFIDENCE.

**Validates: Requirements 4.7, 4.8**

### Property 20: Token Attribution Completeness

*For any* ML detection with explanation, the token attributions should cover all tokens in the detected text.

**Validates: Requirements 5.3**

### Property 21: Top Tokens Count

*For any* explanation, it should identify exactly 5 top tokens (or fewer if the text has fewer than 5 tokens).

**Validates: Requirements 5.4**

### Property 22: Attribution Score Normalization

*For any* explanation, the sum of all attribution scores should equal 1.0 (±0.01 tolerance for floating point precision).

**Validates: Requirements 5.5**

### Property 23: Explanation Generation Time

*For any* detection, explanation generation should complete within 2 seconds.

**Validates: Requirements 5.6**

### Property 24: Explanation Format Completeness

*For any* explanation, the formatted text should contain highlighted tokens and be human-readable.

**Validates: Requirements 5.7**

### Property 25: Explanation Error Handling

*For any* detection where explanation generation fails, the system should log the error and return the detection without explanation (not crash).

**Validates: Requirements 5.8**

### Property 26: Semantic Version Format

*For any* model stored in the registry, its version should follow semantic versioning format (MAJOR.MINOR.PATCH).

**Validates: Requirements 6.1**

### Property 27: Model Metadata Completeness

*For any* model in the registry, its metadata should contain all required fields: training_date, dataset_version, and performance_metrics.

**Validates: Requirements 6.2**

### Property 28: Model Promotion

*For any* model in STAGING environment, promoting it to PRODUCTION should update its environment tag to PRODUCTION.

**Validates: Requirements 6.3**

### Property 29: Version Retention

*For any* model registry with more than 5 models, at least the 5 most recent versions should be retained.

**Validates: Requirements 6.4**

### Property 30: Model Format Validation

*For any* model registration attempt, invalid model formats should be rejected with clear error messages.

**Validates: Requirements 6.5**

### Property 31: Model Lineage Tracking

*For any* model in the registry, its metadata should contain lineage information linking it to the training dataset.

**Validates: Requirements 6.7**

### Property 32: Registry Fallback

*For any* failed model retrieval, the registry should return the last known good model version.

**Validates: Requirements 6.8**

### Property 33: Modified Files Only

*For any* git commit scan, only files modified in that commit should be scanned (not all repository files).

**Validates: Requirements 7.2**

### Property 34: High Confidence Blocking

*For any* commit scan with HIGH_CONFIDENCE findings, the git hook should return a non-zero exit code to block the commit.

**Validates: Requirements 7.3**

### Property 35: Low Confidence Warning

*For any* commit scan with only ML_ONLY findings below 0.90 confidence, the git hook should return zero exit code (allow commit with warning).

**Validates: Requirements 7.4**

### Property 36: Commit Scan Time Constraint

*For any* commit with up to 20 modified files, scanning should complete within 10 seconds.

**Validates: Requirements 7.5**

### Property 37: Terminal Output Completeness

*For any* scan with findings, the terminal output should include explanations for ML detections.

**Validates: Requirements 7.6**

### Property 38: Fallback Mode Operation

*For any* scan where ML Scanner is unavailable, the system should enter fallback mode and use only Regex Scanner.

**Validates: Requirements 7.7, 10.1, 10.2, 10.3**

### Property 39: Audit Logging

*For any* scan performed by git hook, the results should be written to the local audit file.

**Validates: Requirements 7.8**

### Property 40: Model Caching

*For any* sequence of multiple inferences, the model should be loaded once at startup and reused (not reloaded for each inference).

**Validates: Requirements 8.1**

### Property 41: Batch Inference Correctness

*For any* batch of files, batch inference should produce the same results as scanning each file individually.

**Validates: Requirements 8.2**

### Property 42: Memory Warning Threshold

*For any* inference operation where memory usage exceeds 1GB, a warning should be logged and caches should be cleared.

**Validates: Requirements 8.7**

### Property 43: Slack Alert Sending

*For any* detected secret, the Alert Manager should send a Slack webhook notification.

**Validates: Requirements 9.1**

### Property 44: Slack Message Completeness

*For any* Slack alert, the message should contain all required fields: confidence_score, detection_source, token_attribution, file_path, line_number, code_snippet, and action_buttons.

**Validates: Requirements 9.2, 9.3, 9.4, 9.5, 9.7**

### Property 45: Slack Block Kit Format

*For any* Slack alert, the message should use valid Slack Block Kit JSON structure.

**Validates: Requirements 9.6**

### Property 46: Slack Retry Logic

*For any* failed Slack webhook call, the system should retry up to 3 times with exponential backoff before giving up.

**Validates: Requirements 9.8**

### Property 47: Fallback Mode Logging

*For any* scan in fallback mode, a warning should be logged indicating ML Scanner is unavailable.

**Validates: Requirements 10.3**

### Property 48: ML Scanner Reinitialization

*For any* system in fallback mode for more than 5 minutes, at least one reinitialization attempt should have been made.

**Validates: Requirements 10.4**

### Property 49: Inference Timeout Handling

*For any* file where ML inference times out after 10 seconds, only Regex Scanner results should be used for that file.

**Validates: Requirements 10.5**

### Property 50: Availability Metrics Tracking

*For any* scan operation, ML Scanner availability metrics should be tracked and reportable.

**Validates: Requirements 10.6**

### Property 51: Registry Cache Fallback

*For any* scenario where Model Registry is unreachable, the Inference Engine should use the cached local model copy.

**Validates: Requirements 10.7**

### Property 52: Resilient Scanning

*For any* ML component failure, scanning should continue using available components (never fail completely).

**Validates: Requirements 10.8**

### Property 53: Dataset Version Checking

*For any* training run, the pipeline should check for new dataset versions before starting training.

**Validates: Requirements 11.2**

### Property 54: Conditional Retraining

*For any* training run where new data is available, retraining should be automatically initiated.

**Validates: Requirements 11.3**

### Property 55: Model Performance Comparison

*For any* newly trained model, its performance metrics should be compared against the current PRODUCTION model.

**Validates: Requirements 11.4**

### Property 56: Conditional Promotion

*For any* newly trained model where F1-score exceeds current PRODUCTION model by ≥ 2%, it should be promoted to STAGING.

**Validates: Requirements 11.5**

### Property 57: Training Completion Notification

*For any* completed training run, a notification should be sent to the configured Slack channel.

**Validates: Requirements 11.6**

### Property 58: Training Artifact Retention

*For any* training run, logs and artifacts should be stored for audit purposes.

**Validates: Requirements 11.8**

### Property 59: Metrics Logging Completeness

*For any* inference operation, the following metrics should be logged: prediction_latency, confidence_score, detection_rate, and false_positive_rate.

**Validates: Requirements 12.1, 12.2, 12.3, 12.4**

### Property 60: Prometheus Metrics Format

*For any* exported metrics, they should be in valid Prometheus-compatible format.

**Validates: Requirements 12.5**

### Property 61: Low Confidence Alert

*For any* sliding window of 1000 predictions where average confidence score drops below 0.70, an alert should be triggered.

**Validates: Requirements 12.6**

### Property 62: High Latency Alert

*For any* sequence of 10 consecutive files where inference latency exceeds 5 seconds, an alert should be triggered.

**Validates: Requirements 12.7**

### Property 63: Daily Report Generation

*For any* day of operation, a summary report with key performance indicators should be generated.

**Validates: Requirements 12.8**

### Property 64: Configuration Loading and Validation

*For any* valid configuration file (YAML or JSON), the ML Scanner should load it successfully; for any invalid configuration, it should use safe defaults and log warnings.

**Validates: Requirements 13.1, 13.7, 13.8**

### Property 65: Configurable Thresholds

*For any* configured confidence threshold, the system should apply it correctly when determining whether to block or warn.

**Validates: Requirements 13.2**

### Property 66: Configurable Timeouts

*For any* configured timeout value, inference operations should respect that timeout.

**Validates: Requirements 13.3**

### Property 67: Category Filtering

*For any* disabled detection category in configuration, detections of that category should not appear in final results.

**Validates: Requirements 13.4**

### Property 68: Configurable Batch Size

*For any* configured batch size, batch inference should use that batch size.

**Validates: Requirements 13.5**

### Property 69: Configurable Explainer Backend

*For any* configured explainer backend (SHAP or LIME), the system should use that explainer for generating explanations.

**Validates: Requirements 13.6**

### Property 70: False Positive Marking

*For any* detection marked as false positive, the feedback should be stored with all required fields: finding_id, file_path, matched_text, secret_type, timestamp, user, and reason.

**Validates: Requirements 14.1, 14.2, 14.3**

### Property 71: Feedback Integration in Training

*For any* retraining run, false positive feedback from the feedback store should be incorporated into the training dataset as hard negatives.

**Validates: Requirements 14.4, 14.5**

### Property 72: False Positive Rate Tracking

*For any* detection category, the false positive rate should be tracked and calculable from feedback data.

**Validates: Requirements 14.6**

### Property 73: High False Positive Rate Alert

*For any* detection category where false positive rate exceeds 20%, a review alert should be triggered.

**Validates: Requirements 14.7**

### Property 74: Feedback Export

*For any* request to export feedback data, the system should provide all feedback entries in a structured format.

**Validates: Requirements 14.8**

### Property 75: Health Check Endpoints

*For any* deployed service, health check endpoints should return correct status (healthy/unhealthy) based on component state.

**Validates: Requirements 15.6**


## Error Handling

### Error Categories and Strategies

#### 1. Training Pipeline Errors

**Dataset Loading Failures**:
- **Cause**: Network issues, invalid URLs, authentication failures
- **Handling**: Log detailed error with source URL, retry with exponential backoff (3 attempts), fail gracefully if all retries exhausted
- **User Impact**: Training cannot proceed, alert sent to ML team

**Checksum Validation Failures**:
- **Cause**: Corrupted downloads, man-in-the-middle attacks
- **Handling**: Reject dataset, log security warning, do not proceed with training
- **User Impact**: Training blocked until valid dataset obtained

**Model Training Failures**:
- **Cause**: OOM errors, GPU failures, convergence issues
- **Handling**: Log full stack trace, save partial checkpoint, alert ML team
- **User Impact**: No new model produced, existing production model continues

**Performance Threshold Failures**:
- **Cause**: Model doesn't meet minimum precision/recall requirements
- **Handling**: Log metrics, do not promote to staging, alert ML team for investigation
- **User Impact**: No model update, existing production model continues

#### 2. Inference Engine Errors

**Model Loading Failures**:
- **Cause**: Corrupted model files, incompatible versions, missing dependencies
- **Handling**: Fall back to last known good model, log error, alert platform team
- **User Impact**: System uses previous model version, scanning continues

**Inference Timeout**:
- **Cause**: Large files, complex code patterns, slow hardware
- **Handling**: Cancel inference after 10 seconds, use only regex results for that file, log timeout
- **User Impact**: ML detection skipped for that file, regex detection still works

**GPU Out of Memory**:
- **Cause**: Batch size too large, model too big for GPU
- **Handling**: Fall back to CPU inference with quantized model, reduce batch size
- **User Impact**: Slower inference but scanning continues

**Invalid Input**:
- **Cause**: Binary files, encoding issues, malformed code
- **Handling**: Skip ML inference for that file, use regex only, log warning
- **User Impact**: ML detection skipped, regex detection still works

#### 3. Model Registry Errors

**Registry Unreachable**:
- **Cause**: Network issues, service downtime, authentication failures
- **Handling**: Use cached local model copy, retry connection every 5 minutes, log error
- **User Impact**: System uses cached model, scanning continues

**Model Not Found**:
- **Cause**: Invalid version requested, model deleted
- **Handling**: Return last known good model, log warning
- **User Impact**: System uses fallback model, scanning continues

**Storage Quota Exceeded**:
- **Cause**: Too many model versions stored
- **Handling**: Delete oldest models beyond retention limit (keep 5 most recent), log cleanup action
- **User Impact**: Old models removed, current operations unaffected

#### 4. Explainer Errors

**Explanation Generation Timeout**:
- **Cause**: Complex models, large inputs
- **Handling**: Cancel explanation after 2 seconds, return detection without explanation, log timeout
- **User Impact**: Detection still reported, explanation missing

**SHAP/LIME Library Errors**:
- **Cause**: Library bugs, incompatible versions
- **Handling**: Log error, return detection without explanation, alert platform team
- **User Impact**: Detection still reported, explanation missing

#### 5. Alert Manager Errors

**Slack Webhook Failures**:
- **Cause**: Network issues, invalid webhook URL, Slack API downtime
- **Handling**: Retry up to 3 times with exponential backoff (1s, 2s, 4s), log failure if all retries exhausted
- **User Impact**: Alert may not reach Slack, but scan results still available locally

**Message Formatting Errors**:
- **Cause**: Invalid Block Kit JSON, missing required fields
- **Handling**: Fall back to simple text message, log formatting error
- **User Impact**: Alert sent but with reduced formatting

#### 6. Git Hook Integration Errors

**Scanner Initialization Failure**:
- **Cause**: Missing dependencies, configuration errors
- **Handling**: Display error message in terminal, exit with code 2 (error), do not block commit
- **User Impact**: Commit proceeds with warning, scanning skipped

**Timeout Exceeded**:
- **Cause**: Too many files, slow scanning
- **Handling**: Display timeout warning, return partial results, do not block commit
- **User Impact**: Commit proceeds, partial scan results shown

**Git Command Failures**:
- **Cause**: Not in git repository, corrupted git state
- **Handling**: Log error, fall back to scanning all files in directory
- **User Impact**: More files scanned than necessary, slower but functional

### Error Response Codes

```python
class ErrorCode(Enum):
    """Standardized error codes for the ML scanner system."""
    
    # Training Pipeline Errors (1000-1999)
    DATASET_LOAD_FAILED = 1001
    CHECKSUM_VALIDATION_FAILED = 1002
    TRAINING_FAILED = 1003
    PERFORMANCE_THRESHOLD_NOT_MET = 1004
    
    # Inference Engine Errors (2000-2999)
    MODEL_LOAD_FAILED = 2001
    INFERENCE_TIMEOUT = 2002
    GPU_OOM = 2003
    INVALID_INPUT = 2004
    
    # Model Registry Errors (3000-3999)
    REGISTRY_UNREACHABLE = 3001
    MODEL_NOT_FOUND = 3002
    STORAGE_QUOTA_EXCEEDED = 3003
    
    # Explainer Errors (4000-4999)
    EXPLANATION_TIMEOUT = 4001
    EXPLAINER_LIBRARY_ERROR = 4002
    
    # Alert Manager Errors (5000-5999)
    SLACK_WEBHOOK_FAILED = 5001
    MESSAGE_FORMATTING_ERROR = 5002
    
    # Git Hook Errors (6000-6999)
    SCANNER_INIT_FAILED = 6001
    SCAN_TIMEOUT = 6002
    GIT_COMMAND_FAILED = 6003
```

### Logging Strategy

**Log Levels**:
- **DEBUG**: Detailed diagnostic information (tokenization, model loading steps)
- **INFO**: Normal operations (scan started, model loaded, findings detected)
- **WARNING**: Recoverable errors (fallback mode, timeout, retry attempts)
- **ERROR**: Serious errors requiring attention (model load failed, training failed)
- **CRITICAL**: System-level failures (all components down, data corruption)

**Log Format**:
```json
{
  "timestamp": "2024-01-15T10:30:45.123Z",
  "level": "ERROR",
  "component": "InferenceEngine",
  "error_code": 2001,
  "message": "Failed to load model from registry",
  "context": {
    "model_version": "1.2.3",
    "registry_url": "https://registry.example.com",
    "error_details": "Connection timeout after 30s"
  },
  "trace_id": "abc123def456"
}
```

## Testing Strategy

### Dual Testing Approach

The ML-enhanced secret scanner requires both unit testing and property-based testing for comprehensive coverage:

- **Unit Tests**: Verify specific examples, edge cases, error conditions, and integration points
- **Property Tests**: Verify universal properties across all inputs using randomized testing

Both approaches are complementary and necessary. Unit tests catch concrete bugs and validate specific scenarios, while property tests verify general correctness across a wide input space.

### Property-Based Testing

**Framework**: We will use `hypothesis` for Python, which provides powerful property-based testing capabilities.

**Configuration**:
- Minimum 100 iterations per property test (due to randomization)
- Each property test must reference its design document property
- Tag format: `# Feature: ml-codebert-secret-detection, Property {number}: {property_text}`

**Example Property Test**:

```python
from hypothesis import given, strategies as st
import hypothesis

@given(st.text(min_size=1, max_size=10000))
@hypothesis.settings(max_examples=100)
def test_property_3_tokenization_length_constraint(code_sample):
    """
    Feature: ml-codebert-secret-detection
    Property 3: Tokenization Length Constraint
    
    For any code sample, tokenizing it with CodeBERT tokenizer should 
    produce output with length ≤ 512 tokens.
    """
    tokenizer = CodeTokenizer()
    tokens = tokenizer.tokenize(code_sample, max_length=512)
    
    assert len(tokens['input_ids']) <= 512, \
        f"Tokenized length {len(tokens['input_ids'])} exceeds maximum 512"
```

**Property Test Coverage**:
- All 75 correctness properties should have corresponding property tests
- Properties marked as "example" in prework should have unit tests instead
- Edge cases should be covered by both unit tests and property test generators

### Unit Testing

**Framework**: `pytest` for Python

**Test Organization**:
```
tests/
├── unit/
│   ├── test_dataset_loader.py
│   ├── test_model_trainer.py
│   ├── test_inference_engine.py
│   ├── test_ml_scanner.py
│   ├── test_hybrid_detector.py
│   ├── test_explainer.py
│   ├── test_model_registry.py
│   ├── test_alert_manager.py
│   └── test_git_hook.py
├── integration/
│   ├── test_training_pipeline.py
│   ├── test_inference_pipeline.py
│   └── test_end_to_end.py
├── property/
│   ├── test_property_dataset.py
│   ├── test_property_inference.py
│   ├── test_property_hybrid.py
│   ├── test_property_explainer.py
│   └── test_property_registry.py
└── fixtures/
    ├── sample_datasets.py
    ├── mock_models.py
    └── test_data.py
```

**Unit Test Examples**:

```python
# test_dataset_loader.py
def test_load_github_dataset_success():
    """Test successful loading of GitHub dataset."""
    loader = DatasetLoader()
    dataset = loader.load_github_datasets([
        "https://github.com/dxa4481/truffleHogRegexes"
    ])
    
    assert dataset is not None
    assert len(dataset.texts) > 0
    assert len(dataset.labels) == len(dataset.texts)
    assert dataset.source == "github"

def test_load_dataset_invalid_url():
    """Test error handling for invalid dataset URL."""
    loader = DatasetLoader()
    
    with pytest.raises(ValueError) as exc_info:
        loader.load_github_datasets(["https://invalid-url.com/nonexistent"])
    
    assert "Failed to load dataset" in str(exc_info.value)

# test_inference_engine.py
def test_inference_timeout_handling():
    """Test that inference respects timeout and falls back gracefully."""
    engine = InferenceEngine(timeout=1)
    
    # Create a mock that takes longer than timeout
    with patch('model.forward', side_effect=lambda x: time.sleep(2)):
        result = engine.predict("some code")
    
    # Should return empty result, not crash
    assert result == []

# test_hybrid_detector.py
def test_merge_overlapping_detections():
    """Test that overlapping detections are merged correctly."""
    regex_findings = [
        Finding(file_path="test.py", line_number=10, matched_text="AKIA123",
                secret_type="AWS Key", confidence="HIGH", entropy=4.5,
                is_comment=False, source="REGEX_ONLY")
    ]
    
    ml_findings = [
        Finding(file_path="test.py", line_number=10, matched_text="AKIA123",
                secret_type="AWS Key", confidence="HIGH", entropy=4.5,
                is_comment=False, source="ML_ONLY", confidence_score=0.95)
    ]
    
    merger = ResultMerger()
    merged = merger.merge(regex_findings, ml_findings)
    
    assert len(merged) == 1  # Deduplicated
    assert merged[0].source == "HIGH_CONFIDENCE"  # Both detected
    assert merged[0].confidence_score == 0.95  # ML score preserved
```

### Integration Testing

**Scope**: Test interactions between multiple components

**Key Integration Tests**:

1. **Training Pipeline Integration**:
   - Load datasets → Preprocess → Train → Validate → Save to registry
   - Verify end-to-end training workflow

2. **Inference Pipeline Integration**:
   - Load model → Tokenize → Predict → Explain → Format results
   - Verify end-to-end inference workflow

3. **Hybrid Detection Integration**:
   - Regex scan + ML scan → Merge → Alert → Log
   - Verify complete detection workflow

4. **Feedback Loop Integration**:
   - Mark false positive → Store feedback → Incorporate in retraining
   - Verify feedback integration

**Example Integration Test**:

```python
def test_end_to_end_detection_workflow():
    """Test complete detection workflow from file scan to Slack alert."""
    # Setup
    config = load_test_config()
    hybrid_detector = HybridDetector(config)
    alert_manager = AlertManager(config)
    
    # Create test file with known secret
    test_file = create_temp_file_with_secret("AKIAIOSFODNN7EXAMPLE")
    
    # Scan file
    result = hybrid_detector.scan_file(test_file)
    
    # Verify detection
    assert len(result.findings) == 1
    assert result.findings[0].secret_type == "AWS Access Key"
    assert result.findings[0].source in ["HIGH_CONFIDENCE", "REGEX_ONLY", "ML_ONLY"]
    
    # Send alert
    alert_sent = alert_manager.send_alert(result.findings, context={
        "repo": "test-repo",
        "branch": "main",
        "author": "test@example.com"
    })
    
    assert alert_sent is True
    
    # Verify alert content
    assert_slack_message_contains(result.findings[0])
```

### Performance Testing

**Benchmarks**:

1. **Inference Latency**:
   - Target: < 5 seconds per file (up to 10,000 lines)
   - Measure: P50, P95, P99 latencies

2. **Throughput**:
   - Target: ≥ 100 files/minute (CPU), ≥ 500 files/minute (GPU)
   - Measure: Files processed per minute under load

3. **Memory Usage**:
   - Target: < 500MB for inference engine
   - Measure: Peak memory consumption during inference

4. **Training Time**:
   - Target: < 2 hours for full training run
   - Measure: End-to-end training duration

**Performance Test Example**:

```python
import time
import psutil

def test_inference_latency_benchmark():
    """Benchmark inference latency for various file sizes."""
    engine = InferenceEngine()
    
    test_cases = [
        ("small", 100),    # 100 lines
        ("medium", 1000),  # 1000 lines
        ("large", 10000),  # 10000 lines
    ]
    
    for name, line_count in test_cases:
        code = generate_test_code(line_count)
        
        start = time.time()
        result = engine.predict(code)
        latency = time.time() - start
        
        print(f"{name} ({line_count} lines): {latency:.2f}s")
        
        if line_count <= 10000:
            assert latency < 5.0, f"Latency {latency}s exceeds 5s threshold"

def test_memory_usage_benchmark():
    """Benchmark memory usage during inference."""
    process = psutil.Process()
    
    # Measure baseline memory
    baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    # Load model and run inference
    engine = InferenceEngine()
    for _ in range(100):
        engine.predict(generate_test_code(1000))
    
    # Measure peak memory
    peak_memory = process.memory_info().rss / 1024 / 1024  # MB
    memory_increase = peak_memory - baseline_memory
    
    print(f"Memory increase: {memory_increase:.2f} MB")
    assert memory_increase < 500, f"Memory usage {memory_increase}MB exceeds 500MB"
```

### Test Data Management

**Synthetic Test Data**:
- Generate synthetic code samples with known secrets for testing
- Use `hypothesis` strategies to generate diverse inputs
- Create fixtures for common test scenarios

**Real-World Test Data**:
- Curate a test dataset from public repositories (with secrets redacted)
- Include examples of false positives and edge cases
- Maintain separate test/validation splits

**Test Data Generators**:

```python
from hypothesis import strategies as st

# Strategy for generating code with secrets
@st.composite
def code_with_secret(draw):
    """Generate code sample containing a secret."""
    secret_type = draw(st.sampled_from([
        "AWS_KEY", "GITHUB_TOKEN", "API_KEY", "PASSWORD"
    ]))
    
    if secret_type == "AWS_KEY":
        secret = "AKIA" + draw(st.text(alphabet=st.characters(
            whitelist_categories=('Lu', 'Nd')), min_size=16, max_size=16))
    elif secret_type == "GITHUB_TOKEN":
        secret = "ghp_" + draw(st.text(alphabet=st.characters(
            whitelist_categories=('Lu', 'Ll', 'Nd')), min_size=36, max_size=36))
    # ... other secret types
    
    code_template = draw(st.sampled_from([
        f'api_key = "{secret}"',
        f'const token = "{secret}";',
        f'AWS_ACCESS_KEY_ID={secret}',
    ]))
    
    return code_template, secret_type, secret
```

### Continuous Integration

**CI Pipeline**:

```yaml
# .github/workflows/ml-scanner-tests.yml
name: ML Scanner Tests

on: [push, pull_request]

jobs:
  unit-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.9'
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install -r requirements-dev.txt
      - name: Run unit tests
        run: pytest tests/unit/ -v --cov=ml_scanner
      
  property-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.9'
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install -r requirements-dev.txt
      - name: Run property tests
        run: pytest tests/property/ -v --hypothesis-show-statistics
      
  integration-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.9'
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install -r requirements-dev.txt
      - name: Run integration tests
        run: pytest tests/integration/ -v
      
  performance-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.9'
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install -r requirements-dev.txt
      - name: Run performance benchmarks
        run: pytest tests/performance/ -v --benchmark-only
```

### Test Coverage Goals

- **Unit Test Coverage**: ≥ 85% line coverage
- **Property Test Coverage**: All 75 correctness properties
- **Integration Test Coverage**: All major component interactions
- **Performance Test Coverage**: All performance-critical paths

### Testing Best Practices

1. **Isolation**: Each test should be independent and not rely on external state
2. **Determinism**: Tests should produce consistent results across runs
3. **Speed**: Unit tests should run in < 1 second each
4. **Clarity**: Test names should clearly describe what is being tested
5. **Maintainability**: Use fixtures and helpers to reduce duplication
6. **Documentation**: Each test should have a docstring explaining its purpose

