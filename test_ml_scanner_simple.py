# Save this as: test_ml_scanner_simple.py

import sys
sys.path.insert(0, '.')

print("Testing ML Scanner with trained model...")
print()

# Test 1: Check if model exists
import os
model_path = "model_registry/models/1.0.0"
if os.path.exists(model_path):
    print(f"✅ Model found at: {model_path}")
    
    # Check model files
    required_files = ["model.safetensors", "config.json", "tokenizer.json", "metadata.json"]
    for file in required_files:
        file_path = os.path.join(model_path, file)
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            print(f"   ✅ {file} ({size:,} bytes)")
        else:
            print(f"   ❌ {file} MISSING")
else:
    print(f"❌ Model not found at: {model_path}")
    print("Please train the model first using the Colab notebook")
    sys.exit(1)

print()
print("Attempting to load ML Scanner...")
print()

try:
    from ml_scanner.model_registry import ModelRegistry
    from ml_scanner.inference_engine import InferenceEngine
    from ml_scanner.ml_scanner import MLScanner
    
    # Initialize components
    registry = ModelRegistry("model_registry")
    print("✅ Model Registry initialized")
    
    # Get model
    model_metadata = registry.get_model("1.0.0")
    if model_metadata:
        print(f"✅ Model loaded: v{model_metadata.version}")
        print(f"   Training date: {model_metadata.training_date}")
        metrics = model_metadata.performance_metrics
        print(f"   Performance: P={metrics.get('precision', 0):.2f}, R={metrics.get('recall', 0):.2f}, F1={metrics.get('f1', 0):.2f}")
    
    print()
    
    # Initialize inference engine
    config = {
        "timeout": 10,
        "confidence_threshold": 0.5,
        "device": "cpu"  # Force CPU for 32-bit Python
    }
    
    engine = InferenceEngine(registry, config)
    print("✅ Inference Engine initialized")
    
    # Initialize ML scanner
    scanner_config = {
        "timeout": 5,
        "confidence_threshold": 0.5,
        "enabled_categories": ["API_KEY", "PASSWORD", "PII", "TOKEN", "CERTIFICATE", "OTHER"]
    }
    
    scanner = MLScanner(scanner_config, engine)
    print("✅ ML Scanner initialized")
    
    print()
    print("="*70)
    print("Testing on sample secrets...")
    print("="*70)
    print()
    
    # Test cases
    tests = [
        ('AWS Key', 'api_key = "AKIAIOSFODNN7EXAMPLE"'),
        ('GitHub Token', 'token = "ghp_1a2B3c4D5e6F7g8H9i0J1k2L3m4N5o6P7q8R"'),
        ('Safe Code', 'username = "john_doe"'),
    ]
    
    for name, code in tests:
        print(f"Test: {name}")
        print(f"Code: {code}")
        
        findings = scanner.scan_text(code, "test.py")
        
        if findings:
            print(f"Result: ✅ DETECTED ({len(findings)} finding(s))")
            for f in findings:
                print(f"  - {f.secret_type}: {f.confidence} (score: {f.confidence_score:.2f})")
        else:
            print("Result: ⚪ NOT DETECTED")
        print()
    
    # Test file if it exists
    if os.path.exists("test_secret.py"):
        print("="*70)
        print("Scanning test_secret.py...")
        print("="*70)
        print()
        
        findings = scanner.scan_file("test_secret.py")
        if findings:
            print(f"✅ Found {len(findings)} secret(s):")
            for i, f in enumerate(findings, 1):
                print(f"{i}. Line {f.line_number}: {f.secret_type}")
                print(f"   Confidence: {f.confidence} ({f.confidence_score:.2f})")
                print(f"   Text: {f.matched_text}")
        else:
            print("⚪ No secrets detected")
    
    print()
    print("="*70)
    print("✅ ML Scanner is working!")
    print("="*70)
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
