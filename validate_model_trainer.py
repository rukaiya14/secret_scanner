"""
Simple validation script for ModelTrainer implementation.
Checks imports and basic structure without requiring full dependencies.
"""

import ast
import sys


def validate_model_trainer():
    """Validate ModelTrainer implementation."""
    print("Validating ModelTrainer implementation...")
    
    # Read the file
    with open("ml_scanner/model_trainer.py", "r") as f:
        code = f.read()
    
    # Parse the AST
    try:
        tree = ast.parse(code)
        print("✓ Code parses successfully")
    except SyntaxError as e:
        print(f"✗ Syntax error: {e}")
        return False
    
    # Check for required class
    classes = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
    if "ModelTrainer" in classes:
        print("✓ ModelTrainer class found")
    else:
        print("✗ ModelTrainer class not found")
        return False
    
    # Check for required methods
    required_methods = ["__init__", "fine_tune", "save_model"]
    
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "ModelTrainer":
            methods = [m.name for m in node.body if isinstance(m, ast.FunctionDef)]
            print(f"  Found methods: {', '.join(methods)}")
            
            for method in required_methods:
                if method in methods:
                    print(f"  ✓ {method} method found")
                else:
                    print(f"  ✗ {method} method not found")
                    return False
    
    # Check for proper imports
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend([alias.name for alias in node.names])
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module)
    
    required_imports = ["torch", "transformers", "ml_scanner.models", "ml_scanner.exceptions"]
    print("\nChecking imports:")
    for imp in required_imports:
        if any(imp in i for i in imports if i):
            print(f"  ✓ {imp} imported")
        else:
            print(f"  ✗ {imp} not imported")
            return False
    
    # Check method signatures
    print("\nValidating method signatures:")
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "ModelTrainer":
            for method in node.body:
                if isinstance(method, ast.FunctionDef):
                    if method.name == "fine_tune":
                        args = [arg.arg for arg in method.args.args]
                        required_args = ["self", "model", "train_dataset", "val_dataset"]
                        if all(arg in args for arg in required_args):
                            print(f"  ✓ fine_tune has required parameters")
                        else:
                            print(f"  ✗ fine_tune missing required parameters")
                            print(f"    Expected: {required_args}")
                            print(f"    Found: {args}")
                    
                    elif method.name == "save_model":
                        args = [arg.arg for arg in method.args.args]
                        required_args = ["self", "model", "version", "metadata"]
                        if all(arg in args for arg in required_args):
                            print(f"  ✓ save_model has required parameters")
                        else:
                            print(f"  ✗ save_model missing required parameters")
    
    print("\n✓ All validations passed!")
    return True


if __name__ == "__main__":
    success = validate_model_trainer()
    sys.exit(0 if success else 1)
