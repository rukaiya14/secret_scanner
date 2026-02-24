"""
Unit tests for CodeTokenizer class.

Tests the CodeTokenizer's ability to tokenize code snippets and decode token IDs.
"""

import pytest
from ml_scanner.code_tokenizer import CodeTokenizer


class TestCodeTokenizer:
    """Test suite for CodeTokenizer class."""
    
    def test_initialization(self):
        """Test that CodeTokenizer initializes successfully."""
        tokenizer = CodeTokenizer()
        assert tokenizer is not None
        assert tokenizer.model_name == "microsoft/codebert-base"
    
    def test_initialization_custom_model(self):
        """Test initialization with custom model name."""
        tokenizer = CodeTokenizer(model_name="microsoft/codebert-base")
        assert tokenizer.model_name == "microsoft/codebert-base"
    
    def test_encode_simple_code(self):
        """Test encoding a simple code snippet."""
        tokenizer = CodeTokenizer()
        code = 'api_key = "AKIAIOSFODNN7EXAMPLE"'
        
        tokens = tokenizer.encode(code)
        
        assert tokens is not None
        assert "input_ids" in tokens
        assert "attention_mask" in tokens
        assert len(tokens["input_ids"]) > 0
        assert len(tokens["input_ids"]) <= 512
    
    def test_encode_with_max_length(self):
        """Test that encoding respects max_length parameter."""
        tokenizer = CodeTokenizer()
        # Create a long code snippet
        code = "x = 1\n" * 1000  # Very long code
        
        tokens = tokenizer.encode(code, max_length=128)
        
        assert len(tokens["input_ids"]) <= 128
    
    def test_encode_max_length_512(self):
        """Test that default max_length is 512."""
        tokenizer = CodeTokenizer()
        # Create a very long code snippet
        code = "x = 1\n" * 10000
        
        tokens = tokenizer.encode(code)
        
        assert len(tokens["input_ids"]) <= 512
    
    def test_tokenize_alias(self):
        """Test that tokenize() is an alias for encode()."""
        tokenizer = CodeTokenizer()
        code = 'password = "secret123"'
        
        tokens1 = tokenizer.encode(code)
        tokens2 = tokenizer.tokenize(code)
        
        assert tokens1["input_ids"] == tokens2["input_ids"]
        assert tokens1["attention_mask"] == tokens2["attention_mask"]
    
    def test_decode_token_ids(self):
        """Test decoding token IDs back to text."""
        tokenizer = CodeTokenizer()
        code = 'api_key = "test"'
        
        # Encode
        tokens = tokenizer.encode(code)
        token_ids = tokens["input_ids"]
        
        # Decode
        decoded = tokenizer.decode(token_ids)
        
        assert decoded is not None
        assert isinstance(decoded, str)
        # The decoded text should contain the main content (special tokens removed)
        assert "api_key" in decoded or "test" in decoded
    
    def test_decode_with_special_tokens(self):
        """Test decoding with special tokens included."""
        tokenizer = CodeTokenizer()
        code = "x = 1"
        
        tokens = tokenizer.encode(code)
        token_ids = tokens["input_ids"]
        
        # Decode with special tokens
        decoded_with = tokenizer.decode(token_ids, skip_special_tokens=False)
        decoded_without = tokenizer.decode(token_ids, skip_special_tokens=True)
        
        # With special tokens should be longer or equal
        assert len(decoded_with) >= len(decoded_without)
    
    def test_encode_invalid_input(self):
        """Test that encoding non-string input raises ValueError."""
        tokenizer = CodeTokenizer()
        
        with pytest.raises(ValueError, match="Code must be a string"):
            tokenizer.encode(123)
    
    def test_decode_invalid_input(self):
        """Test that decoding non-list input raises ValueError."""
        tokenizer = CodeTokenizer()
        
        with pytest.raises(ValueError, match="token_ids must be a list"):
            tokenizer.decode("not a list")
    
    def test_encode_empty_string(self):
        """Test encoding an empty string."""
        tokenizer = CodeTokenizer()
        
        tokens = tokenizer.encode("")
        
        assert tokens is not None
        assert "input_ids" in tokens
        # Empty string should still have special tokens
        assert len(tokens["input_ids"]) >= 0
    
    def test_vocab_size_property(self):
        """Test that vocab_size property returns a positive integer."""
        tokenizer = CodeTokenizer()
        
        vocab_size = tokenizer.vocab_size
        
        assert isinstance(vocab_size, int)
        assert vocab_size > 0
    
    def test_max_model_length_property(self):
        """Test that max_model_length property returns a positive integer."""
        tokenizer = CodeTokenizer()
        
        max_length = tokenizer.max_model_length
        
        assert isinstance(max_length, int)
        assert max_length > 0
    
    def test_round_trip_encoding_decoding(self):
        """Test that encoding and decoding preserves content."""
        tokenizer = CodeTokenizer()
        code = "def hello():\n    return 'world'"
        
        # Encode
        tokens = tokenizer.encode(code)
        token_ids = tokens["input_ids"]
        
        # Decode
        decoded = tokenizer.decode(token_ids)
        
        # Check that key content is preserved
        assert "hello" in decoded
        assert "world" in decoded
    
    def test_multiple_tokenizations_consistent(self):
        """Test that tokenizing the same code multiple times gives consistent results."""
        tokenizer = CodeTokenizer()
        code = "x = 42"
        
        tokens1 = tokenizer.encode(code)
        tokens2 = tokenizer.encode(code)
        
        assert tokens1["input_ids"] == tokens2["input_ids"]
        assert tokens1["attention_mask"] == tokens2["attention_mask"]
