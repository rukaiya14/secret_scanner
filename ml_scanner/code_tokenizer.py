"""
Code Tokenizer for ML-Enhanced Secret Scanner.

This module provides the CodeTokenizer class that wraps the CodeBERT tokenizer
with encode/decode methods and handles max sequence length truncation.
"""

from typing import List
from transformers import AutoTokenizer, BatchEncoding

from ml_scanner.logger import get_logger

logger = get_logger(__name__)


class CodeTokenizer:
    """
    Wraps CodeBERT tokenizer with encode/decode methods.
    
    The CodeTokenizer handles:
    - Tokenization of code snippets using CodeBERT tokenizer
    - Max sequence length truncation (512 tokens)
    - Decoding token IDs back to text
    
    Requirements: 3.1
    """
    
    def __init__(self, model_name: str = "microsoft/codebert-base"):
        """
        Initialize CodeBERT tokenizer.
        
        Args:
            model_name: Name of the CodeBERT model to use for tokenization
                       (default: "microsoft/codebert-base")
        """
        self.model_name = model_name
        logger.info(f"Initializing CodeTokenizer with model: {model_name}")
        
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(model_name)
            logger.info("CodeTokenizer initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize tokenizer: {str(e)}", exc_info=True)
            raise
    
    def encode(self, code: str, max_length: int = 512) -> BatchEncoding:
        """
        Tokenize code for model input with max sequence length truncation.
        
        This method tokenizes the input code and ensures the output does not
        exceed the specified max_length (default 512 tokens).
        
        Args:
            code: Code snippet to tokenize
            max_length: Maximum sequence length in tokens (default 512)
            
        Returns:
            BatchEncoding containing tokenized input with:
            - input_ids: Token IDs
            - attention_mask: Attention mask
            - Other model-specific tensors
        """
        if not isinstance(code, str):
            raise ValueError(f"Code must be a string, got {type(code).__name__}")
        
        # Tokenize with truncation to max_length
        tokens = self._tokenizer(
            code,
            truncation=True,
            max_length=max_length,
            padding=False,
            return_tensors=None,  # Return lists, not tensors
        )
        
        logger.debug(
            f"Tokenized code: {len(tokens['input_ids'])} tokens "
            f"(max_length={max_length})"
        )
        
        return tokens
    
    def tokenize(self, code: str, max_length: int = 512) -> BatchEncoding:
        """
        Tokenize code for model input (alias for encode).
        
        This method is an alias for encode() to match the design document interface.
        
        Args:
            code: Code snippet to tokenize
            max_length: Maximum sequence length in tokens (default 512)
            
        Returns:
            BatchEncoding containing tokenized input
        """
        return self.encode(code, max_length)
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decode token IDs back to text.
        
        Args:
            token_ids: List of token IDs to decode
            skip_special_tokens: Whether to skip special tokens like [CLS], [SEP], [PAD]
                                (default True)
            
        Returns:
            Decoded text string
        """
        if not isinstance(token_ids, list):
            raise ValueError(f"token_ids must be a list, got {type(token_ids).__name__}")
        
        decoded_text = self._tokenizer.decode(
            token_ids,
            skip_special_tokens=skip_special_tokens
        )
        
        logger.debug(f"Decoded {len(token_ids)} tokens to text")
        
        return decoded_text
    
    @property
    def vocab_size(self) -> int:
        """Get the vocabulary size of the tokenizer."""
        return len(self._tokenizer)
    
    @property
    def max_model_length(self) -> int:
        """Get the maximum model input length."""
        return self._tokenizer.model_max_length
