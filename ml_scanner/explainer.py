"""
Explainer components for ML-Enhanced Secret Scanner.

This module provides interpretability for ML detections using SHAP and LIME.
Explainers generate token attributions showing which parts of the code
contributed most to the detection decision.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Optional
import logging
import time
import numpy as np

from ml_scanner.models import Detection, Explanation
from ml_scanner.exceptions import ErrorCode, ExplainerLibraryError

logger = logging.getLogger(__name__)


class Explainer(ABC):
    """
    Abstract base class for explanation generation.
    
    Defines the interface that all explainer implementations must follow.
    Supports both SHAP and LIME backends for model interpretability.
    """
    
    @abstractmethod
    def explain(self, model, tokenizer, text: str, prediction: Detection, 
                timeout: float = 2.0) -> Optional[Explanation]:
        """
        Generate explanation for a detection.
        
        Args:
            model: The trained model that made the prediction
            tokenizer: Tokenizer used to process the text
            text: Original text that was analyzed
            prediction: Detection object containing the prediction details
            timeout: Maximum time in seconds for explanation generation (default 2.0)
            
        Returns:
            Explanation object with token attributions, or None if generation fails
            
        Raises:
            ExplainerLibraryError: If explanation generation encounters an error
        """
        pass
    
    def _normalize_attributions(self, attributions: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
        """
        Normalize attribution scores to sum to 1.0.
        
        Args:
            attributions: List of (token, score) tuples
            
        Returns:
            Normalized list of (token, score) tuples where scores sum to 1.0
        """
        if not attributions:
            return []
        
        # Extract scores and compute sum
        scores = np.array([score for _, score in attributions])
        total = np.abs(scores).sum()
        
        if total == 0:
            # All scores are zero, distribute equally
            normalized_score = 1.0 / len(attributions)
            return [(token, normalized_score) for token, _ in attributions]
        
        # Normalize using absolute values to ensure positive scores
        normalized = [(token, abs(score) / total) for token, score in attributions]
        
        return normalized
    
    def _get_top_tokens(self, attributions: List[Tuple[str, float]], k: int = 5) -> List[str]:
        """
        Extract top K most influential tokens.
        
        Args:
            attributions: List of (token, score) tuples
            k: Number of top tokens to return (default 5)
            
        Returns:
            List of top K token strings
        """
        if not attributions:
            return []
        
        # Sort by score descending and take top k
        sorted_attrs = sorted(attributions, key=lambda x: x[1], reverse=True)
        return [token for token, _ in sorted_attrs[:k]]
    
    def _format_explanation(self, text: str, top_tokens: List[str], 
                           attributions: List[Tuple[str, float]]) -> str:
        """
        Format explanation as human-readable text with highlighted tokens.
        
        Args:
            text: Original text
            top_tokens: List of most influential tokens
            attributions: Full list of token attributions
            
        Returns:
            Formatted explanation string
        """
        # Create attribution map for quick lookup
        attr_map = {token: score for token, score in attributions}
        
        # Build explanation header
        lines = ["Token Attributions:"]
        
        # Add top tokens with scores
        for i, token in enumerate(top_tokens, 1):
            score = attr_map.get(token, 0.0)
            bar_length = int(score * 50)  # Scale to 50 chars max
            bar = "█" * bar_length
            lines.append(f"  {i}. '{token}' ({score:.3f}) {bar}")
        
        # Add highlighted text
        lines.append("\nHighlighted code:")
        highlighted = text
        for token in top_tokens:
            # Simple highlighting with brackets
            highlighted = highlighted.replace(token, f"[{token}]")
        
        lines.append(f"  {highlighted}")
        
        return "\n".join(lines)


class SHAPExplainer(Explainer):
    """
    SHAP-based explainer for model interpretability.
    
    Uses SHapley Additive exPlanations to compute token attributions.
    SHAP provides theoretically grounded feature importance based on
    cooperative game theory.
    """
    
    def __init__(self):
        """Initialize SHAP explainer."""
        try:
            import shap
            self.shap = shap
            logger.info("SHAPExplainer initialized successfully")
        except ImportError as e:
            logger.error(f"Failed to import SHAP library: {e}")
            raise ExplainerLibraryError(
                "SHAP library not available. Install with: pip install shap"
            )
    
    def explain(self, model, tokenizer, text: str, prediction: Detection,
                timeout: float = 2.0) -> Optional[Explanation]:
        """
        Generate SHAP-based explanation for a detection.
        
        Args:
            model: The trained model that made the prediction
            tokenizer: Tokenizer used to process the text
            text: Original text that was analyzed
            prediction: Detection object containing the prediction details
            timeout: Maximum time in seconds for explanation generation (default 2.0)
            
        Returns:
            Explanation object with token attributions, or None if generation fails
        """
        start_time = time.time()
        
        try:
            # Tokenize the text
            tokens = tokenizer.tokenize(text, max_length=512)
            input_ids = tokens['input_ids']
            
            # Decode tokens back to strings for attribution
            token_strings = [tokenizer.tokenizer.decode([tid]) for tid in input_ids]
            
            # Create a prediction function for SHAP
            def predict_fn(token_ids_batch):
                """Wrapper function for model prediction."""
                import torch
                outputs = []
                for token_ids in token_ids_batch:
                    # Convert to tensor
                    input_tensor = torch.tensor([token_ids]).to(model.device)
                    with torch.no_grad():
                        output = model(input_tensor)
                        # Get probability for positive class (secret detected)
                        prob = torch.softmax(output.logits, dim=-1)[0, 1].item()
                    outputs.append(prob)
                return np.array(outputs)
            
            # Check timeout
            if time.time() - start_time > timeout:
                logger.warning(f"SHAP explanation generation timed out after {timeout}s")
                return None
            
            # Create SHAP explainer (using a simple masking approach)
            # For efficiency, we'll use a simplified attribution method
            # that masks tokens and measures impact on prediction
            attributions = []
            
            # Get baseline prediction
            baseline_prob = predict_fn([input_ids])[0]
            
            # Compute attribution for each token by masking it
            for i, token_str in enumerate(token_strings):
                # Check timeout periodically
                if time.time() - start_time > timeout:
                    logger.warning(f"SHAP explanation generation timed out after {timeout}s")
                    return None
                
                # Create masked version (replace token with mask token)
                masked_ids = input_ids.copy()
                masked_ids[i] = tokenizer.tokenizer.mask_token_id
                
                # Get prediction with masked token
                masked_prob = predict_fn([masked_ids])[0]
                
                # Attribution is the difference in prediction
                attribution = baseline_prob - masked_prob
                attributions.append((token_str, float(attribution)))
            
            # Normalize attributions to sum to 1.0
            normalized_attributions = self._normalize_attributions(attributions)
            
            # Get top 5 influential tokens
            top_tokens = self._get_top_tokens(normalized_attributions, k=5)
            
            # Format explanation
            formatted_text = self._format_explanation(text, top_tokens, normalized_attributions)
            
            # Create Explanation object
            explanation = Explanation(
                detection=prediction,
                token_attributions=normalized_attributions,
                top_tokens=top_tokens,
                formatted_text=formatted_text
            )
            
            elapsed = time.time() - start_time
            logger.info(f"SHAP explanation generated in {elapsed:.2f}s")
            
            return explanation
            
        except Exception as e:
            logger.error(f"SHAP explanation generation failed: {e}", exc_info=True)
            # Return None instead of raising to allow detection without explanation
            return None


class LIMEExplainer(Explainer):
    """
    LIME-based explainer for model interpretability.
    
    Uses Local Interpretable Model-agnostic Explanations to compute
    token attributions. LIME explains predictions by approximating the
    model locally with an interpretable model.
    """
    
    def __init__(self):
        """Initialize LIME explainer."""
        try:
            import lime
            import lime.lime_text
            self.lime = lime
            logger.info("LIMEExplainer initialized successfully")
        except ImportError as e:
            logger.error(f"Failed to import LIME library: {e}")
            raise ExplainerLibraryError(
                "LIME library not available. Install with: pip install lime"
            )
    
    def explain(self, model, tokenizer, text: str, prediction: Detection,
                timeout: float = 2.0) -> Optional[Explanation]:
        """
        Generate LIME-based explanation for a detection.
        
        Args:
            model: The trained model that made the prediction
            tokenizer: Tokenizer used to process the text
            text: Original text that was analyzed
            prediction: Detection object containing the prediction details
            timeout: Maximum time in seconds for explanation generation (default 2.0)
            
        Returns:
            Explanation object with token attributions, or None if generation fails
        """
        start_time = time.time()
        
        try:
            # Create prediction function for LIME
            def predict_fn(texts):
                """Wrapper function for model prediction."""
                import torch
                probs = []
                for txt in texts:
                    # Tokenize
                    tokens = tokenizer.tokenize(txt, max_length=512)
                    input_ids = torch.tensor([tokens['input_ids']]).to(model.device)
                    
                    # Predict
                    with torch.no_grad():
                        output = model(input_ids)
                        # Get probabilities for both classes
                        prob = torch.softmax(output.logits, dim=-1)[0].cpu().numpy()
                    probs.append(prob)
                
                return np.array(probs)
            
            # Check timeout
            if time.time() - start_time > timeout:
                logger.warning(f"LIME explanation generation timed out after {timeout}s")
                return None
            
            # Create LIME explainer
            explainer = self.lime.lime_text.LimeTextExplainer(
                class_names=['non_secret', 'secret'],
                split_expression=r'\W+',  # Split on non-word characters
                random_state=42
            )
            
            # Generate explanation
            # Use fewer samples for speed
            exp = explainer.explain_instance(
                text,
                predict_fn,
                num_features=10,
                num_samples=100
            )
            
            # Check timeout
            if time.time() - start_time > timeout:
                logger.warning(f"LIME explanation generation timed out after {timeout}s")
                return None
            
            # Extract token attributions for the positive class (secret detected)
            lime_attributions = exp.as_list(label=1)  # label=1 is 'secret' class
            
            # Convert to our format
            attributions = [(token, float(score)) for token, score in lime_attributions]
            
            # Normalize attributions to sum to 1.0
            normalized_attributions = self._normalize_attributions(attributions)
            
            # Get top 5 influential tokens
            top_tokens = self._get_top_tokens(normalized_attributions, k=5)
            
            # Format explanation
            formatted_text = self._format_explanation(text, top_tokens, normalized_attributions)
            
            # Create Explanation object
            explanation = Explanation(
                detection=prediction,
                token_attributions=normalized_attributions,
                top_tokens=top_tokens,
                formatted_text=formatted_text
            )
            
            elapsed = time.time() - start_time
            logger.info(f"LIME explanation generated in {elapsed:.2f}s")
            
            return explanation
            
        except Exception as e:
            logger.error(f"LIME explanation generation failed: {e}", exc_info=True)
            # Return None instead of raising to allow detection without explanation
            return None
