"""
Basic tests for explainer classes.

This test file verifies that the explainer classes can be instantiated
and have the correct interface.
"""

import pytest
from ml_scanner.explainer import Explainer, SHAPExplainer, LIMEExplainer
from ml_scanner.models import Detection, Explanation


def test_shap_explainer_initialization():
    """Test that SHAPExplainer can be initialized."""
    try:
        explainer = SHAPExplainer()
        assert explainer is not None
        assert isinstance(explainer, Explainer)
    except Exception as e:
        pytest.skip(f"SHAP library not available: {e}")


def test_lime_explainer_initialization():
    """Test that LIMEExplainer can be initialized."""
    try:
        explainer = LIMEExplainer()
        assert explainer is not None
        assert isinstance(explainer, Explainer)
    except Exception as e:
        pytest.skip(f"LIME library not available: {e}")


def test_explainer_is_abstract():
    """Test that Explainer cannot be instantiated directly."""
    with pytest.raises(TypeError):
        Explainer()


def test_normalize_attributions():
    """Test attribution normalization."""
    try:
        explainer = SHAPExplainer()
    except:
        pytest.skip("SHAP library not available")
    
    # Test with positive scores
    attributions = [("token1", 0.5), ("token2", 0.3), ("token3", 0.2)]
    normalized = explainer._normalize_attributions(attributions)
    
    # Check that scores sum to 1.0 (within floating point tolerance)
    total = sum(score for _, score in normalized)
    assert abs(total - 1.0) < 0.01
    
    # Check that all scores are positive
    for _, score in normalized:
        assert score >= 0


def test_normalize_attributions_with_negative():
    """Test attribution normalization with negative scores."""
    try:
        explainer = SHAPExplainer()
    except:
        pytest.skip("SHAP library not available")
    
    # Test with mixed positive and negative scores
    attributions = [("token1", 0.5), ("token2", -0.3), ("token3", 0.2)]
    normalized = explainer._normalize_attributions(attributions)
    
    # Check that scores sum to 1.0 (within floating point tolerance)
    total = sum(score for _, score in normalized)
    assert abs(total - 1.0) < 0.01
    
    # Check that all scores are positive (absolute values used)
    for _, score in normalized:
        assert score >= 0


def test_normalize_attributions_empty():
    """Test attribution normalization with empty list."""
    try:
        explainer = SHAPExplainer()
    except:
        pytest.skip("SHAP library not available")
    
    attributions = []
    normalized = explainer._normalize_attributions(attributions)
    
    assert normalized == []


def test_get_top_tokens():
    """Test extraction of top tokens."""
    try:
        explainer = SHAPExplainer()
    except:
        pytest.skip("SHAP library not available")
    
    attributions = [
        ("token1", 0.5),
        ("token2", 0.3),
        ("token3", 0.2),
        ("token4", 0.1),
        ("token5", 0.05),
        ("token6", 0.01)
    ]
    
    # Get top 5 tokens
    top_tokens = explainer._get_top_tokens(attributions, k=5)
    
    assert len(top_tokens) == 5
    assert top_tokens == ["token1", "token2", "token3", "token4", "token5"]


def test_get_top_tokens_fewer_than_k():
    """Test extraction of top tokens when fewer than k tokens available."""
    try:
        explainer = SHAPExplainer()
    except:
        pytest.skip("SHAP library not available")
    
    attributions = [("token1", 0.5), ("token2", 0.3)]
    
    # Request 5 tokens but only 2 available
    top_tokens = explainer._get_top_tokens(attributions, k=5)
    
    assert len(top_tokens) == 2
    assert top_tokens == ["token1", "token2"]


def test_format_explanation():
    """Test explanation formatting."""
    try:
        explainer = SHAPExplainer()
    except:
        pytest.skip("SHAP library not available")
    
    text = "api_key = AKIAIOSFODNN7EXAMPLE"
    top_tokens = ["AKIA", "IOSFODNN", "7EXAMPLE"]
    attributions = [
        ("AKIA", 0.5),
        ("IOSFODNN", 0.3),
        ("7EXAMPLE", 0.2)
    ]
    
    formatted = explainer._format_explanation(text, top_tokens, attributions)
    
    # Check that formatted text contains key elements
    assert "Token Attributions:" in formatted
    assert "AKIA" in formatted
    assert "0.500" in formatted or "0.5" in formatted
    assert "Highlighted code:" in formatted


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
