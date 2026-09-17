import pytest
import numpy as np
from src.generate_reply import ReplyGenerator

def test_reply_generator_mock():
    generator = ReplyGenerator(provider="mock")
    # Even without a real index (if not built yet), it should handle gracefully
    res = generator.generate("Hello", "general_inquiry")
    assert "reply" in res
    assert "grounding_used" in res
    assert "retrieval_similarity" in res
