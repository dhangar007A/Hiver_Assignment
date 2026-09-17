import pytest
from src.preprocess import clean_text

def test_clean_text():
    text1 = "My order 123456789012 is lost."
    assert clean_text(text1) == "My order [ORDER_NUMBER] is lost."
    
    text2 = "Email me at test@example.com or call 555-123-4567"
    assert clean_text(text2) == "Email me at [EMAIL] or call [PHONE]"
    
    assert clean_text(None) == ""
