import json
import pytest
from src.schema import Candidate
from src.config import SAMPLE_CANDIDATES_JSON

def test_schema_compliance():
    """
    Load sample_candidates.json and ensure every single profile complies with
    the Pydantic Candidate schema.
    """
    assert SAMPLE_CANDIDATES_JSON.exists(), f"Sample candidates file not found at {SAMPLE_CANDIDATES_JSON}"
    
    with open(SAMPLE_CANDIDATES_JSON, "r", encoding="utf-8") as f:
        candidates = json.load(f)
        
    assert len(candidates) > 0, "No candidates loaded from sample file"
    
    for i, c_dict in enumerate(candidates):
        try:
            cand = Candidate(**c_dict)
            assert cand.candidate_id == c_dict["candidate_id"]
        except Exception as e:
            pytest.fail(f"Candidate index {i} (ID: {c_dict.get('candidate_id')}) failed schema compliance: {e}")
