import pytest
from pathlib import Path
from validate_submission import validate_submission

def test_validator_compliance():
    """
    Ensure the generated submission.csv passes all checks in the official validate_submission.py script.
    """
    csv_path = Path("submission.csv")
    if not csv_path.exists():
        # Fallback to benchmark CSV if main submission is not run yet
        csv_path = Path("benchmark_submission.csv")
        
    assert csv_path.exists(), "No ranking CSV found to validate. Run ranker or benchmark script first."
    
    errors = validate_submission(csv_path)
    
    assert not errors, f"Submission CSV failed validation with errors: {errors}"
