import gzip
import json
from pathlib import Path
from typing import Generator, Dict, Any, Union
from src.schema import Candidate
from src.utils import get_logger

logger = get_logger("data_loader")

def stream_candidates(
    file_path: Union[str, Path], 
    validate: bool = False
) -> Generator[Dict[str, Any], None, None]:
    path = Path(file_path)
    if not path.exists():
        # Try gzip fallback if candidate.jsonl is passed but only candidate.jsonl.gz exists, or vice versa
        if path.suffix == ".gz":
            unzipped_path = path.with_suffix("")
            if unzipped_path.exists():
                path = unzipped_path
                logger.info(f"File {file_path} not found, falling back to {path}")
        else:
            zipped_path = path.with_name(path.name + ".gz")
            if zipped_path.exists():
                path = zipped_path
                logger.info(f"File {file_path} not found, falling back to {path}")
            
    if not path.exists():
        raise FileNotFoundError(f"Candidate file not found: {file_path}")

    logger.info(f"Loading candidates from {path} (validation={validate})...")
    
    open_func = gzip.open if path.suffix == ".gz" else open
    mode = "rt" if path.suffix == ".gz" else "r"
    
    with open_func(path, mode, encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                candidate_dict = json.loads(line)
                if validate:
                    try:
                        # Validate using Pydantic model
                        validated_cand = Candidate(**candidate_dict)
                        yield validated_cand.model_dump()
                    except Exception as ve:
                        logger.warning(f"Validation failed on line {line_num} (candidate_id: {candidate_dict.get('candidate_id')}): {ve}")
                        # Yield the unvalidated dict but log the warning
                        yield candidate_dict
                else:
                    yield candidate_dict
            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error on line {line_num}: {e}")
                continue
