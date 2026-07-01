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
    
    # Check if the file is a standard JSON file (not JSONL)
    if path.suffix == ".json":
        logger.info(f"Loading candidates as a standard JSON array from {path}...")
        try:
            with open(path, "r", encoding="utf-8") as f:
                candidates_list = json.load(f)
            for line_num, candidate_dict in enumerate(candidates_list, 1):
                if "candidate_id" in candidate_dict:
                    candidate_dict["candidate_id"] = str(candidate_dict["candidate_id"])
                if validate:
                    try:
                        validated_cand = Candidate(**candidate_dict)
                        yield validated_cand.model_dump()
                    except Exception as ve:
                        logger.warning(f"Validation failed on item {line_num} (candidate_id: {candidate_dict.get('candidate_id')}): {ve}")
                        yield candidate_dict
                else:
                    yield candidate_dict
            return
        except Exception as e:
            logger.error(f"Failed to read as standard JSON array: {e}. Falling back to streaming line-by-line.")

    open_func = gzip.open if path.suffix == ".gz" else open
    mode = "rt" if path.suffix == ".gz" else "r"
    
    with open_func(path, mode, encoding="utf-8") as f:
        # Check if the first character is '[', indicating a JSON array, which we should handle
        first_char = ""
        try:
            # peek first non-empty char
            for line in f:
                stripped = line.strip()
                if stripped:
                    first_char = stripped[0]
                    break
            # reset file pointer
            f.seek(0)
        except Exception:
            pass

        if first_char == "[":
            logger.warning(f"File {path} starts with '[' indicating a JSON array. Reading whole file.")
            try:
                content = f.read()
                candidates_list = json.loads(content)
                for line_num, candidate_dict in enumerate(candidates_list, 1):
                    if "candidate_id" in candidate_dict:
                        candidate_dict["candidate_id"] = str(candidate_dict["candidate_id"])
                    if validate:
                        try:
                            validated_cand = Candidate(**candidate_dict)
                            yield validated_cand.model_dump()
                        except Exception as ve:
                            logger.warning(f"Validation failed on item {line_num}: {ve}")
                            yield candidate_dict
                    else:
                        yield candidate_dict
                return
            except Exception as e:
                logger.error(f"Failed to read JSON array stream: {e}")
                f.seek(0)

        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                candidate_dict = json.loads(line)
                if "candidate_id" in candidate_dict:
                    candidate_dict["candidate_id"] = str(candidate_dict["candidate_id"])
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
