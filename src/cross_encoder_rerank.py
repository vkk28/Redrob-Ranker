from sentence_transformers import CrossEncoder
from typing import List, Dict, Any, Tuple
import os
from pathlib import Path
from src.utils import get_logger
import src.config as config

logger = get_logger("cross_encoder_rerank")

_cross_encoder_model = None

def get_cross_encoder() -> CrossEncoder:
    global _cross_encoder_model
    if _cross_encoder_model is None:
        model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
        local_model_path = config.ARTIFACTS_DIR / "cross_encoder_model"
        
        # Check if we should load from local cached artifacts directory (Phase 2 ranking sandbox, no internet)
        if local_model_path.exists():
            logger.info(f"Loading CrossEncoder from local cache {local_model_path}...")
            _cross_encoder_model = CrossEncoder(str(local_model_path), device="cpu")
        else:
            # Phase 1 setup (network available)
            logger.info(f"Loading CrossEncoder {model_name} from HuggingFace...")
            _cross_encoder_model = CrossEncoder(model_name, device="cpu")
    return _cross_encoder_model

def save_cross_encoder_locally():
    """
    Downloads and caches the cross-encoder model locally to the artifacts directory.
    This must be run in Phase 1 setup so it's available in the no-network ranking step.
    """
    local_model_path = config.ARTIFACTS_DIR / "cross_encoder_model"
    if not local_model_path.exists():
        logger.info("Caching CrossEncoder model files locally for offline rank run...")
        model = get_cross_encoder()
        # Save model
        model.save(str(local_model_path))
        logger.info(f"CrossEncoder cached at {local_model_path}")

def rerank_candidates(
    query_text: str, 
    candidates: List[Dict[str, Any]], 
    candidate_texts: List[str]
) -> List[Tuple[Dict[str, Any], float]]:
    """
    Rerank a subset of candidates using the Cross-Encoder.
    Returns list of (candidate, score) pairs.
    """
    if not candidates:
        return []
        
    try:
        model = get_cross_encoder()
        pairs = [(query_text, text) for text in candidate_texts]
        
        # Predict scores
        logger.info(f"Reranking {len(candidates)} candidates via CrossEncoder...")
        scores = model.predict(pairs, show_progress_bar=False)
        
        # Normalize scores to [0.0, 1.0] roughly using sigmoid/min-max or raw scores
        # ms-marco scores are logits, often in range [-10, 10]
        # Let's map them to 0-1
        min_s, max_s = min(scores), max(scores)
        if max_s > min_s:
            norm_scores = [(s - min_s) / (max_s - min_s) for s in scores]
        else:
            norm_scores = [0.5 for _ in scores]
            
        results = []
        for i, cand in enumerate(candidates):
            results.append((cand, float(norm_scores[i])))
        return results
    except Exception as e:
        logger.error(f"Error in CrossEncoder reranking: {e}. Falling back to embedding cosine similarity.")
        # Fallback to embedding similarity if any error occurs
        results = []
        for cand in candidates:
            # Default fallback score based on candidate final_score
            results.append((cand, float(cand.get("final_score", 0.5))))
        return results
