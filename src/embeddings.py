from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Union
import numpy as np
from src.utils import get_logger

logger = get_logger("embeddings")

_model = None

def get_embedding_model() -> SentenceTransformer:
    global _model
    if _model is None:
        logger.info("Loading sentence-transformers BAAI/bge-large-en-v1.5 (CPU)...")
        # Load local or remote model
        _model = SentenceTransformer("BAAI/bge-large-en-v1.5", device="cpu")
    return _model

def get_candidate_text(candidate: Dict[str, Any]) -> str:
    """
    Concatenation recipe for candidate profiles.
    Concatenates: headline + summary + top 3 career descriptions + skills list.
    """
    profile = candidate.get("profile", {})
    headline = profile.get("headline", "")
    summary = profile.get("summary", "")
    
    # Career history descriptions (limit to top 3 roles)
    history = candidate.get("career_history", [])
    history_desc_list = []
    for role in history[:3]:
        desc = role.get("description", "")
        if desc:
            history_desc_list.append(f"{role.get('title', '')} at {role.get('company', '')}: {desc}")
    history_text = " | ".join(history_desc_list)
    
    # Skills list
    skills = candidate.get("skills", [])
    skills_text = ", ".join(s.get("name", "") for s in skills if s.get("name"))
    
    parts = []
    if headline:
        parts.append(f"Headline: {headline}")
    if summary:
        parts.append(f"Summary: {summary}")
    if history_text:
        parts.append(f"Experience: {history_text}")
    if skills_text:
        parts.append(f"Skills: {skills_text}")
        
    return "\n".join(parts)

def embed_text(text: Union[str, List[str]], is_query: bool = False) -> np.ndarray:
    """
    Embed string or list of strings using the BGE model.
    Prepend query instruction if is_query=True (required for BAAI/bge retrieval).
    """
    model = get_embedding_model()
    
    if isinstance(text, str):
        texts = [text]
    else:
        texts = text
        
    if is_query:
        # BGE query instruction prefix
        prefix = "Represent this sentence for searching relevant passages: "
        texts = [prefix + t for t in texts]
        
    embeddings = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
    
    if isinstance(text, str):
        return embeddings[0]
    return embeddings
