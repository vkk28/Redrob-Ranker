from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Union
import numpy as np
from src.utils import get_logger

logger = get_logger("embeddings")

_model = None

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

def get_embedding_model() -> SentenceTransformer:
    global _model
    if _model is None:
        logger.info(f"Loading sentence-transformers {EMBEDDING_MODEL_NAME} (CPU)...")
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME, device="cpu")
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
    
    # Education list
    education = candidate.get("education", [])
    edu_list = []
    for edu in education:
        deg = edu.get("degree", "")
        major = edu.get("field_of_study", "")
        inst = edu.get("institution", "")
        if deg or major:
            edu_list.append(f"{deg} in {major} from {inst}")
    edu_text = " | ".join(edu_list)
    
    # Certifications
    certs = candidate.get("certifications", []) or []
    certs_text = ", ".join(c.get("name", "") for c in certs if c.get("name"))
    
    # Languages
    langs = candidate.get("languages", []) or []
    langs_text = ", ".join(l.get("language", "") for l in langs if l.get("language"))
    
    parts = []
    if headline:
        parts.append(f"Headline: {headline}")
    if summary:
        parts.append(f"Summary: {summary}")
    if history_text:
        parts.append(f"Experience: {history_text}")
    if skills_text:
        parts.append(f"Skills: {skills_text}")
    if edu_text:
        parts.append(f"Education: {edu_text}")
    if certs_text:
        parts.append(f"Certifications: {certs_text}")
    if langs_text:
        parts.append(f"Languages: {langs_text}")
        
    return "\n".join(parts)

def embed_text(text: Union[str, List[str]], is_query: bool = False) -> np.ndarray:
    """
    Embed string or list of strings using the sentence-transformers model.
    The is_query flag is accepted for API compatibility but MiniLM does not
    require a special query prefix (unlike BGE models).
    """
    model = get_embedding_model()
    
    if isinstance(text, str):
        texts = [text]
    else:
        texts = text
        
    embeddings = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
    
    if isinstance(text, str):
        return embeddings[0]
    return embeddings

