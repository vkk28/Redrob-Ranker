from typing import List, Dict, Any
from src.utils import get_logger

logger = get_logger("tier5_finder")

AI_BUZZWORDS = [
    "rag", "llm", "large language model", "generative ai", "pinecone", "weaviate", 
    "qdrant", "milvus", "gpt", "claude", "bert", "transformer", "prompt engineering", 
    "vector search", "vector database"
]

CLASSIC_REC_SEARCH_KEYWORDS = [
    "recommendation system", "recommender", "collaborative filtering", "search engine", 
    "information retrieval", "ranking algorithm", "elasticsearch", "solr", "lucene", 
    "inverted index", "matching algorithm", "learning to rank", "ranking system", 
    "indexing system", "search indexing", "indexing pipeline", "matrix factorization"
]

def is_tier5_candidate(candidate: Dict[str, Any]) -> bool:
    """
    Check if a candidate meets Tier-5 criteria:
    - No trendy AI buzzwords in headline, summary, or skills list.
    - Career history contains evidence of building classic search, ranking, or recommendation systems.
    """
    profile = candidate.get("profile", {})
    headline = profile.get("headline", "").lower()
    summary = profile.get("summary", "").lower()
    skills = [s.get("name", "").lower() for s in candidate.get("skills", [])]
    
    # 1. Reject if candidate has trendy AI buzzwords
    has_buzzwords = False
    for word in AI_BUZZWORDS:
        if word in headline or word in summary or any(word in s for s in skills):
            has_buzzwords = True
            break
            
    if has_buzzwords:
        return False
        
    # 2. Check for classic search/rec system building evidence in career history
    history = candidate.get("career_history", [])
    history_text = " ".join([r.get("description", "") for r in history]).lower()
    
    has_classic_evidence = False
    for word in CLASSIC_REC_SEARCH_KEYWORDS:
        if word in history_text:
            has_classic_evidence = True
            break
            
    return has_classic_evidence

def find_tier5_candidates(
    candidates_pool: List[Dict[str, Any]], 
    count: int = 5
) -> List[Dict[str, Any]]:
    """
    Scans the candidate pool and returns the top `count` Tier-5 candidates.
    Sorts by years of experience and base score.
    """
    logger.info("Scanning for Tier-5 candidates...")
    tier5_cands = []
    
    for cand in candidates_pool:
        # Candidate must have passed gates
        if cand.get("_gate_multiplier", 1.0) == 0.0:
            continue
            
        if is_tier5_candidate(cand):
            tier5_cands.append(cand)
            
    # Sort tier-5 candidates by experience and overall quality
    tier5_cands.sort(
        key=lambda c: (
            c.get("profile", {}).get("years_of_experience", 0), 
            c.get("base_score", 0.0)
        ), 
        reverse=True
    )
    
    logger.info(f"Found {len(tier5_cands)} Tier-5 candidates. Force-including top {min(count, len(tier5_cands))}.")
    return tier5_cands[:count]
