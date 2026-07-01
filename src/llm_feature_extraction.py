import os
import json
import re
from typing import Dict, Any, List, Tuple
from pathlib import Path
from src.utils import get_logger

logger = get_logger("llm_feature_extraction")

def local_extract_features(candidate: Dict[str, Any]) -> Dict[str, Any]:
    """
    High-fidelity deterministic feature extraction when LLM API keys are missing.
    Scans the candidate's career history and profile for production, indexing, and eval experience.
    """
    profile = candidate.get("profile", {})
    history = candidate.get("career_history", [])
    skills = {s.get("name", "").lower() for s in candidate.get("skills", [])}
    
    # 1. Look for production evidence of search, ranking, or recommendation systems
    production_keywords = [
        "production", "deploy", "scale", "million", "user", "latency", 
        "optimize", "pipeline", "shipped", "real-time", "real time"
    ]
    search_keywords = [
        "rag", "retrieval", "vector", "search", "indexing", "ranking", 
        "recommendation", "faiss", "milvus", "qdrant", "pinecone", 
        "elasticsearch", "opensearch", "hybrid search"
    ]
    
    history_text = " ".join([r.get("description", "") for r in history]).lower()
    summary_text = profile.get("summary", "").lower()
    headline_text = profile.get("headline", "").lower()
    full_text = f"{headline_text} {summary_text} {history_text}"
    
    # Calculate production search evidence
    prod_matches = sum(1 for kw in production_keywords if kw in full_text)
    search_matches = sum(1 for kw in search_keywords if kw in full_text)
    
    prod_score = 0.0
    if search_matches > 0:
        prod_score = min(1.0, 0.3 + (search_matches * 0.15) + (prod_matches * 0.1))
    
    # 2. Look for evaluation framework depth
    eval_keywords = ["ndcg", "mrr", "map", "precision@", "recall@", "a/b test", "ab test", "eval"]
    eval_matches = sum(1 for kw in eval_keywords if kw in full_text)
    
    eval_score = 0.0
    if eval_matches > 0:
        eval_score = min(1.0, 0.4 + (eval_matches * 0.2))
    elif any("eval" in s for s in skills):
        eval_score = 0.3
        
    # 3. YOE fit
    yoe = profile.get("years_of_experience", 0)
    if 5 <= yoe <= 9:
        yoe_score = 1.0
    elif 4 <= yoe <= 10:
        yoe_score = 0.8
    else:
        yoe_score = max(0.2, 1.0 - abs(yoe - 7.0) * 0.1)

    # 4. Generate fit reasoning narrative
    narrative = generate_local_narrative(candidate, prod_score, eval_score)

    return {
        "candidate_id": candidate["candidate_id"],
        "production_experience_score": round(prod_score, 2),
        "eval_depth_score": round(eval_score, 2),
        "yoe_fit_score": round(yoe_score, 2),
        "fit_narrative": narrative
    }

def generate_local_narrative(candidate: Dict[str, Any], prod_score: float, eval_score: float) -> str:
    """
    Generate a candidate-specific reasoning narrative based on their actual profile.
    Avoids all hardcoded template suffixes — each narrative is unique to the candidate.
    """
    profile = candidate.get("profile", {})
    yoe = profile.get("years_of_experience", 0)
    title = profile.get("current_title", "Engineer")
    company = profile.get("current_company", "")
    
    # Extract key achievements from history
    history = candidate.get("career_history", [])
    achievements = []
    
    for role in history:
        desc = role.get("description", "")
        sentences = re.split(r'[.!?]', desc)
        for s in sentences:
            s_clean = s.strip()
            if not s_clean or len(s_clean) < 20:
                continue
            # Only pick achievement-oriented sentences
            if any(kw in s_clean.lower() for kw in [
                "built", "designed", "led", "shipped", "owned", "scaled",
                "implemented", "architected", "developed", "launched",
                "created", "optimized", "reduced", "increased", "improved"
            ]):
                achievements.append(s_clean)
                if len(achievements) >= 2:
                    break
        if len(achievements) >= 2:
            break
    
    # Find which specific JD-relevant technologies candidate knows
    skills = [s.get("name", "").lower() for s in candidate.get("skills", [])]
    history_text = " ".join([r.get("description", "") for r in history]).lower()
    summary_text = profile.get("summary", "").lower()
    
    vector_dbs_in_history = [t for t in ["faiss", "milvus", "qdrant", "pinecone", "weaviate", "pgvector", "opensearch", "elasticsearch"]
                            if t in history_text or t in summary_text]
    vector_dbs_in_skills = [t for t in ["faiss", "milvus", "qdrant", "pinecone", "weaviate", "pgvector", "opensearch", "elasticsearch"]
                           if t in " ".join(skills) and t not in vector_dbs_in_history]
                           
    eval_tools_in_history = [t for t in ["ndcg", "mrr", "map", "a/b test", "evaluation"]
                            if t in history_text or t in summary_text]
    eval_tools_in_skills = [t for t in ["ndcg", "mrr", "map", "a/b test", "evaluation"]
                           if t in " ".join(skills) and t not in eval_tools_in_history]
    
    # Build narrative from actual profile data
    if achievements:
        main_achievement = achievements[0]
        main_achievement = re.sub(r'\s+', ' ', main_achievement).strip()
        if len(main_achievement) > 150:
            main_achievement = main_achievement[:147] + "..."
        if not main_achievement.endswith('.'):
            main_achievement += '.'
        narrative = f"{yoe} years experience as {title} at {company}. Shipped key systems: {main_achievement}"
    else:
        # Fallback using actual skills
        top_skills = [s.get("name") for s in candidate.get("skills", [])[:3]]
        skills_str = ", ".join(top_skills) if top_skills else "software engineering"
        narrative = f"{title} at {company} with {yoe} YOE. Handled systems involving {skills_str}."
    
    # Add specific JD-technology connections (not generic praise)
    if vector_dbs_in_history:
        narrative += f" Shipped systems using {', '.join(vector_dbs_in_history[:2]).upper()}."
    elif vector_dbs_in_skills:
        narrative += f" Lists skills: {', '.join(vector_dbs_in_skills[:2]).upper()}."
        
    if eval_tools_in_history:
        narrative += f" Experience with {', '.join(eval_tools_in_history[:2])} evaluation."
    elif eval_tools_in_skills:
        narrative += f" Lists evaluation skills: {', '.join(eval_tools_in_skills[:2])}."
    
    return narrative

def run_extraction(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Runs feature extraction for a list of candidates.
    Currently defaults to local extraction for reproducibility.
    """
    results = []
    for cand in candidates:
        features = local_extract_features(cand)
        results.append(features)
    return results
