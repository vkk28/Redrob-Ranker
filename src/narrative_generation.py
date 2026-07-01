from typing import Dict, Any, List
import re
from src.utils import get_logger

logger = get_logger("narrative_generation")

# JD-specific requirement buckets for reasoning connections
JD_REQUIREMENT_BUCKETS = {
    "vector_db": {
        "keywords": ["faiss", "milvus", "qdrant", "pinecone", "weaviate", "pgvector", "opensearch", "elasticsearch"],
        "description": "vector database experience (FAISS/Milvus/Qdrant)",
    },
    "embeddings": {
        "keywords": ["embedding", "sentence-transformer", "bge", "dense retrieval", "vector representation", "sentence transformers"],
        "description": "embeddings and dense retrieval",
    },
    "search_ranking": {
        "keywords": ["search", "ranking", "retrieval", "recommendation", "information retrieval", "learning to rank", "reranking"],
        "description": "search/ranking/retrieval systems",
    },
    "eval_metrics": {
        "keywords": ["ndcg", "mrr", "map", "precision@", "recall@", "a/b test", "evaluation", "offline-to-online"],
        "description": "evaluation frameworks and ranking metrics",
    },
    "production_ml": {
        "keywords": ["production", "deploy", "scale", "pipeline", "shipped", "real-time", "latency", "mlops"],
        "description": "production ML system deployment",
    },
    "python_stack": {
        "keywords": ["python", "pytorch", "tensorflow", "scikit-learn", "numpy", "pandas", "pyspark"],
        "description": "Python ML stack",
    },
}


def _get_matched_jd_buckets(candidate: Dict[str, Any]) -> List[str]:
    """Find which JD requirement buckets a candidate's profile matches."""
    skills = [s.get("name", "").lower() for s in candidate.get("skills", [])]
    history_text = " ".join(r.get("description", "") for r in candidate.get("career_history", [])).lower()
    profile = candidate.get("profile", {})
    summary = profile.get("summary", "").lower()
    headline = profile.get("headline", "").lower()
    full_text = f"{headline} {summary} {history_text} {' '.join(skills)}"

    matched = []
    for bucket_name, bucket in JD_REQUIREMENT_BUCKETS.items():
        if any(kw in full_text for kw in bucket["keywords"]):
            matched.append(bucket["description"])
    return matched


def _get_top_skills_str(candidate: Dict[str, Any], n: int = 3) -> str:
    """Get top N skills by endorsements/duration."""
    skills = candidate.get("skills", [])
    ranked = sorted(
        skills,
        key=lambda s: (s.get("endorsements", 0), s.get("duration_months", 0)),
        reverse=True,
    )[:n]
    return ", ".join(s["name"] for s in ranked if s.get("name"))


def _get_signal_details(candidate: Dict[str, Any]) -> Dict[str, Any]:
    """Extract useful signal values for reasoning."""
    signals = candidate.get("redrob_signals", {})
    return {
        "notice_days": signals.get("notice_period_days", 90),
        "response_rate": signals.get("recruiter_response_rate", 0.5),
        "open_to_work": signals.get("open_to_work_flag", False),
        "last_active": signals.get("last_active_date", ""),
        "work_mode": signals.get("preferred_work_mode", ""),
        "willing_relocate": signals.get("willing_to_relocate", False),
        "salary_max": signals.get("expected_salary_range_inr_lpa", {}).get("max", 0),
        "interview_rate": signals.get("interview_completion_rate", 1.0),
    }


def _describe_notice(notice_days: int) -> str:
    if notice_days <= 15:
        return "immediately available"
    elif notice_days <= 30:
        return f"{notice_days}-day notice (quick joiner)"
    elif notice_days <= 60:
        return f"{notice_days}-day notice period"
    elif notice_days <= 90:
        return f"{notice_days}-day notice (standard)"
    else:
        return f"long {notice_days}-day notice period"


def _extract_achievement(candidate: Dict[str, Any]) -> str:
    """Extract a unique achievement sentence from this candidate's career history."""
    history = candidate.get("career_history", [])
    for role in history:
        desc = role.get("description", "")
        if not desc:
            continue
        sentences = re.split(r'(?<=[.!?])\s+', desc)
        for s in sentences:
            s_clean = s.strip()
            if len(s_clean) > 30 and any(
                kw in s_clean.lower()
                for kw in ["built", "designed", "led", "shipped", "owned", "scaled",
                           "implemented", "architected", "developed", "launched",
                           "created", "optimized", "reduced", "increased", "improved"]
            ):
                # Truncate cleanly if too long
                if len(s_clean) > 150:
                    truncated = s_clean[:147]
                    last_space = truncated.rfind(' ')
                    if last_space > 0:
                        s_clean = truncated[:last_space] + "..."
                    else:
                        s_clean = truncated + "..."
                if not s_clean.endswith("."):
                    s_clean += "."
                return s_clean
    return ""


def offline_verify_narrative(
    candidate: Dict[str, Any], 
    narrative: str, 
    skill_ontology: Any = None, 
    max_attempts: int = 3
) -> str:
    """
    Verifies that the claims made in the narrative correspond to actual values in the candidate's profile.
    If the claims cannot be verified (hallucinations found), it returns the clean deterministic fallback reasoning.
    """
    profile = candidate.get("profile", {})
    history = candidate.get("career_history", [])
    skills = candidate.get("skills", [])
    
    actual_skills = {s["name"].lower() for s in skills}
    history_text = " ".join(r.get("description", "") for r in history).lower()
    
    all_companies = {profile.get("current_company", "").lower()} | {
        r.get("company", "").lower() for r in history if r.get("company")
    }
    all_companies = {c for c in all_companies if c}
    
    actual_yoe = profile.get("years_of_experience", 0)

    # Perform static checks on the narrative
    narrative_lower = narrative.lower()

    # Check for Hallucinated Skills
    tech_keywords = [
        "rag", "pinecone", "weaviate", "milvus", "qdrant", "faiss", "elasticsearch", "opensearch",
        "langchain", "llamaindex", "bert", "gpt", "claude", "llama", "tensorflow", "pytorch",
        "keras", "scikit", "numpy", "pandas", "spark", "hadoop", "kafka", "aws", "gcp", "azure",
        "docker", "kubernetes", "sql", "postgres", "mongodb", "redis", "fastapi", "flask", "django"
    ]
    
    for tech in tech_keywords:
        if re.search(r'\b' + re.escape(tech) + r'\b', narrative_lower):
            in_skills = any(tech in s for s in actual_skills)
            in_history = tech in history_text
            if not (in_skills or in_history):
                logger.warning(f"Hallucination warning: Tech '{tech}' mentioned in narrative but not found in profile or career history of candidate {candidate['candidate_id']}")
                return build_fallback_reasoning(candidate, rank=50)

    # Check YOE discrepancy
    yoe_matches = re.findall(r'(\d+(?:\.\d+)?)\s*(?:years|yoe|yrs|year)', narrative_lower)
    for match in yoe_matches:
        val = float(match)
        if abs(val - actual_yoe) > 1.5:
            logger.warning(f"Hallucination warning: Narrative claims {val} years of experience, but profile has {actual_yoe} years for candidate {candidate['candidate_id']}")
            return build_fallback_reasoning(candidate, rank=50)

    return narrative


def build_fallback_reasoning(candidate: Dict[str, Any], rank: int = 50) -> str:
    """
    Build a high-quality, non-generic fallback reasoning using actual candidate facts.
    """
    profile = candidate.get("profile", {})
    yoe = profile.get("years_of_experience", 0)
    title = profile.get("current_title", "Software Engineer")
    company = profile.get("current_company", "")
    location = profile.get("location", "")
    
    top_skills = _get_top_skills_str(candidate, 3)
    jd_matches = _get_matched_jd_buckets(candidate)
    signals = _get_signal_details(candidate)

    parts = [f"{title} at {company} with {yoe:.1f} years experience"]
    
    if location:
        parts[0] += f" ({location})"
    parts[0] += "."
    
    if top_skills:
        parts.append(f"Strongest verified skills: {top_skills}.")
    
    if jd_matches:
        parts.append(f"Aligns with JD on: {', '.join(jd_matches[:2])}.")
    
    if rank > 70:
        if signals["notice_days"] > 60:
            parts.append(f"Caveat: {_describe_notice(signals['notice_days'])}.")
    
    return " ".join(parts)


def assemble_reasoning(candidate: Dict[str, Any], rank: int, pre_verified_narrative: str) -> str:
    """
    Assemble the final reasoning string with rank-aware context and candidate-specific details.
    Each reasoning is unique: combines the narrative with specific signal values, JD connections,
    and honest concerns tailored to the candidate's rank.
    """
    profile = candidate.get("profile", {})
    yoe = profile.get("years_of_experience", 0)
    title = profile.get("current_title", "Engineer")
    company = profile.get("current_company", "")
    location = profile.get("location", "")
    
    signals = _get_signal_details(candidate)
    jd_matches = _get_matched_jd_buckets(candidate)
    top_skills = _get_top_skills_str(candidate, 3)
    achievement = _extract_achievement(candidate)
    
    core_fit = candidate.get("_core_fit_score", 0.0)
    logistics_score = candidate.get("_logistics_score", 1.0)
    
    parts = []
    
    # --- Opening: varies by rank tier for tone calibration ---
    if rank <= 10:
        # Top tier: lead with achievement or strong headline
        if achievement:
            parts.append(f"{yoe:.0f}y {title} at {company}. Key achievement: {achievement}")
        else:
            parts.append(f"{title} at {company}, {yoe:.1f} years. Top-ranked for strong alignment with JD requirements.")
        
        # Add specific JD match details
        if jd_matches:
            match_str = ", ".join(jd_matches[:3])
            templates = [
                f"Directly maps to: {match_str}.",
                f"Strongly aligns with core JD requirements: {match_str}.",
                f"Demonstrates validated expertise in: {match_str}.",
                f"Brings deep experience in: {match_str}."
            ]
            parts.append(templates[rank % len(templates)])
        if top_skills:
            parts.append(f"Core skills: {top_skills}.")
        
        # Add positive signals
        if signals["open_to_work"]:
            parts.append("Actively open to opportunities.")
        if signals["notice_days"] <= 30:
            parts.append(f"{_describe_notice(signals['notice_days'])}.")
        if signals["response_rate"] >= 0.7:
            parts.append(f"High recruiter engagement ({signals['response_rate']:.0%} response rate).")
            
    elif rank <= 30:
        # Strong candidates: balanced detail
        if achievement:
            parts.append(f"{title} at {company} ({yoe:.1f} YOE). Notable: {achievement}")
        else:
            parts.append(f"{title} at {company} with {yoe:.1f} years experience.")
        
        if jd_matches:
            match_str = ", ".join(jd_matches[:2])
            templates = [
                f"JD alignment: {match_str}.",
                f"Shows strong fit on: {match_str}.",
                f"Demonstrates capability in: {match_str}.",
                f"Matches key areas: {match_str}."
            ]
            parts.append(templates[rank % len(templates)])
        if top_skills:
            parts.append(f"Verified skills: {top_skills}.")
        
        # Mix of positives and neutrals
        notice_str = _describe_notice(signals["notice_days"])
        if signals["notice_days"] <= 60:
            parts.append(f"Logistics: {notice_str}, {location or 'location undisclosed'}.")
        else:
            parts.append(f"Note: {notice_str}.")
            
    elif rank <= 60:
        # Mid-range: neutral tone with trade-offs
        parts.append(f"{title} at {company}, {yoe:.1f} years.")
        
        if top_skills:
            parts.append(f"Skills: {top_skills}.")
        
        # Highlight what matches and what's missing
        if jd_matches:
            match_str = ", ".join(jd_matches[:2])
            templates = [
                f"Partial JD fit on: {match_str}.",
                f"Demonstrated competence in: {match_str}.",
                f"Brings partial overlap in: {match_str}.",
                f"Partially aligned with: {match_str}."
            ]
            parts.append(templates[rank % len(templates)])
        else:
            parts.append("Limited direct overlap with core JD requirements (embeddings, vector DBs, eval metrics).")
        
        # Add logistics context
        if signals["notice_days"] > 60:
            parts.append(f"Logistics concern: {_describe_notice(signals['notice_days'])}.")
        if not signals["willing_relocate"] and location:
            parts.append(f"Based in {location}, not open to relocation.")
            
    elif rank <= 80:
        # Lower-mid: acknowledge gaps clearly
        parts.append(f"{title} at {company} ({yoe:.1f} YOE).")
        
        if top_skills:
            parts.append(f"Skills: {top_skills}.")
        
        # Explicit gap analysis
        if core_fit < 0.4:
            parts.append("Gap: limited production search/retrieval evidence in career history.")
        if not jd_matches:
            parts.append("No direct match with key JD areas (vector DBs, embeddings, ranking metrics).")
        elif len(jd_matches) == 1:
            parts.append(f"Tangential JD overlap limited to {jd_matches[0]}.")
        
        if logistics_score < 0.5:
            parts.append(f"Logistics friction: {_describe_notice(signals['notice_days'])}; work mode: {signals['work_mode'] or 'unspecified'}.")
            
    else:
        # Bottom tier (81–100): explicit concerns, honest about why they're here
        parts.append(f"{title} at {company}, {yoe:.1f} years.")
        
        # Must include concerns to match low rank
        concerns = []
        if core_fit < 0.5:
            concerns.append("weak production ML systems evidence")
        if not jd_matches:
            concerns.append("no direct JD requirement overlap")
        if signals["notice_days"] > 90:
            concerns.append(f"long notice ({signals['notice_days']}d)")
        if logistics_score < 0.5:
            concerns.append("logistics mismatch")
        if signals["response_rate"] < 0.3:
            concerns.append("low recruiter engagement")
        if not signals["open_to_work"]:
            concerns.append("not actively job-seeking")
            
        if concerns:
            parts.append(f"Ranked lower due to: {'; '.join(concerns[:3])}.")
        else:
            parts.append("Marginal fit — included to complete top-100 but below ideal threshold.")
        
        if top_skills:
            parts.append(f"Has: {top_skills}.")
    
    # Build final reasoning
    reasoning = " ".join(parts)
    
    # Ensure 1-3 sentences max (truncate if too long)
    sentences = re.split(r'(?<=[.!?])\s+', reasoning)
    if len(sentences) > 4:
        reasoning = " ".join(sentences[:3])
        if not reasoning.endswith("."):
            reasoning += "."
    
    return reasoning
