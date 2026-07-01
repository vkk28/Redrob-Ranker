from typing import Dict, Any
from datetime import datetime
from src.utils import get_logger, parse_date
import src.config as config

logger = get_logger("scoring")

def calculate_education_score(candidate: Dict[str, Any]) -> float:
    """
    Compute education score based on school tiers.
    """
    education = candidate.get("education", [])
    if not education:
        return 0.3 # Unknown/tier_4 default
        
    best_score = 0.3
    for edu in education:
        tier = edu.get("tier", "unknown").lower()
        if tier == "tier_1":
            score = 1.0
        elif tier == "tier_2":
            score = 0.8
        elif tier == "tier_3":
            score = 0.5
        else:
            score = 0.3
        if score > best_score:
            best_score = score
            
    return best_score

def calculate_logistics_score(candidate: Dict[str, Any]) -> float:
    """
    Compute logistics score based on location, notice period, work mode and salary expectations.
    """
    profile = candidate.get("profile", {})
    signals = candidate.get("redrob_signals", {})
    
    # 1. Location Fit
    loc = profile.get("location", "").lower()
    willing_relocate = signals.get("willing_to_relocate", False)
    
    # Pune, Noida, Delhi NCR, Gurgaon, Faridabad, Ghaziabad, etc.
    pune_noida_ncr = ["pune", "noida", "delhi", "ncr", "gurgaon", "gurugram", "faridabad", "ghaziabad"]
    welcome_cities = ["hyderabad", "mumbai", "bangalore", "bengaluru", "chennai"]
    
    is_pune_noida_ncr = any(c in loc for c in pune_noida_ncr)
    is_welcome_city = any(c in loc for c in welcome_cities)
    
    if is_pune_noida_ncr:
        location_score = 1.0
    elif is_welcome_city:
        # Welcome cities open to relocation
        location_score = 0.9 if willing_relocate else 0.5
    elif willing_relocate:
        location_score = 0.8
    else:
        location_score = 0.2

    # 2. Notice Period Fit
    notice = signals.get("notice_period_days", 90)
    if notice <= 30:
        notice_score = 1.0
    elif notice <= 60:
        notice_score = 0.8
    elif notice <= 90:
        notice_score = 0.5
    else:
        notice_score = 0.1

    # 3. Work Mode Fit
    mode = signals.get("preferred_work_mode", "flexible").lower()
    if mode in ["hybrid", "flexible", "onsite"]:
        mode_score = 1.0
    elif mode == "remote":
        mode_score = 0.5
    else:
        mode_score = 0.8

    # 4. Salary Fit (LPA)
    salary_range = signals.get("expected_salary_range_inr_lpa", {})
    salary_max = salary_range.get("max", 0)
    if salary_max <= 30:
        salary_score = 1.0
    elif salary_max <= 45:
        salary_score = 0.8
    elif salary_max <= 60:
        salary_score = 0.6
    else:
        salary_score = 0.3

    # Combine components
    return (0.4 * location_score + 0.3 * notice_score + 0.15 * mode_score + 0.15 * salary_score)

def calculate_behavioral_multiplier(candidate: Dict[str, Any]) -> float:
    """
    Compute a behavioral multiplier based on responsiveness, activity date, and open-to-work flag.
    The JD explicitly says: "A perfect-on-paper candidate who hasn't logged in for 6 months 
    and has a 5% recruiter response rate is, for hiring purposes, not actually available."
    Bounded between 0.40 and 1.0 (sharper penalty than before).
    """
    signals = candidate.get("redrob_signals", {})
    
    # 1. Recruiter Response Rate — JD says this is critical
    response_rate = signals.get("recruiter_response_rate", 0.5)
    
    # 2. Last Active Date Decay — JD says 6 months inactive = not available
    last_active_str = signals.get("last_active_date")
    ref_date = datetime.strptime(config.REFERENCE_DATE_STR, "%Y-%m-%d")
    
    active_date = parse_date(last_active_str)
    if active_date:
        days_since_active = abs((ref_date - active_date).days)
        if days_since_active <= 30:
            activity_factor = 1.0
        elif days_since_active <= 90:
            activity_factor = 0.85
        elif days_since_active <= 180:
            activity_factor = 0.5
        else:
            activity_factor = 0.2  # "not actually available" per JD
    else:
        activity_factor = 0.3
        
    # 3. Open to Work
    open_to_work = signals.get("open_to_work_flag", False)
    otw_factor = 1.0 if open_to_work else 0.75
    
    # 4. Interview Completion
    interview_rate = signals.get("interview_completion_rate", 1.0)
    interview_factor = 1.0 if interview_rate >= 0.8 else max(0.4, interview_rate)

    raw_behavioral = response_rate * activity_factor * otw_factor * interview_factor
    
    # Sharper behavioral multiplier: 0.40 + 0.60 * raw_behavioral
    # This gives a wider range [0.40, 1.0] so behavioral signals matter more
    return 0.40 + (0.60 * raw_behavioral)

def calculate_skill_match_score(candidate: Dict[str, Any]) -> float:
    """
    Compute a skills matching score based on critical JD skills.
    
    Key insight from the JD: "The right answer is not 'find candidates whose skills section 
    contains the most AI keywords.' That's a trap we've explicitly built into the dataset."
    
    So we weight skills found in career history descriptions much higher than skills
    merely listed in the skills section.
    """
    skills = candidate.get("skills", [])
    history = candidate.get("career_history", [])
    summary = candidate.get("profile", {}).get("summary", "").lower()
    
    # Build history text for evidence checking
    history_text = " ".join([r.get("description", "") for r in history]).lower()
    full_evidence_text = f"{summary} {history_text}"
    
    # Core JD skills grouped by importance
    critical_skills = {
        "faiss", "milvus", "qdrant", "pinecone", "weaviate", "elasticsearch", "opensearch",
        "rag", "retrieval", "vector search", "hybrid search", "embeddings", "sentence-transformers",
    }
    important_skills = {
        "nlp", "evaluation", "ndcg", "map", "mrr", "ranking", "recommendation",
        "learning to rank", "information retrieval",
    }
    supporting_skills = {
        "python", "pytorch", "tensorflow", "scikit-learn", "xgboost",
    }
    
    score = 0.0
    matches = 0
    
    for s in skills:
        name = s.get("name", "").lower()
        prof = s.get("proficiency", "beginner").lower()
        dur = s.get("duration_months", 0) or 0
        
        # Determine which tier this skill belongs to
        skill_tier_weight = 0.0
        if any(ts in name for ts in critical_skills):
            skill_tier_weight = 1.0
        elif any(ts in name for ts in important_skills):
            skill_tier_weight = 0.7
        elif any(ts in name for ts in supporting_skills):
            skill_tier_weight = 0.4
        else:
            continue  # Not a JD-relevant skill
        
        # Proficiency weighting
        if prof == "expert":
            prof_mult = 1.0
        elif prof == "advanced":
            prof_mult = 0.8
        elif prof == "intermediate":
            prof_mult = 0.6
        else:
            prof_mult = 0.3
            
        # Duration weighting (cap at 3 years)
        dur_mult = min(1.0, dur / 36.0)
        
        # CRITICAL: Is this skill evidenced in career history or just listed?
        # Skills verified in history get full weight; skills only in list get 40%
        name_tokens = name.split()
        evidenced_in_history = any(token in full_evidence_text for token in name_tokens if len(token) > 2)
        evidence_mult = 1.0 if evidenced_in_history else 0.4
        
        skill_score = skill_tier_weight * prof_mult * (0.5 + 0.5 * dur_mult) * evidence_mult
        score += skill_score
        matches += 1
        
    if matches == 0:
        return 0.0
        
    # Average matched skill quality scaled by number of matches (diminishing returns)
    norm_score = min(1.0, (score / matches) * (0.6 + 0.4 * min(1.0, matches / 5.0)))
    return norm_score

def score_candidate(
    candidate: Dict[str, Any], 
    llm_features: Dict[str, Any], 
    gate_multiplier: float
) -> Dict[str, Any]:
    """
    Scores a candidate combining core fit, logistics, education, behavioral, and gate multipliers.
    """
    # Core Fit component: combines pre-computed LLM scores and skill match score
    prod_exp = llm_features.get("production_experience_score", 0.0)
    eval_depth = llm_features.get("eval_depth_score", 0.0)
    skill_match = calculate_skill_match_score(candidate)
    
    core_fit_score = (0.50 * prod_exp + 0.30 * eval_depth + 0.20 * skill_match)
    
    # Other components
    logistics_score = calculate_logistics_score(candidate)
    education_score = calculate_education_score(candidate)
    behavioral_mult = calculate_behavioral_multiplier(candidate)
    
    # Combined base score
    base_score = (
        config.CORE_FIT_WEIGHT * core_fit_score +
        config.LOGISTICS_WEIGHT * logistics_score +
        config.EDUCATION_WEIGHT * education_score
    )
    
    # Apply multipliers
    final_score = base_score * behavioral_mult * gate_multiplier
    
    # Store internal sub-scores for narrative generation and debugging
    candidate_scored = candidate.copy()
    candidate_scored["_core_fit_score"] = round(core_fit_score, 3)
    candidate_scored["_logistics_score"] = round(logistics_score, 3)
    candidate_scored["_education_score"] = round(education_score, 3)
    candidate_scored["_behavioral_mult"] = round(behavioral_mult, 3)
    candidate_scored["_gate_multiplier"] = round(gate_multiplier, 3)
    candidate_scored["base_score"] = round(base_score, 4)
    candidate_scored["final_score"] = round(final_score, 4)
    candidate_scored["fit_narrative"] = llm_features.get("fit_narrative", "")
    
    return candidate_scored
