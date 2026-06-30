from typing import Dict, Any, Tuple
from src.honeypot_detection import is_honeypot, is_consulting_only
from src.utils import get_logger

logger = get_logger("hard_gates")

# Specific titles
TARGET_TITLES = [
    "ai engineer", "ml engineer", "machine learning engineer", "nlp engineer", 
    "deep learning engineer", "search engineer", "recommendation engineer", 
    "nlp research", "speech engineer", "computer vision engineer", "founding engineer"
]

ADJACENT_TITLES = [
    "backend engineer", "software engineer", "data engineer", "data scientist", 
    "full stack engineer", "fullstack engineer", "systems engineer", "tech lead", 
    "technical lead", "engineering manager"
]

MISMATCH_TITLES = [
    "marketing", "sales", "hr ", "recruiter", "operations", "ui/ux", "product manager",
    "project manager", "designer", "scrum master", "business analyst", "finance",
    "mechanical engineer", "civil engineer", "electrical engineer", "chemical engineer",
    "manufacturing engineer", "industrial engineer", "quality engineer", "qa manager",
    "content writer", "copywriter", "graphic designer", "accountant",
]

def check_title_match(title: str) -> float:
    """
    Check how well the candidate's title matches target roles.
    Returns a score multiplier:
    1.0: Perfect match
    0.7: Adjacent match (e.g. backend/data engineer)
    0.1: Complete mismatch (e.g. marketing manager)
    """
    title_lower = title.lower()
    
    # Check mismatch first
    if any(m in title_lower for m in MISMATCH_TITLES):
        return 0.1
        
    # Check target titles
    if any(t in title_lower for t in TARGET_TITLES):
        return 1.0
        
    # Check adjacent titles
    if any(a in title_lower for a in ADJACENT_TITLES):
        return 0.7
        
    # Default fallback: if it doesn't match mismatch but doesn't match adjacent either
    # It might be a generic "engineer" or "developer"
    if "engineer" in title_lower or "developer" in title_lower or "programmer" in title_lower:
        return 0.6
        
    return 0.3

def apply_gates(candidate: Dict[str, Any], current_year: int = 2026) -> Tuple[float, str]:
    """
    Apply hard and soft gates to candidates.
    Returns:
        multiplier: A float between 0.0 (disqualified) and 1.0 (perfect pass)
        reason: Description of the gate status
    """
    # 1. Honeypots: strict exclusion
    if is_honeypot(candidate, current_year):
        return 0.0, "disqualified_honeypot"

    # 2. Consulting-only WITCH-tier: strict exclusion
    if is_consulting_only(candidate):
        return 0.0, "disqualified_consulting_only"

    # 3. Job Title matching
    profile = candidate.get("profile", {})
    current_title = profile.get("current_title", "")
    title_multiplier = check_title_match(current_title)

    # 4. Domain / Summary mismatch checks
    # If the summary is completely non-technical, or the current industry is mismatch
    industry = profile.get("current_industry", "").lower()
    summary = profile.get("summary", "").lower()
    
    domain_multiplier = 1.0
    if "marketing" in industry or "sales" in industry or "human resources" in industry or "retail" in industry:
        domain_multiplier = 0.7
        
    # Check for double soft gate (adjacent title and adjacent/mismatched domain)
    overall_multiplier = title_multiplier * domain_multiplier
    
    if overall_multiplier <= 0.1:
        return 0.0, "disqualified_title_domain_mismatch"
        
    if overall_multiplier < 1.0:
        return overall_multiplier, f"soft_gated_penalty_{overall_multiplier:.2f}"

    return 1.0, "passed_gates"
