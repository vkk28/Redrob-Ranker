import re
from datetime import datetime
from typing import Dict, Any, List
from src.utils import get_logger, parse_date

logger = get_logger("honeypot_detection")

FICTIONAL_COMPANIES = {
    "dunder mifflin", "acme corp", "acme corporation", "globex", "globex inc",
    "globex corporation", "wayne enterprises", "stark industries", "initech",
    "umbrella corporation", "umbrella corp", "hooli", "pied piper",
    "wonka industries", "cyberdyne systems", "cyberdyne", "weyland-yutani",
    "weyland yutani", "soylent corp", "soylent corporation", "massive dynamic",
    "ingen", "oscorp", "oscorp industries", "lexcorp", "lex corp",
    "tyrell corporation", "tyrell corp", "prestige worldwide",
    "sabre", "vance refrigeration", "michael scott paper company",
    "los pollos hermanos", "sterling cooper", "paper street soap",
}

def is_honeypot(candidate: Dict[str, Any], current_year: int = 2026) -> bool:
    """
    Detect if a candidate is a honeypot (trap) based on structural inconsistencies.
    Returns True if the candidate is flagged as a honeypot, False otherwise.
    """
    profile = candidate.get("profile", {})
    history = candidate.get("career_history", [])
    skills = candidate.get("skills", [])
    education = candidate.get("education", [])
    signals = candidate.get("redrob_signals", {})

    # 0. Check for fictional/impossible company names (fast-path rejection)
    current_company = (profile.get("current_company") or "").lower().strip()
    if current_company in FICTIONAL_COMPANIES:
        logger.debug(f"Honeypot: Fictional company '{current_company}' in current_company")
        return True
    for role in history:
        company = (role.get("company") or "").lower().strip()
        if company in FICTIONAL_COMPANIES:
            logger.debug(f"Honeypot: Fictional company '{company}' in career history")
            return True

    # 1. Check for basic date order consistency in career history
    for role in history:
        start_str = role.get("start_date")
        end_str = role.get("end_date")
        
        start = parse_date(start_str)
        end = parse_date(end_str) if end_str else datetime(current_year, 7, 1)

        if start and end and start > end:
            logger.debug(f"Honeypot: Impossible date ordering for role: {role.get('title')} at {role.get('company')} (start={start_str}, end={end_str})")
            return True

        # Check for roles starting far in the future
        if start and start > datetime(current_year, 12, 31):
            logger.debug(f"Honeypot: Role starting in future: {role.get('title')} at {role.get('company')} (start={start_str})")
            return True

    # 2. Check for "expert" proficiency with <= 1 month of duration
    for skill in skills:
        prof = skill.get("proficiency", "").lower()
        dur = skill.get("duration_months", 0)
        # Handle cases where duration_months is None or missing
        if dur is None:
            dur = 0
        if prof == "expert" and dur <= 1:
            logger.debug(f"Honeypot: Expert skill '{skill.get('name')}' with <= 1 month duration (duration={dur})")
            return True

    # 3. Check for massive discrepancies between claimed YOE and career history duration
    # Since candidates might have concurrent roles or gaps, let's build non-overlapping intervals
    intervals = []
    for role in history:
        start_str = role.get("start_date")
        end_str = role.get("end_date")
        
        start = parse_date(start_str)
        end = parse_date(end_str) if end_str else datetime(current_year, 7, 1)
        
        if start and end:
            intervals.append((start, end))
            
    # Compute total non-overlapping months of experience
    total_months = 0
    if intervals:
        intervals.sort(key=lambda x: x[0])
        merged = []
        current_start, current_end = intervals[0]
        for start, end in intervals[1:]:
            if start <= current_end:
                current_end = max(current_end, end)
            else:
                merged.append((current_start, current_end))
                current_start, current_end = start, end
        merged.append((current_start, current_end))
        
        for start, end in merged:
            total_months += (end - start).days / 30.437

    claimed_yoe = profile.get("years_of_experience", 0)
    claimed_yoe_months = claimed_yoe * 12
    
    # If candidate claims 0 YOE but has more than 12 months history, or vice versa
    if total_months > 0:
        # If claimed YOE is less than 30% of actual history months, or more than 300% (and both are significant)
        if claimed_yoe_months < total_months * 0.30 and total_months > 12:
            logger.debug(f"Honeypot: Claimed YOE ({claimed_yoe} yrs) is significantly less than career history ({total_months/12:.1f} yrs)")
            return True
        if claimed_yoe_months > total_months * 3.0 and claimed_yoe_months > 24:
            logger.debug(f"Honeypot: Claimed YOE ({claimed_yoe} yrs) is significantly more than career history ({total_months/12:.1f} yrs)")
            return True

    # 4. Check for overlapping concurrent roles that aren't part-time or advisory
    # If the candidate has multiple full-time roles at different companies that overlap by more than 6 months
    sorted_roles = sorted(
        [r for r in history if r.get("start_date")],
        key=lambda r: r["start_date"]
    )
    for i in range(len(sorted_roles)):
        for j in range(i + 1, len(sorted_roles)):
            r1 = sorted_roles[i]
            r2 = sorted_roles[j]
            
            # Check if different companies
            if r1.get("company", "").lower() == r2.get("company", "").lower():
                continue
                
            start1 = parse_date(r1.get("start_date"))
            end1 = parse_date(r1.get("end_date")) if r1.get("end_date") else datetime(current_year, 7, 1)
            
            start2 = parse_date(r2.get("start_date"))
            end2 = parse_date(r2.get("end_date")) if r2.get("end_date") else datetime(current_year, 7, 1)
            
            if start1 and end1 and start2 and end2:
                # Find overlap
                overlap_start = max(start1, start2)
                overlap_end = min(end1, end2)
                
                if overlap_start < overlap_end:
                    overlap_days = (overlap_end - overlap_start).days
                    overlap_months = overlap_days / 30.437
                    
                    # If overlap is greater than 6 months
                    if overlap_months > 6.0:
                        # Exclude roles that look advisory, part-time, freelance, or internship
                        t1 = r1.get("title", "").lower()
                        t2 = r2.get("title", "").lower()
                        d1 = r1.get("description", "").lower()
                        d2 = r2.get("description", "").lower()
                        
                        keywords = ["adviser", "advisor", "consultant", "part-time", "part time", "freelance", "intern", "contractor", "founder", "co-founder"]
                        is_flexible1 = any(kw in t1 or kw in d1 for kw in keywords)
                        is_flexible2 = any(kw in t2 or kw in d2 for kw in keywords)
                        
                        if not (is_flexible1 or is_flexible2):
                            logger.debug(f"Honeypot: Overlapping concurrent full-time roles: '{r1.get('title')}' at {r1.get('company')} and '{r2.get('title')}' at {r2.get('company')} overlapping by {overlap_months:.1f} months")
                            return True

    # 5. Check for education years that are completely out of line with career history
    # E.g. starting working full-time roles before graduating.
    # We should only check degrees that are full-time (B.E., B.Tech, M.S., Ph.D.) and not online/distance.
    for edu in education:
        end_yr = edu.get("end_year")
        if end_yr and history:
            # Earliest full-time career start
            earliest_ft_start = None
            for role in history:
                title = role.get("title", "").lower()
                desc = role.get("description", "").lower()
                if "intern" in title or "intern" in desc:
                    continue
                start_dt = parse_date(role.get("start_date"))
                if start_dt:
                    if earliest_ft_start is None or start_dt < earliest_ft_start:
                        earliest_ft_start = start_dt
            
            if earliest_ft_start:
                # If they started full-time work more than 2 years before graduating from their primary degree
                # (e.g. bachelor's or master's)
                degree = edu.get("degree", "").lower()
                is_major_degree = any(d in degree for d in ["b.", "m.", "ph.d", "bachelor", "master", "doctor"])
                if is_major_degree and earliest_ft_start.year < (end_yr - 2):
                    logger.debug(f"Honeypot: Started working full-time in {earliest_ft_start.year} but graduated {edu.get('degree')} in {end_yr}")
                    return True

    # 6. Endorsements received is disproportionate to connection count
    conn_count = signals.get("connection_count", 0)
    endorsements = signals.get("endorsements_received", 0)
    if conn_count == 0 and endorsements > 10:
        logger.debug(f"Honeypot: Endorsements received ({endorsements}) with 0 connections")
        return True
    if conn_count > 0 and endorsements > conn_count * 5.0 and endorsements > 50:
        logger.debug(f"Honeypot: High endorsements ({endorsements}) relative to connections ({conn_count})")
        return True

    return False

def is_consulting_only(candidate: Dict[str, Any]) -> bool:
    """
    Check if the candidate has only worked at WITCH-tier consulting firms
    (TCS, Infosys, Wipro, Accenture, Cognizant, Capgemini).
    """
    history = candidate.get("career_history", [])
    profile = candidate.get("profile", {})
    
    witch_tier = {"tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini"}
    
    companies = set()
    current_company = profile.get("current_company")
    if current_company:
        companies.add(current_company.lower())
        
    for role in history:
        comp = role.get("company")
        if comp:
            companies.add(comp.lower())
            
    # Filter out empty strings
    companies = {c for c in companies if c.strip()}
    
    if not companies:
        return False
        
    # Check if companies is a subset of WITCH-tier
    # In python, issubset checks if all elements of companies are in witch_tier
    # We must also normalize checks (e.g. "tata consultancy services" -> "tcs", "wipro technologies" -> "wipro")
    normalized_companies = set()
    for c in companies:
        if "tata consultancy" in c or "tcs" in c:
            normalized_companies.add("tcs")
        elif "infosys" in c:
            normalized_companies.add("infosys")
        elif "wipro" in c:
            normalized_companies.add("wipro")
        elif "accenture" in c:
            normalized_companies.add("accenture")
        elif "cognizant" in c or "cts" in c:
            normalized_companies.add("cognizant")
        elif "capgemini" in c:
            normalized_companies.add("capgemini")
        else:
            normalized_companies.add(c)
            
    return normalized_companies.issubset(witch_tier)
