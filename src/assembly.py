import csv
from typing import List, Dict, Any
from src.utils import get_logger
from src.narrative_generation import assemble_reasoning

logger = get_logger("assembly")

def assemble_final_csv(
    top_200_fused: List[Dict[str, Any]], 
    tier_5_force_includes: List[Dict[str, Any]], 
    target_size: int = 100, 
    tier5_slot: int = 5
) -> List[Dict[str, Any]]:
    """
    Combines the top-ranking candidates (first 95) with Tier-5 candidates (5 candidates),
    ensuring zero duplicates and preserving score monotonicity.
    """
    top95 = top_200_fused[:target_size - tier5_slot]
    top95_ids = {c["candidate_id"] for c in top95}
    
    # Filter Tier-5s that are NOT already in the top 95 to avoid duplicates
    tier_5_unique = [
        c for c in tier_5_force_includes if c["candidate_id"] not in top95_ids
    ]
    
    # Slice unique ones to fit the remaining slots
    tier_5_unique = tier_5_unique[:tier5_slot]
    
    final_100 = list(top95) + tier_5_unique
    
    # Backfill if we are short of 100 (e.g. not enough unique Tier-5s)
    if len(final_100) < target_size:
        included_ids = {c["candidate_id"] for c in final_100}
        backfill_pool = top_200_fused[target_size - tier5_slot:]
        for candidate in backfill_pool:
            if len(final_100) == target_size:
                break
            if candidate["candidate_id"] not in included_ids:
                final_100.append(candidate)
                included_ids.add(candidate["candidate_id"])
                
    assert len(final_100) == target_size, (
        f"Assembly produced {len(final_100)} rows, expected {target_size} — "
        "check candidate pools."
    )
    assert len({c["candidate_id"] for c in final_100}) == target_size, "duplicate ids in final assembly"
    
    # Score reconciliation: ensure Tier-5 unique additions are scored below the 95th candidate
    # to maintain strict monotonicity.
    floor_score = top95[-1]["final_ranked_score"] if top95 else 1.0
    for i, c in enumerate(tier_5_unique):
        c["final_ranked_score"] = floor_score - (0.001 * (i + 1))
        
    # Re-sort to maintain score ordering
    final_100.sort(key=lambda c: c["final_ranked_score"], reverse=True)
    
    # Assign rounded scores first
    for c in final_100:
        c["score"] = round(c["final_ranked_score"], 4)
        
    # Re-sort to guarantee:
    # 1. score is non-increasing
    # 2. ties are broken by candidate_id ascending (alphabetically)
    final_100.sort(key=lambda c: (-c["score"], c["candidate_id"]))
    
    # Assign ranks and finalize reasonings
    for i, c in enumerate(final_100, 1):
        c["rank"] = i
        c["reasoning"] = assemble_reasoning(c, i, c.get("fit_narrative", ""))
        
    return final_100

def write_submission_csv(candidates_list: List[Dict[str, Any]], out_path: str):
    """
    Write the final 100 candidates to a CSV file matching the validator specification.
    """
    logger.info(f"Writing final submission CSV to {out_path}...")
    headers = ["candidate_id", "rank", "score", "reasoning"]
    
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        
        for cand in candidates_list:
            writer.writerow([
                cand["candidate_id"],
                cand["rank"],
                cand["score"],
                cand["reasoning"]
            ])
            
    logger.info(f"Successfully wrote {len(candidates_list)} rows to {out_path}")
