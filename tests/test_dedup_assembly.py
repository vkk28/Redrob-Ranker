import pytest
from src.assembly import assemble_final_csv

def test_dedup_assembly():
    # Construct mock top-200 fused candidates list
    top_200_fused = []
    for i in range(1, 151):
        top_200_fused.append({
            "candidate_id": f"CAND_{i:07d}",
            "final_ranked_score": 1.0 - (i * 0.005),
            "fit_narrative": f"Candidate {i} description."
        })

    # Suppose we have 5 Tier-5 candidates.
    # 2 of them are already in the top 95 (CAND_0000010, CAND_0000050)
    # 3 of them are new (CAND_0000140, CAND_0000201, CAND_0000202)
    tier_5_force_includes = [
        {"candidate_id": "CAND_0000010", "final_ranked_score": 0.95, "fit_narrative": "Tier 5 No.1"},
        {"candidate_id": "CAND_0000050", "final_ranked_score": 0.75, "fit_narrative": "Tier 5 No.2"},
        {"candidate_id": "CAND_0000140", "final_ranked_score": 0.30, "fit_narrative": "Tier 5 No.3"},
        {"candidate_id": "CAND_0000201", "final_ranked_score": 0.20, "fit_narrative": "Tier 5 No.4"},
        {"candidate_id": "CAND_0000202", "final_ranked_score": 0.10, "fit_narrative": "Tier 5 No.5"}
    ]

    final_100 = assemble_final_csv(
        top_200_fused=top_200_fused,
        tier_5_force_includes=tier_5_force_includes,
        target_size=100,
        tier5_slot=5
    )

    # Assertions
    assert len(final_100) == 100
    assert len({c["candidate_id"] for c in final_100}) == 100
    
    # Check that scores are strictly non-increasing
    scores = [c["score"] for c in final_100]
    for a, b in zip(scores, scores[1:]):
        assert a >= b, f"Monotonicity violated: {a} < {b}"
        
    # Check that force includes that were not in the top 95 are present
    included_ids = {c["candidate_id"] for c in final_100}
    assert "CAND_0000140" in included_ids
    assert "CAND_0000201" in included_ids
    assert "CAND_0000202" in included_ids
    
    # Check rank order consistency
    ranks = [c["rank"] for c in final_100]
    assert ranks == list(range(1, 101))
