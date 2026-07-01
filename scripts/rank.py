import argparse
import json
import os
import sys
from pathlib import Path
import numpy as np
# Add base directory to path so we can import src
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.utils import get_logger
from src.data_loader import stream_candidates
from src.hard_gates import apply_gates
from src.scoring import score_candidate
from src.cross_encoder_rerank import rerank_candidates, get_cross_encoder
from src.tier5_finder import find_tier5_candidates
from src.assembly import assemble_final_csv, write_submission_csv
from src.jd_parser import parse_job_description
from src.llm_feature_extraction import local_extract_features
import src.config as config
from validate_submission import validate_submission

logger = get_logger("rank")

def main():
    parser = argparse.ArgumentParser(description="Phase 2: Timed candidate ranking.")
    parser.add_argument("--candidates", type=str, required=True, help="Path to candidates pool file (.jsonl or .jsonl.gz)")
    parser.add_argument("--out", type=str, required=True, help="Path to write the final ranked CSV")
    parser.add_argument("--verbose", action="store_true", help="Print detailed validation and cohort statistics of the final top-100 list")
    args = parser.parse_args()
    
    # 1. Load Pre-Computed Artifacts
    logger.info("Warming up PyTorch runtime...")
    try:
        ce_model = get_cross_encoder()
        ce_model.predict([("a", "b")])
    except Exception as e:
        logger.warning(f"Failed PyTorch warm up: {e}")
        
    import faiss
    logger.info("Loading pre-computed artifacts...")
    jd_embedding = np.load(config.ARTIFACTS_DIR / "jd_embedding.npy")
    index = faiss.read_index(str(config.ARTIFACTS_DIR / "faiss_index.bin"))
    
    with open(config.ARTIFACTS_DIR / "candidate_id_map.json", "r", encoding="utf-8") as f:
        candidate_id_map = json.load(f)
        
    # Map index to candidate ID
    # candidate_id_map keys are strings from json load, convert to int
    id_map = {int(k): v for k, v in candidate_id_map.items()}

    # Load LLM features
    logger.info("Loading LLM pre-computed features...")
    llm_features_dict = {}
    llm_features_path = config.ARTIFACTS_DIR / "llm_features.jsonl"
    if llm_features_path.exists():
        with open(llm_features_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    feat = json.loads(line)
                    llm_features_dict[feat["candidate_id"]] = feat
    
    # 2. Retrieve Top Candidates via FAISS
    logger.info("Step 2: Retrieving candidates using HNSW index...")
    top_n = min(len(id_map), config.REDUCED_POOL_SIZE)
    distances, indices = index.search(jd_embedding.reshape(1, -1), top_n)
    
    retrieved_ids = set()
    for idx in indices[0]:
        cand_id = id_map.get(idx)
        if cand_id:
            retrieved_ids.add(cand_id)
            
    # 3. Stream and load only the retrieved candidates from input file
    logger.info("Step 3: Streaming and parsing retrieved candidates pool...")
    retrieved_cands_dict = {}
    for cand in stream_candidates(args.candidates, validate=False):
        cid = cand["candidate_id"]
        if cid in retrieved_ids:
            retrieved_cands_dict[cid] = cand
            
    retrieved_cands = []
    # Preserve FAISS ranking order in retrieved_cands list
    for idx in indices[0]:
        cand_id = id_map.get(idx)
        if cand_id and cand_id in retrieved_cands_dict:
            retrieved_cands.append(retrieved_cands_dict[cand_id])
            
    logger.info(f"Retrieved {len(retrieved_cands)} candidates from FAISS index.")

    # 4. Gating & Core Scoring
    logger.info("Step 4: Applying gates and scoring candidates...")
    scored_candidates = []
    for cand in retrieved_cands:
        gate_mult, reason = apply_gates(cand)
        if gate_mult == 0.0:
            continue
            
        # Get precomputed LLM features or extract them locally
        cid = cand["candidate_id"]
        feat = llm_features_dict.get(cid)
        if not feat:
            feat = local_extract_features(cand)
            
        scored = score_candidate(cand, feat, gate_mult)
        scored_candidates.append(scored)
        
    logger.info(f"Scored {len(scored_candidates)} candidates after hard-gating.")

    # Sort by final_score to find top 200 candidates for reranking
    scored_candidates.sort(key=lambda c: c["final_score"], reverse=True)
    top_k_pool = scored_candidates[:config.CROSS_ENCODER_TOP_K]

    # 5. Cross-Encoder Rerank
    logger.info("Step 5: Reranking top candidates via Cross-Encoder...")
    from src.embeddings import get_candidate_text
    
    # Generate query text matching what the cross-encoder was cached on
    jd_reqs = parse_job_description(config.JOB_DESCRIPTION_MD)
    jd_query = (
        f"Senior AI Engineer Founding Team. "
        f"Required: production embeddings-based retrieval systems, vector databases (FAISS, Milvus, Pinecone), evaluation frameworks (NDCG, MAP, A/B test), python."
    )
    
    candidate_texts = [get_candidate_text(c) for c in top_k_pool]
    
    reranked_pairs = rerank_candidates(jd_query, top_k_pool, candidate_texts)
    
    # Calculate final fused score
    for cand, ce_score in reranked_pairs:
        # Fuse base final_score with cross-encoder score
        cand["final_ranked_score"] = 0.7 * cand["final_score"] + 0.3 * ce_score

    # Re-sort pool by fused score
    top_k_pool.sort(key=lambda c: c["final_ranked_score"], reverse=True)

    # 6. Find Tier-5 Candidates (force includes)
    # Scan the scored FAISS-retrieved pool for hidden-gem candidates
    # (classic search/rec builders without trendy AI buzzwords)
    logger.info("Step 6: Scanning scored pool for Tier-5 candidates...")
    tier_5_includes = find_tier5_candidates(scored_candidates, count=5)

    # 7. Deduplicate & Final Assembly
    logger.info("Step 7: Assembling final top-100 list...")
    final_100 = assemble_final_csv(
        top_200_fused=top_k_pool,
        tier_5_force_includes=tier_5_includes,
        target_size=100,
        tier5_slot=5
    )

    # 8. Output CSV
    write_submission_csv(final_100, args.out)

    # 9. Verify CSV compliant with validation rules
    logger.info("Verifying submission CSV against validator...")
    errors = validate_submission(args.out)
    if errors:
        logger.error(f"Validator failed with {len(errors)} error(s):")
        for e in errors:
            logger.error(f"- {e}")
        sys.exit(1)
    else:
        logger.info("Validator verification successful! CSV is 100% compliant.")
        if args.verbose:
            print("\n" + "="*80)
            print("VERBOSE SCORES AND COHORT REPORT")
            print("="*80)
            avg_score = sum(c["score"] for c in final_100) / len(final_100)
            avg_yoe = sum(c.get("profile", {}).get("years_of_experience", 0) for c in final_100) / len(final_100)
            locations = [c.get("profile", {}).get("location", "Unknown").strip() for c in final_100]
            from collections import Counter
            loc_counts = Counter(locations)
            
            print(f"Cohort Size:           {len(final_100)} candidates")
            print(f"Average Fusion Score:  {avg_score:.4f}")
            print(f"Average Experience:    {avg_yoe:.2f} years")
            print("\nTop Location Distribution:")
            for loc, count in loc_counts.most_common(4):
                print(f"  - {loc}: {count} candidates")
                
            print("\nTop 5 Candidates Preview:")
            for c in final_100[:5]:
                p = c.get("profile", {})
                print(f"  Rank {c['rank']}: ID={c['candidate_id']} | Score={c['score']:.4f} | {p.get('current_title', 'Engineer')} at {p.get('current_company', 'N/A')} ({p.get('years_of_experience', 0)} YOE)")
            print("="*80 + "\n")

if __name__ == '__main__':
    main()
