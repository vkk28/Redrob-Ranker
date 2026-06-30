import argparse
import time
import sys
import os
import resource
from pathlib import Path
import numpy as np
import json

# Add base directory to path so we can import src
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.utils import get_logger
from src.data_loader import stream_candidates
from src.hard_gates import apply_gates
from src.scoring import score_candidate
from src.cross_encoder_rerank import rerank_candidates
from src.tier5_finder import find_tier5_candidates
from src.assembly import assemble_final_csv, write_submission_csv
from src.llm_feature_extraction import local_extract_features
import src.config as config
from validate_submission import validate_submission

logger = get_logger("benchmark_compute")

def get_peak_ram_mb() -> float:
    # On macOS, maxrss is in bytes, on Linux it is in kilobytes.
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return usage / (1024 * 1024)
    else:
        return usage / 1024

def main():
    parser = argparse.ArgumentParser(description="Benchmark computational performance (timed step).")
    parser.add_argument("--candidates", type=str, default=str(config.CANDIDATES_JSONL), help="Path to candidates pool file")
    parser.add_argument("--out", type=str, default="benchmark_submission.csv", help="Path to output CSV")
    args = parser.parse_args()

    t_start = time.time()
    
    # 1. Loading Artifacts
    try:
        from src.cross_encoder_rerank import get_cross_encoder
        ce_model = get_cross_encoder()
        ce_model.predict([("a", "b")])
    except Exception as e:
        logger.warning(f"Failed PyTorch warm up: {e}")
        
    import faiss
    t0 = time.time()
    logger.info("Loading pre-computed artifacts...")
    jd_embedding = np.load(config.ARTIFACTS_DIR / "jd_embedding.npy")
    index = faiss.read_index(str(config.ARTIFACTS_DIR / "faiss_index.bin"))
    with open(config.ARTIFACTS_DIR / "candidate_id_map.json", "r", encoding="utf-8") as f:
        candidate_id_map = json.load(f)
    id_map = {int(k): v for k, v in candidate_id_map.items()}

    # Load pre-computed LLM features
    llm_features_dict = {}
    llm_features_path = config.ARTIFACTS_DIR / "llm_features.jsonl"
    if llm_features_path.exists():
        with open(llm_features_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    feat = json.loads(line)
                    llm_features_dict[feat["candidate_id"]] = feat
                    
    t_artifacts = time.time() - t0
    logger.info(f"Loaded artifacts in {t_artifacts:.2f} seconds. Peak RAM: {get_peak_ram_mb():.1f} MB")

    # 2. Loading Pool
    t0 = time.time()
    logger.info("Loading candidates pool...")
    candidates_dict = {}
    all_candidates_list = []
    
    for cand in stream_candidates(args.candidates, validate=False):
        candidates_dict[cand["candidate_id"]] = cand
        all_candidates_list.append(cand)
        
    t_load = time.time() - t0
    logger.info(f"Loaded {len(candidates_dict)} candidates in {t_load:.2f} seconds. Peak RAM: {get_peak_ram_mb():.1f} MB")

    # 3. FAISS Retrieval
    t0 = time.time()
    logger.info("Running FAISS search...")
    top_n = min(len(candidates_dict), config.REDUCED_POOL_SIZE)
    distances, indices = index.search(jd_embedding.reshape(1, -1), top_n)
    
    retrieved_cands = []
    for idx in indices[0]:
        cand_id = id_map.get(idx)
        if cand_id and cand_id in candidates_dict:
            retrieved_cands.append(candidates_dict[cand_id])
            
    t_faiss = time.time() - t0
    logger.info(f"FAISS retrieved {len(retrieved_cands)} candidates in {t_faiss:.3f} seconds. Peak RAM: {get_peak_ram_mb():.1f} MB")

    # 4. Gating & Scoring
    t0 = time.time()
    logger.info("Applying gates and scoring...")
    scored_candidates = []
    for cand in retrieved_cands:
        gate_mult, reason = apply_gates(cand)
        if gate_mult == 0.0:
            continue
            
        cid = cand["candidate_id"]
        feat = llm_features_dict.get(cid)
        if not feat:
            feat = local_extract_features(cand)
            
        scored = score_candidate(cand, feat, gate_mult)
        scored_candidates.append(scored)
        
    t_gating = time.time() - t0
    logger.info(f"Gates & Core scored {len(scored_candidates)} candidates in {t_gating:.2f} seconds. Peak RAM: {get_peak_ram_mb():.1f} MB")

    # 5. CrossEncoder Rerank
    t0 = time.time()
    logger.info("Running CrossEncoder rerank on top candidates...")
    scored_candidates.sort(key=lambda c: c["final_score"], reverse=True)
    top_k_pool = scored_candidates[:config.CROSS_ENCODER_TOP_K]
    
    from src.embeddings import get_candidate_text
    candidate_texts = [get_candidate_text(c) for c in top_k_pool]
    
    jd_query = "Senior AI Engineer Founding Team. Production embeddings retrieval, vector databases, eval frameworks, python."
    reranked_pairs = rerank_candidates(jd_query, top_k_pool, candidate_texts)
    
    for cand, ce_score in reranked_pairs:
        cand["final_ranked_score"] = 0.7 * cand["final_score"] + 0.3 * ce_score
        
    top_k_pool.sort(key=lambda c: c["final_ranked_score"], reverse=True)
    t_rerank = time.time() - t0
    logger.info(f"CrossEncoder rerank completed in {t_rerank:.2f} seconds. Peak RAM: {get_peak_ram_mb():.1f} MB")

    # 6. Tier-5 Scan
    t0 = time.time()
    logger.info("Scanning full pool for Tier-5 candidates...")
    full_pool_evaluated = []
    for cand in all_candidates_list:
        gate_mult, _ = apply_gates(cand)
        if gate_mult == 0.0:
            continue
        feat = llm_features_dict.get(cand["candidate_id"])
        if not feat:
            feat = local_extract_features(cand)
        scored = score_candidate(cand, feat, gate_mult)
        full_pool_evaluated.append(scored)
        
    tier_5_includes = find_tier5_candidates(full_pool_evaluated, count=5)
    t_tier5 = time.time() - t0
    logger.info(f"Tier-5 scan completed in {t_tier5:.2f} seconds. Peak RAM: {get_peak_ram_mb():.1f} MB")

    # 7. Assembly and Write CSV
    t0 = time.time()
    logger.info("Assembling final top 100...")
    final_100 = assemble_final_csv(
        top_200_fused=top_k_pool,
        tier_5_force_includes=tier_5_includes,
        target_size=100,
        tier5_slot=5
    )
    write_submission_csv(final_100, args.out)
    t_assembly = time.time() - t0
    
    t_total = time.time() - t_start
    peak_ram = get_peak_ram_mb()
    
    logger.info("================================================================================")
    logger.info("BENCHMARK COMPUTE RESULTS")
    logger.info("================================================================================")
    logger.info(f"Artifacts load:   {t_artifacts:.2f}s")
    logger.info(f"Candidate pool load: {t_load:.2f}s")
    logger.info(f"FAISS search:      {t_faiss:.3f}s")
    logger.info(f"Gating & scoring:  {t_gating:.2f}s")
    logger.info(f"CrossEncoder rerank: {t_rerank:.2f}s")
    logger.info(f"Tier-5 candidate scan: {t_tier5:.2f}s")
    logger.info(f"Assembly & write CSV: {t_assembly:.2f}s")
    logger.info(f"Total timed runtime: {t_total:.2f}s (Budget: ≤ 300s)")
    logger.info(f"Peak memory usage:   {peak_ram:.1f} MB (Budget: ≤ 16,384 MB)")
    logger.info("================================================================================")
    
    # Run validate_submission to make sure the benchmark CSV is valid
    errors = validate_submission(args.out)
    if errors:
        logger.error(f"Benchmark CSV failed validation with {len(errors)} error(s)!")
    else:
        logger.info("Benchmark CSV validation passed successfully.")

if __name__ == '__main__':
    main()
