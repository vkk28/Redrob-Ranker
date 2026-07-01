import argparse
import json
import os
import sys
from pathlib import Path
import numpy as np
# Add base directory to path so we can import src
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.utils import get_logger
from src.jd_parser import parse_job_description
from src.data_loader import stream_candidates
from src.embeddings import embed_text, get_candidate_text
from src.skill_ontology import SkillOntology
from src.llm_feature_extraction import run_extraction
from src.cross_encoder_rerank import save_cross_encoder_locally
import src.config as config

logger = get_logger("setup_artifacts")

def main():
    parser = argparse.ArgumentParser(description="Phase 1: Pre-compute embeddings, indexes, and LLM features.")
    parser.add_argument("--candidates", type=str, default=str(config.CANDIDATES_JSONL), help="Path to candidates jsonl")
    parser.add_argument("--limit-embed", type=int, default=100000, help="Limit number of real embeddings generated (use lower values for dev speed; default embeds all)")
    args = parser.parse_args()

    # 1. Parse JD
    logger.info("Step 1: Parsing Job Description...")
    print("[TRACE] Start parsing JD")
    jd_reqs = parse_job_description(config.JOB_DESCRIPTION_MD)
    print("[TRACE] Finished parsing JD")
    
    # Embed JD
    logger.info("Embedding Job Description...")
    jd_text = (
        f"Role: {jd_reqs.role_name}. "
        f"Must haves: {', '.join(jd_reqs.must_have_embeddings + jd_reqs.must_have_vector_db)}. "
        f"Preferred location: {', '.join(jd_reqs.preferred_locations)}."
    )
    print("[TRACE] Start embedding JD")
    jd_embedding = embed_text(jd_text, is_query=True)
    print("[TRACE] Finished embedding JD")
    
    # Save JD embedding
    config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    np.save(config.ARTIFACTS_DIR / "jd_embedding.npy", jd_embedding)
    logger.info("Saved jd_embedding.npy")
    print("[TRACE] Saved JD embedding to disk")
    import faiss

    # 2. Ingest and Embed Candidates
    logger.info("Step 2: Loading candidate list and generating embeddings...")
    print("[TRACE] Start streaming candidates")
    candidates = list(stream_candidates(args.candidates, validate=False))
    num_candidates = len(candidates)
    logger.info(f"Loaded {num_candidates} candidates.")
    print(f"[TRACE] Finished streaming {num_candidates} candidates")
    
    # We will generate real embeddings for a subset (up to args.limit_embed)
    # and fill the rest with dummy embeddings to satisfy the 100,000 size and shape constraint of the FAISS index
    limit = min(num_candidates, args.limit_embed)
    logger.info(f"Generating real embeddings for first {limit} candidates...")
    print("[TRACE] Generating real candidate texts")
    
    candidate_texts = [get_candidate_text(c) for c in candidates[:limit]]
    print("[TRACE] Finished generating candidate texts")
    
    # Embed candidates in batches
    batch_size = 256
    real_embeddings = []
    print("[TRACE] Starting batch embedding loop")
    for i in range(0, len(candidate_texts), batch_size):
        batch = candidate_texts[i:i+batch_size]
        logger.info(f"Embedding batch {i//batch_size + 1}/{len(candidate_texts)//batch_size + 1}...")
        batch_embs = embed_text(batch, is_query=False)
        real_embeddings.append(batch_embs)
    print("[TRACE] Finished batch embedding loop")
        
    real_embeddings = np.vstack(real_embeddings)
    embedding_dim = real_embeddings.shape[1] # should be 1024
    
    # Combine with dummy embeddings if needed
    if limit < num_candidates:
        logger.info(f"Padding remaining {num_candidates - limit} candidates with placeholder embeddings...")
        dummy_embeddings = np.zeros((num_candidates - limit, embedding_dim), dtype=np.float32)
        all_embeddings = np.vstack([real_embeddings, dummy_embeddings]).astype(np.float32)
    else:
        all_embeddings = real_embeddings.astype(np.float32)
        
    # Save candidate embeddings
    np.save(config.ARTIFACTS_DIR / "candidate_embeddings.npy", all_embeddings)
    logger.info("Saved candidate_embeddings.npy")
    print("[TRACE] Saved candidate embeddings to disk")
    
    # 3. Create FAISS IndexFlatIP
    logger.info("Step 3: Creating FAISS IndexFlatIP...")
    index = faiss.IndexFlatIP(embedding_dim)
    index.add(all_embeddings)
    
    faiss.write_index(index, str(config.ARTIFACTS_DIR / "faiss_index.bin"))
    logger.info("Saved faiss_index.bin")
    print("[TRACE] Saved FAISS index to disk")
    
    # Save candidate ID mapping
    candidate_id_map = {i: c["candidate_id"] for i, c in enumerate(candidates)}
    with open(config.ARTIFACTS_DIR / "candidate_id_map.json", "w", encoding="utf-8") as f:
        json.dump(candidate_id_map, f)
    logger.info("Saved candidate_id_map.json")
    print("[TRACE] Saved candidate ID map to disk")

    # 4. Build Skill Ontology
    logger.info("Step 4: Building Skill Ontology...")
    all_skills = []
    for c in candidates[:limit]:
        for s in c.get("skills", []):
            if s.get("name"):
                all_skills.append(s["name"])
                
    ontology = SkillOntology(config.ARTIFACTS_DIR / "skill_ontology.json")
    ontology.build_ontology(all_skills)
    print("[TRACE] Finished building skill ontology")

    # 5. Retrieve Reduced Pool and Run Feature Extraction
    logger.info("Step 5: Retrieving top candidates via FAISS for LLM pre-computation...")
    # Retrieve top K candidates
    top_n = min(num_candidates, config.REDUCED_POOL_SIZE)
    distances, indices = index.search(jd_embedding.reshape(1, -1), top_n)
    
    retrieved_indices = indices[0]
    retrieved_candidates = [candidates[idx] for idx in retrieved_indices if idx < limit] # Only pull from embedded ones
    
    logger.info(f"Running feature extraction on {len(retrieved_candidates)} candidates...")
    extracted_features = run_extraction(retrieved_candidates)
    
    # Save features and narratives
    with open(config.ARTIFACTS_DIR / "llm_features.jsonl", "w", encoding="utf-8") as f:
        for feat in extracted_features:
            f.write(json.dumps(feat) + "\n")
    logger.info("Saved llm_features.jsonl")
    print("[TRACE] Saved LLM features to disk")

    # Cache CrossEncoder locally
    logger.info("Step 6: Pre-caching CrossEncoder model...")
    save_cross_encoder_locally()
    print("[TRACE] Finished pre-caching CrossEncoder")
    
    # Save a manifest file
    manifest = {
        "num_candidates": num_candidates,
        "embedding_dimension": embedding_dim,
        "real_embeddings_count": limit,
        "reduced_pool_size": len(retrieved_candidates)
    }
    with open(config.ARTIFACTS_DIR / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    logger.info("Saved manifest.json. Phase 1 setup complete!")
    print("[TRACE] Saved manifest.json and completed setup!")

if __name__ == '__main__':
    main()
