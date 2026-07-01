import json
import os
import sys
from pathlib import Path
import numpy as np

# Add base directory to path so we can import src
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.utils import get_logger
from src.data_loader import stream_candidates
from src.skill_ontology import SkillOntology
from src.llm_feature_extraction import run_extraction
from src.cross_encoder_rerank import save_cross_encoder_locally
import src.config as config

logger = get_logger("finish_setup")

def main():
    logger.info("Loading candidates...")
    candidates = list(stream_candidates(config.CANDIDATES_JSONL, validate=False))
    num_candidates = len(candidates)
    logger.info(f"Loaded {num_candidates} candidates.")
    
    # Load retrieved indices
    logger.info("Loading retrieved indices from FAISS search...")
    with open(config.ARTIFACTS_DIR / "retrieved_indices.json", "r") as f:
        retrieved_indices = json.load(f)
    
    retrieved_candidates = [candidates[idx] for idx in retrieved_indices if idx < len(candidates)]
    logger.info(f"Retrieved {len(retrieved_candidates)} candidates.")
    
    # 1. Build Skill Ontology
    logger.info("Building Skill Ontology...")
    all_skills = []
    for c in candidates:
        for s in c.get("skills", []):
            if s.get("name"):
                all_skills.append(s["name"])
    ontology = SkillOntology(config.ARTIFACTS_DIR / "skill_ontology.json")
    ontology.build_ontology(all_skills)
    logger.info("Saved skill_ontology.json")
    
    # 2. Save Candidate ID Mapping
    candidate_id_map = {i: c["candidate_id"] for i, c in enumerate(candidates)}
    with open(config.ARTIFACTS_DIR / "candidate_id_map.json", "w", encoding="utf-8") as f:
        json.dump(candidate_id_map, f)
    logger.info("Saved candidate_id_map.json")
    
    # 3. Run Feature Extraction
    logger.info(f"Running feature extraction on {len(retrieved_candidates)} candidates...")
    extracted_features = run_extraction(retrieved_candidates)
    
    # Save features and narratives
    with open(config.ARTIFACTS_DIR / "llm_features.jsonl", "w", encoding="utf-8") as f:
        for feat in extracted_features:
            f.write(json.dumps(feat) + "\n")
    logger.info("Saved llm_features.jsonl")
    
    # 4. Cache CrossEncoder locally
    logger.info("Pre-caching CrossEncoder...")
    save_cross_encoder_locally()
    
    # 5. Save manifest
    manifest = {
        "num_candidates": num_candidates,
        "embedding_dimension": 384,
        "real_embeddings_count": num_candidates,
        "reduced_pool_size": len(retrieved_candidates)
    }
    with open(config.ARTIFACTS_DIR / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    logger.info("Saved manifest.json. Setup complete!")

if __name__ == '__main__':
    main()
