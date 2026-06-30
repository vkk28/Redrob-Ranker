import json
from pathlib import Path
from src.hard_gates import apply_gates
from src.llm_feature_extraction import local_extract_features
from src.scoring import score_candidate
import src.config as config

def calibrate():
    synthetic_file = Path("/Users/varunkashyap/Desktop/Redrob-Ranker/data/synthetic/synthetic_candidates.json")
    if not synthetic_file.exists():
        print(f"Synthetic file not found at {synthetic_file}")
        return
        
    with open(synthetic_file, "r", encoding="utf-8") as f:
        candidates = json.load(f)
        
    print(f"Scoring {len(candidates)} synthetic candidates for calibration:\n")
    print("| Candidate ID | Name | Current Title | Gate Multiplier | Base Score | Final Score | Fit Narrative |")
    print("|---|---|---|---|---|---|---|")
    
    for cand in candidates:
        # Step 1: LLM Features Extraction (local fallback)
        features = local_extract_features(cand)
        
        # Step 2: Gating
        gate_mult, reason = apply_gates(cand)
        
        # Step 3: Scoring
        scored = score_candidate(cand, features, gate_mult)
        
        print(f"| {scored['candidate_id']} | {scored['profile']['anonymized_name']} | {scored['profile']['current_title']} | {scored['_gate_multiplier']} ({reason}) | {scored['base_score']} | {scored['final_score']} | {scored['fit_narrative'][:60]}... |")
        
    # Save default calibration weights
    weights = {
        "core_fit_weight": config.CORE_FIT_WEIGHT,
        "logistics_weight": config.LOGISTICS_WEIGHT,
        "education_weight": config.EDUCATION_WEIGHT,
        "description": "Calibrated weights for Redrob founding engineer ranking"
    }
    
    config.CALIBRATION_WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(config.CALIBRATION_WEIGHTS_PATH, "w", encoding="utf-8") as f:
        json.dump(weights, f, indent=2)
    print(f"\nSaved calibration weights to {config.CALIBRATION_WEIGHTS_PATH}")

if __name__ == '__main__':
    calibrate()
