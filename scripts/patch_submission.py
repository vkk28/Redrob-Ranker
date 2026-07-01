import pandas as pd
import json
from pathlib import Path
from src.honeypot_detection import is_honeypot, is_consulting_only
from src.hard_gates import apply_gates
from src.llm_feature_extraction import local_extract_features

SUBMISSION_CSV = Path("submission.csv")
CANDIDATES_JSONL = Path("candidates.jsonl")

def main():
    if not SUBMISSION_CSV.exists() or not CANDIDATES_JSONL.exists():
        print("Missing required files.")
        return

    # 1. Load current submission
    df = pd.read_csv(SUBMISSION_CSV)
    print(f"Original submission size: {len(df)}")
    
    # 2. Load all profiles from candidates.jsonl to perform dynamic gating checks
    print("Loading candidate profiles to perform dynamic gating checks...")
    all_profiles = {}
    with open(CANDIDATES_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            cand = json.loads(line)
            all_profiles[cand["candidate_id"]] = cand

    # 3. Dynamically filter candidates from the submission who fail the gate logic
    filtered_rows = []
    disqualified_ids = set()
    for idx, row in df.iterrows():
        cid = row["candidate_id"]
        cand = all_profiles.get(cid)
        if not cand:
            continue
            
        profile = cand.get("profile", {})
        yoe = profile.get("years_of_experience", 0)
        
        # Check gates
        if is_honeypot(cand) or is_consulting_only(cand) or yoe < 3.0:
            print(f"Dynamically disqualified candidate: {cid} | Title: {profile.get('current_title')} | YOE: {yoe}")
            disqualified_ids.add(cid)
            continue
            
        filtered_rows.append(row)
        
    filtered_df = pd.DataFrame(filtered_rows)
    print(f"Filtered submission size: {len(filtered_df)}")
    
    current_ids = set(filtered_df["candidate_id"].tolist())
    needed = 100 - len(filtered_df)
    
    # 4. Scan candidates.jsonl for replacements if needed
    replacements = []
    search_keywords = {"faiss", "milvus", "qdrant", "pinecone", "weaviate", "elasticsearch", "opensearch", "rag", "retrieval", "embeddings"}
    
    if needed > 0:
        print(f"Need to find {needed} replacement candidates. Scanning candidates.jsonl...")
        with open(CANDIDATES_JSONL, "r", encoding="utf-8") as f:
            for line in f:
                if len(replacements) >= needed:
                    break
                if not line.strip():
                    continue
                cand = json.loads(line)
                cid = cand["candidate_id"]
                
                # Skip if already in top 100 or previously disqualified
                if cid in current_ids or cid in disqualified_ids:
                    continue
                    
                # Apply hard gates
                if is_honeypot(cand) or is_consulting_only(cand):
                    continue
                    
                profile = cand.get("profile", {})
                yoe = profile.get("years_of_experience", 0)
                if yoe < 3.0:
                    continue
                    
                # Filter for ML/Search relevance
                skills = {s.get("name", "").lower() for s in cand.get("skills", [])}
                if not (skills & search_keywords):
                    continue
                    
                title = profile.get("current_title", "").lower()
                if not any(t in title for t in ["engineer", "scientist", "analyst"]):
                    continue
                    
                replacements.append({
                    "candidate_id": cid,
                    "score": 0.25, # placeholder, will be sorted/calibrated later
                    "reasoning": ""
                })
        
        new_rows = pd.DataFrame(replacements)
        final_df = pd.concat([filtered_df, new_rows], ignore_index=True)
    else:
        final_df = filtered_df.copy()

    # 5. Regenerate reasoning for ALL 100 final candidates to apply new narrative logic
    print("Regenerating narratives for all 100 candidates...")
    for idx, row in final_df.iterrows():
        cid = row["candidate_id"]
        cand = all_profiles.get(cid)
        if cand:
            feats = local_extract_features(cand)
            final_df.at[idx, "reasoning"] = feats["fit_narrative"]
            
            # Make sure score is scaled reasonably
            profile = cand.get("profile", {})
            yoe = profile.get("years_of_experience", 0)
            skills = {s.get("name", "").lower() for s in cand.get("skills", [])}
            # Adjust score if it is a new replacement (older ones keep their score)
            if cid in [r["candidate_id"] for r in replacements]:
                final_df.at[idx, "score"] = round(0.25 + (yoe % 5) * 0.01 + (len(skills & search_keywords) * 0.005), 4)

    # 6. Sort and format
    final_df = final_df.sort_values(by=["score", "candidate_id"], ascending=[False, True]).reset_index(drop=True)
    final_df["rank"] = final_df.index + 1
    final_df = final_df[["candidate_id", "rank", "score", "reasoning"]]
    
    # 7. Save CSV and XLSX
    final_df.to_csv(SUBMISSION_CSV, index=False)
    final_df.to_excel("submission.xlsx", index=False)
    print("Successfully updated reasoning narratives, saved submission.csv and regenerated submission.xlsx!")

if __name__ == "__main__":
    main()
