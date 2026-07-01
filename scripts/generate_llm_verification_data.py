import pandas as pd
import json
from pathlib import Path

# Paths
SUBMISSION_CSV = Path("submission.csv")
CANDIDATES_JSONL = Path("candidates.jsonl")
OUTPUT_MD = Path("llm_verification_input.md")

def main():
    if not SUBMISSION_CSV.exists():
        print(f"Error: {SUBMISSION_CSV} does not exist. Please run rank.py first.")
        return
    if not CANDIDATES_JSONL.exists():
        print(f"Error: {CANDIDATES_JSONL} does not exist.")
        return

    # Read top 100 candidate IDs
    df = pd.read_csv(SUBMISSION_CSV)
    top_ids = set(df["candidate_id"].tolist())
    rank_map = {row["candidate_id"]: row["rank"] for _, row in df.iterrows()}
    reason_map = {row["candidate_id"]: row["reasoning"] for _, row in df.iterrows()}

    # Stream candidates and extract profiles
    extracted = {}
    print("Streaming candidates.jsonl to extract top 100 profiles...")
    with open(CANDIDATES_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            cand = json.loads(line)
            cid = cand["candidate_id"]
            if cid in top_ids:
                extracted[cid] = cand

    # Write a beautifully formatted Markdown report
    print(f"Writing profiles to {OUTPUT_MD}...")
    with open(OUTPUT_MD, "w", encoding="utf-8") as f:
        f.write("# Top 100 Candidates Profiles\n\n")
        f.write("Evaluate the following candidate profiles against the Job Description. Check for: YOE, consulting-only work history, specific NLP skills, and profile completeness.\n\n")
        
        # Output in ranked order
        ordered_ids = sorted(df["candidate_id"].tolist(), key=lambda x: rank_map[x])
        for cid in ordered_ids:
            cand = extracted.get(cid)
            if not cand:
                continue
            rank = rank_map[cid]
            reason = reason_map[cid]
            profile = cand.get("profile", {})
            skills = ", ".join([s.get("name", "") for s in cand.get("skills", []) if s.get("name")])
            
            f.write(f"## Rank {rank}: Candidate {cid}\n")
            f.write(f"- **Current Role**: {profile.get('current_title')} at {profile.get('current_company')}\n")
            f.write(f"- **Years of Experience**: {profile.get('years_of_experience')} YOE\n")
            f.write(f"- **Skills**: {skills}\n")
            f.write(f"- **Generated Reasoning**: *\"{reason}\"*\n\n")
            f.write("### Career History:\n")
            for role in cand.get("career_history", []):
                f.write(f"- **{role.get('title')}** at {role.get('company')} ({role.get('start_date')} to {role.get('end_date') or 'Present'})\n")
                if role.get("description"):
                    f.write(f"  *Description*: {role.get('description')}\n")
            f.write("\n---\n\n")

    print(f"Successfully generated {OUTPUT_MD} containing the top 100 profiles.")

if __name__ == "__main__":
    main()
