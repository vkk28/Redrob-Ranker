# Redrob Hackathon — Implementation Plan
**Intelligent Candidate Discovery & Ranking — Build Spec for AI-Assisted Development**

This document is a step-by-step execution plan, written to be handed to an AI coding IDE (Claude Code, Cursor, etc.) and worked through top to bottom. It assumes zero existing code. Each phase has a goal, concrete deliverables, and acceptance criteria so the agent (or you) can verify progress before moving on.

---

## Task 0 — Verify source-of-truth files before writing any code

**Do this first, before anything else.** Two prior planning documents exist for this project (a "Problem Statement" summary and a "Strategy v3.5" doc). Both predate this README and may not exactly match the real bundle. The README explicitly lists the authoritative files:

| Authoritative file (from README) | Format | Notes |
|---|---|---|
| `candidates.jsonl.gz` | gzipped JSONL | ~52 MB compressed / ~465 MB uncompressed, 100,000 rows |
| `sample_candidates.json` | JSON | First 50 candidates, pretty-printed |
| `job_description.md` | Markdown | **Has a section at the end specifically for hackathon participants** — read it fully, it may contain rules not in the problem-statement summary |
| `submission_spec.md` | Markdown | Full rules — treat as ground truth over any prior summary |
| `redrob_signals_doc.md` | Markdown | Signal definitions + trap descriptions |
| `candidate_schema.json` | JSON Schema (draft-07) | Field-level constraints |
| `submission_metadata_template.yaml` | YAML | Fill and commit at repo root |
| `sample_submission.csv` | CSV | Format only, not a quality reference |
| `validate_submission.py` | Python | Run before every submission |

**Action items:**
1. Open `job_description.md` and read the participant-specific section at the end in full — this is new information not summarized anywhere yet.
2. Diff `submission_spec.md` against the assumed scoring formula (`0.50×NDCG@10 + 0.30×NDCG@50 + 0.15×MAP + 0.05×P@10`) and the deck/PDF requirement (Section 10e in the problem-statement summary) — confirm both are still accurate.
3. Confirm `validate_submission.py` actually accepts `.jsonl.gz` as the README claims, and check what exit codes / messages it gives on failure, so `rank.py` and CI can fail fast on the same checks pre-submission.
4. Note: the README says "Submissions with honeypot rate > 10% in top 100 are disqualified" — confirm this is identical to the problem-statement's Stage 3 rule before relying on both as one spec.

Do not proceed past Task 0 silently — if anything in this plan conflicts with the real `.md` files once read, the `.md` files win.

---

## 1. Objective Recap

Rank 100,000 candidates against one fixed JD (Senior AI Engineer — Founding Team, Redrob AI), output top 100 as CSV with `candidate_id, rank, score, reasoning`. Hard constraints on the **ranking step only**:

- ≤ 5 minutes wall-clock
- ≤ 16 GB RAM
- CPU only, no GPU
- No network calls during ranking
- Honeypot rate in top-100 must be ≤ 10% (auto-disqualification at Stage 3 if exceeded, regardless of score)
- 3 submissions max, last valid one counts, no live feedback

Pre-computation (embeddings, indexes, LLM-derived features) is allowed offline, before the 5-minute window, but must be reproducible and documented.

---

## 2. Architecture: 2-Phase Retrieve-and-Rerank

```
PHASE 1 (offline, network allowed, run once, artifacts committed to repo)
  JD parsing → embeddings → FAISS index → skill ontology →
  LLM feature extraction (on a reduced candidate set, not all 100K) →
  narrative generation + verification → calibration

PHASE 2 (rank.py, ≤5 min, no network, must run inside the Stage 3 sandbox unmodified)
  FAISS retrieve top-N → hard gates (honeypot/consulting-only/title/domain) →
  core fit scoring → behavioral/logistics scoring → cross-encoder rerank top-K →
  Tier-5 finder pass → dedup-safe top-100 assembly → reasoning assembly → CSV
```

The key design discipline: **Phase 2 must do zero LLM calls and zero network calls.** Everything expensive happens in Phase 1 and is cached to disk as versioned artifacts.

---

## 3. Repository Structure

```
redrob-ranker/
├── README.md                       # setup + single reproduce command
├── requirements.txt
├── submission_metadata.yaml        # filled from template, repo root
├── data/
│   ├── raw/                        # candidates.jsonl.gz, sample_candidates.json, job_description.md, etc. (gitignored)
│   └── synthetic/                  # hand-built edge-case profiles for calibration
├── artifacts/                      # PRE-COMPUTED, committed (Git LFS for large files)
│   ├── manifest.json               # checksums + generation timestamp for every artifact
│   ├── jd_embedding.npy
│   ├── candidate_embeddings.npy
│   ├── candidate_id_map.json
│   ├── faiss_index.bin
│   ├── skill_ontology.json
│   ├── llm_features.jsonl          # only for the reduced candidate set, see §6.4
│   ├── fit_narratives.jsonl        # pre-verified reasoning text + narrative embeddings
│   └── calibration_weights.json
├── src/
│   ├── config.py                   # all weights/thresholds in one place, never hardcoded inline
│   ├── schema.py                   # pydantic models mirroring candidate_schema.json
│   ├── jd_parser.py
│   ├── data_loader.py              # handles both .jsonl and .jsonl.gz
│   ├── embeddings.py
│   ├── skill_ontology.py
│   ├── llm_feature_extraction.py
│   ├── narrative_generation.py
│   ├── honeypot_detection.py
│   ├── hard_gates.py
│   ├── scoring.py
│   ├── cross_encoder_rerank.py
│   ├── tier5_finder.py
│   ├── assembly.py                 # dedup-safe top-100 assembly, see §9
│   └── utils.py
├── scripts/
│   ├── setup_artifacts.py          # Phase 1 orchestrator — network allowed, run offline
│   ├── rank.py                     # Phase 2 entrypoint — NO network, this is the Stage 3 reproduce command
│   ├── calibrate.py                # Phase 1.5 — sample + synthetic profile tuning
│   ├── benchmark_compute.py        # measures wall-clock + peak RAM on the real 100K file
│   └── make_synthetic_profiles.py
├── tests/
│   ├── test_honeypot_detection.py
│   ├── test_dedup_assembly.py
│   ├── test_validator_compliance.py
│   ├── test_schema_compliance.py
│   ├── test_no_network_in_rank.py  # static check: rank.py imports no networking libs
│   └── fixtures/
├── docker/
│   └── Dockerfile                  # mirrors the Stage 3 sandbox: CPU-only, 16GB cap
└── docs/
    ├── methodology_summary.md      # ≤200 words, for portal submission
    └── deck_source/                # source material for the required deck/PDF
```

---

## 4. Environment & Dependencies

```
sentence-transformers     # BAAI/bge-large-en-v1.5 for embeddings
faiss-cpu                 # HNSW index, CPU only — confirm faiss-cpu not faiss-gpu in requirements.txt
hdbscan                   # skill ontology clustering
scikit-learn
numpy, pandas
pydantic                  # schema validation
anthropic OR openai       # Phase 1 only — must NOT appear as an import in rank.py
pytest
```

Pin every version in `requirements.txt`. The Stage 3 sandbox rebuilds your environment from this file — unpinned versions are a silent reproducibility risk.

**Acceptance check:** `grep -r "anthropic\|openai\|requests\|httpx" src/ scripts/rank.py` should return nothing. Add this as `tests/test_no_network_in_rank.py` so it's enforced automatically, not just checked manually once.

---

## 5. Phase 0 — Data Ingestion & Schema Validation

**Goal:** get from raw bundle files to a clean, typed, in-memory (or memory-mapped) candidate pool, and confirm the schema assumptions before building anything on top of them.

1. `data_loader.py`: load `candidates.jsonl.gz` (and fall back to plain `.jsonl` if present) into a list/iterator of dicts. Given ~465 MB uncompressed, prefer streaming/chunked processing over loading everything as Python objects at once where possible in Phase 1; for Phase 2 ranking you likely do need the full pool in memory or memory-mapped, since gates and scoring touch every candidate.
2. Validate every record against `candidate_schema.json` using `pydantic` or `jsonschema`. Log (don't crash on) any records that fail validation — the dataset may intentionally include edge cases.
3. Parse `job_description.md` programmatically (`jd_parser.py`): extract must-haves, nice-to-haves, explicit disqualifiers, and the "ideal profile" text as separate structured fields, not just one blob. You'll reference these independently in gating and scoring.
4. Read `redrob_signals_doc.md` in full and write down the exact definition of each of the 23 signals in `src/config.py` as named constants/comments — don't re-derive signal semantics from field names alone, since several (e.g. `offer_acceptance_rate` ranging into `-1`, `github_activity_score` ranging into `-1`) have special sentinel values that need explicit handling.

**Acceptance:** `python scripts/benchmark_compute.py --stage load` reports load time and peak RAM for the full 100K file before you build anything else on top of it. This number feeds directly into your 5-minute budget — measure it now, not at the end.

---

## 6. Phase 1 — Offline Pre-Computation

### 6.1 JD Embedding
Embed the JD (and a few paraphrased variants of it, if you want retrieval robustness) with `BAAI/bge-large-en-v1.5`. Save as `artifacts/jd_embedding.npy`.

### 6.2 Candidate Embeddings + FAISS Index
Embed all 100,000 candidates (concatenate headline + summary + top career history descriptions + skills list into one text per candidate — document and version this concatenation recipe in `embeddings.py`, since changing it later invalidates the index). Build a FAISS HNSW index. Save both `candidate_embeddings.npy` and `faiss_index.bin`, plus `candidate_id_map.json` mapping FAISS internal indices back to `candidate_id`.

### 6.3 Skill Ontology
Embed all unique skill name strings across the dataset, cluster with HDBSCAN, map variants (e.g. "Vector DB", "vector database", "Vector Databases") to canonical IDs. Save `skill_ontology.json`. This directly supports the JD's must-have skill matching without being defeated by string-level keyword mismatches — and it's also how you catch keyword-stuffer honeypots (skills with no canonical-cluster support in career history).

### 6.4 LLM Feature Extraction — scope this to a reduced set, not all 100K
Running LLM extraction over all 100,000 candidates is expensive and slow to complete, and the JD is fixed, so most of the pool will never plausibly reach the top 100 anyway. Recommended approach:

1. Run FAISS retrieval (top ~3,000–5,000) against the JD embedding **offline, in Phase 1** — this is deterministic given a fixed JD and fixed embedding model, so it's safe to precompute.
2. Run LLM feature extraction and narrative generation **only on this reduced pool**, saved to `artifacts/llm_features.jsonl` and `artifacts/fit_narratives.jsonl`, keyed by `candidate_id`.
3. In Phase 2, `rank.py` does its own (fast, local) semantic retrieval over the full 100K — but for any candidate that also appears in the Phase 1 reduced pool, it loads pre-computed LLM features and narrative instead of recomputing. Candidates outside the reduced pool that surface in Phase 2 (rare, since both retrievals use the same JD embedding) fall back to the deterministic, non-LLM scoring path described in §10.

This cuts LLM call volume by roughly 20-30x with negligible loss of top-100 coverage, since anything outside FAISS top-3-5K is already extremely unlikely to belong in the final 100.

**Important:** commit the resulting artifacts to the repo (Git LFS if needed). Do **not** have `rank.py` regenerate them on a cache-miss — if `setup_artifacts.py` has a "fall back to building from scratch" path that calls live LLM APIs, make sure that path is never reachable from `rank.py`, and ideally isn't reachable during Stage 3 at all. A live LLM call during Stage 3 reproduction risks both non-determinism and exceeding the time budget — either one is an automatic disqualification, independent of your composite score.

### 6.5 Narrative Generation & Verification
For each candidate in the reduced pool, generate a `fit_narrative` via LLM, then verify every factual claim in it against the candidate's actual structured data and raw career-history text before accepting it.

```python
# src/narrative_generation.py — offline_verify_narrative

def offline_verify_narrative(candidate, narrative, skill_ontology, max_attempts=3):
    actual_skills = {s["name"].lower() for s in candidate["skills"]}
    history_text = " ".join(
        r.get("description", "") for r in candidate["career_history"]
    ).lower()
    all_companies = {candidate["profile"]["current_company"].lower()} | {
        r["company"].lower() for r in candidate["career_history"]
    }
    actual_yoe = candidate["profile"]["years_of_experience"]
    actual_loc = candidate["profile"]["location"].lower()

    for attempt in range(max_attempts):
        claims = extract_claims(narrative, skill_ontology)
        unsupported = []
        for claim in claims:
            if claim.type == "skill":
                # Skill claims sourced from free-text career history count as supported —
                # don't require the skill to also appear in the structured skills[] array.
                if claim.value not in actual_skills and claim.value not in history_text:
                    unsupported.append(claim)
            elif claim.type == "company" and claim.value not in all_companies:
                unsupported.append(claim)
            elif claim.type == "yoe" and claim.value != actual_yoe:
                unsupported.append(claim)
            elif claim.type == "location" and claim.value not in actual_loc:
                unsupported.append(claim)

        if not unsupported:
            return narrative, "llm_verified"

        if attempt < max_attempts - 1:
            narrative = regenerate_reasoning_with_llm(candidate, unsupported)

    # Deterministic fallback — see §11 for why this template must stay non-generic
    return build_fallback_reasoning(candidate), "fallback_template"
```

Track the `"llm_verified"` vs `"fallback_template"` ratio across the reduced pool and log it — you want visibility into how often the fallback fires (see §11).

### 6.6 Phase 1.5 — Calibration
Before running anything on the full pool, validate scoring logic on data you can manually sanity-check:

1. Run all 50 `sample_candidates.json` profiles through the Phase 2 scoring logic. Manually review the resulting ranking against your own read of the JD — does it make intuitive sense?
2. Hand-construct 8–12 synthetic edge-case profiles covering:
   - A true Tier-5 candidate (no AI buzzwords, but a built/shipped ranking or recommendation system) — tune the narrative-similarity threshold against this.
   - A keyword-stuffer (e.g. "Marketing Manager" with "expert" Pinecone/RAG/FAISS skills, zero usage months) — confirm this gets gated to near-zero, not just penalized.
   - A honeypot with internally inconsistent dates (YOE longer than employer's founding age; "expert" skill with 0 `duration_months`) — confirm `is_honeypot()` catches it.
   - A WITCH-tier-only consulting career (TCS/Infosys/Wipro/Accenture/Cognizant/Capgemini, no product company) — confirm `is_consulting_only()` zeroes it.
   - A pure-research-no-production candidate.
   - A recent (<12 month) LangChain-wrapper-only candidate with no prior ML production history.
   - A senior candidate with no production code commits in 18+ months.
   - A CV/Speech/Robotics specialist with no NLP/IR exposure.
   - A serial job-hopper (every ~1.5 years).
   - A borderline double-soft-gate case (adjacent title *and* adjacent domain simultaneously) — tune the combined-penalty floor.
3. Save tuned weights/thresholds to `artifacts/calibration_weights.json`, loaded by `config.py` — never hardcode magic numbers directly in scoring code.

**Acceptance:** every synthetic edge case produces the expected outcome (gated to ~0, penalized-but-included, or scored normally) before moving to Phase 2 implementation.

---

## 7. Phase 2 — Ranking Step (`rank.py`, ≤ 5 min, no network)

```
Stage 1: Load FAISS index + candidate pool + artifacts
Stage 2: Semantic retrieval — FAISS top-N vs JD embedding
Stage 3: Hard gates — honeypot / consulting-only / title / domain
Stage 4: Core fit scoring (production evidence, retrieval/eval-framework match)
Stage 5: Behavioral + logistics scoring (redrob_signals)
Stage 6: Cross-encoder rerank — top-K only (expensive step, bound K carefully)
Stage 7: Tier-5 finder pass — capped semantic match over the gated pool
Stage 8: Dedup-safe top-100 assembly + score reconciliation
Stage 9: Reasoning assembly (pre-verified narrative + rank-aware notes)
Stage 10: Write CSV, run validate_submission.py inline, fail loudly on any violation
```

`rank.py` must take exactly the CLI form specified in the submission spec:
```
python rank.py --candidates ./candidates.jsonl --out ./submission.csv
```
Confirm against `submission_spec.md` whether the grader will pass `.jsonl` or `.jsonl.gz` — handle both regardless, since the README mentions the validator accepts either.

---

## 8. Honeypot & Disqualifier Detection — this is the highest-stakes function in the project

`>10% honeypots in top-100` is an automatic disqualification at Stage 3 **regardless of composite score** — there's no partial credit, and no live feedback loop to discover a miss before final submission. Treat this function as the single most important piece of code in the repo and over-invest in test coverage relative to its share of the scoring formula.

```python
# src/honeypot_detection.py

def is_honeypot(candidate, current_year=2026):
    profile = candidate["profile"]
    history = candidate["career_history"]

    # 1. Tenure predates employer existence — requires a company-founding-year
    #    reference; if unavailable in the dataset, infer a floor from the
    #    earliest start_date across ALL candidates at that company as a proxy,
    #    and flag (don't silently pass) any case you can't verify confidently.
    for role in history:
        if role.get("start_date") and role.get("end_date"):
            if role["start_date"] > role["end_date"]:
                return True  # impossible date ordering

    # 2. "Expert" proficiency with ~0 usage duration
    for skill in candidate["skills"]:
        if skill["proficiency"] == "expert" and skill.get("duration_months", 0) <= 1:
            return True

    # 3. Years of experience inconsistent with sum/span of career_history
    total_history_months = sum(r.get("duration_months", 0) for r in history)
    claimed_yoe_months = profile["years_of_experience"] * 12
    if total_history_months > 0 and claimed_yoe_months < total_history_months * 0.5:
        return True  # claims far less experience than history implies, or vice versa
    if claimed_yoe_months > 0 and total_history_months > claimed_yoe_months * 1.5:
        return True

    # 4. Overlapping concurrent roles that aren't plausibly part-time/advisory
    sorted_roles = sorted(
        [r for r in history if r.get("start_date") and r.get("end_date")],
        key=lambda r: r["start_date"],
    )
    for a, b in zip(sorted_roles, sorted_roles[1:]):
        if a["end_date"] > b["start_date"]:
            return True  # significant overlap between two full-time-looking roles

    # 5. Education years that don't precede or align with career start
    for edu in candidate.get("education", []):
        if edu.get("end_year") and history:
            earliest_start = min(
                (r["start_date"][:4] for r in history if r.get("start_date")),
                default=None,
            )
            if earliest_start and int(edu["end_year"]) > int(earliest_start) + 1:
                return True  # started working before plausibly finishing this degree

    return False


def is_consulting_only(candidate):
    witch_tier = {
        "tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini"
    }
    companies = {candidate["profile"]["current_company"].lower()} | {
        r["company"].lower() for r in candidate["career_history"]
    }
    # Only zero out if EVERY employer is WITCH-tier — partial consulting
    # background with later product-company experience should NOT be gated.
    return companies.issubset(witch_tier) and len(companies) > 0
```

**Do not treat this as a one-and-done implementation.** Required follow-up work:
- Write `tests/test_honeypot_detection.py` covering every pattern named in the problem statement (impossible tenure, 0-duration "expert" skills, inconsistent dates) **plus** patterns it doesn't explicitly name (overlapping employment, education/career sequencing, endorsement counts wildly disproportionate to connection count, salary expectations inconsistent with stated experience tier).
- Run the full `is_honeypot()` pass over the entire 100K pool and manually spot-check a sample of both flagged and unflagged candidates — false negatives are the disqualification risk, but false positives quietly cost you good candidates, so check both directions.
- Cross-reference your honeypot rate against the stated "~80 honeypot candidates" in the dataset — if your detector flags a wildly different count (e.g. 5 or 500), that's a signal something is mis-calibrated before you ever see Stage 3.

---

## 9. Top-100 Assembly — dedup-safe (corrects a real bug in the v3.5 draft)

The earlier draft appended Tier-5 force-includes onto the cross-encoder top-95 without checking for overlap. A genuinely strong Tier-5 candidate (good career history, no buzzwords) can legitimately also rank in the top-95 by embedding/cross-encoder similarity alone — in which case the naive append produces a duplicate `candidate_id`, which fails `validate_submission.py`'s uniqueness check and is an automatic Stage 1 rejection. Given the 3-submission cap, this needs to be airtight before any submission is made.

```python
# src/assembly.py

def assemble_final_csv(top_200_fused, tier_5_force_includes, target_size=100, tier5_slot=5):
    top95_ids = {c["candidate_id"] for c in top_200_fused[:target_size - tier5_slot]}

    tier_5_unique = [
        c for c in tier_5_force_includes if c["candidate_id"] not in top95_ids
    ]

    final_100 = list(top_200_fused[: target_size - tier5_slot]) + tier_5_unique

    # Backfill if dedup left us short of 100 — pull next-best from top_200_fused
    # that isn't already included, in score order.
    if len(final_100) < target_size:
        included_ids = {c["candidate_id"] for c in final_100}
        backfill_pool = iter(top_200_fused[target_size - tier5_slot:])
        while len(final_100) < target_size:
            candidate = next(backfill_pool, None)
            if candidate is None:
                break  # exhausted top_200_fused — widen the upstream candidate pool
            if candidate["candidate_id"] not in included_ids:
                final_100.append(candidate)
                included_ids.add(candidate["candidate_id"])

    assert len(final_100) == target_size, (
        f"Assembly produced {len(final_100)} rows, expected {target_size} — "
        "widen top_200_fused or tier5 candidate pool."
    )
    assert len({c["candidate_id"] for c in final_100}) == target_size, "duplicate ids in final assembly"

    # Score reconciliation — keep monotonicity without burying genuinely high-scoring Tier-5s
    floor_score = final_100[target_size - tier5_slot - 1]["final_ranked_score"]
    for i, c in enumerate(tier_5_unique):
        c["final_ranked_score"] = floor_score - (0.001 * (i + 1))

    final_100.sort(key=lambda c: c["final_ranked_score"], reverse=True)
    return final_100
```

**Required test:** `tests/test_dedup_assembly.py` should construct a fixture where one Tier-5 candidate is deliberately *also* present in the top-95, and assert the output has exactly 100 unique IDs and non-increasing scores. Run this before every submission, not just once.

---

## 10. Scoring — weights as a starting point, not a final answer

```python
# src/config.py — defaults, calibrate against §6.6 before locking

CORE_FIT_WEIGHT = 0.55      # production retrieval/ranking evidence, eval-framework depth
LOGISTICS_WEIGHT = 0.25     # location/relocation, notice period, salary band, work mode
EDUCATION_WEIGHT = 0.20     # institution tier — de-emphasized deliberately, see rationale below
```

**Rationale for shifting weight toward core fit:** the JD's own "ideal profile" section and its explicit framing ("the right answer is NOT to find candidates whose skills contain the most AI keywords... a Tier-5 candidate who built a recommendation system at a product company is a fit even if they don't mention 'RAG'") puts almost all its emphasis on production evidence and shipped systems, and never names institution tier as a differentiator. The original draft's 0.50/0.30/0.20 split gives logistics nearly as much weight as core fit, which risks a technically excellent candidate with, say, a 90-day notice period getting outranked by a weaker but more "compliant" one. Since NDCG@10 is 50% of the composite score, run this exact question as a sensitivity test against the 50 sample candidates before finalizing: **does shifting weight from logistics/education toward core_fit change who lands in the top 10, and does the new top 10 look more aligned with the JD's stated philosophy?**

```python
base_score = (
    CORE_FIT_WEIGHT * core_fit_score +
    LOGISTICS_WEIGHT * logistics_score +
    EDUCATION_WEIGHT * education_score
)
sharpened_behavioral = 0.60 + (0.40 * raw_behavioral_multiplier)
final_score = base_score * sharpened_behavioral * gate_penalty
final_ranked_score = 0.7 * final_score + 0.3 * cross_encoder_score  # top-200 only
```

Document the calibration result (before/after weight comparison on sample candidates) in `docs/methodology_summary.md` — this is also useful material for the Stage 5 technical interview, where you'll be asked to defend exactly this kind of design choice.

---

## 11. Reasoning Assembly — avoid Stage 4 template penalties

Stage 4 manual review explicitly penalizes "templated strings." If the fallback template (§6.5) fires on more than a handful of your visible top-100 rows, that's a real risk. Track the fallback rate from Phase 1 and treat anything above ~5–10% of the reduced pool as a signal to debug the verification loop rather than just accept the fallback rate.

Make the fallback itself less generic than a pure title+YOE+skills string:

```python
def build_fallback_reasoning(candidate):
    top_skills = sorted(
        candidate["skills"], key=lambda s: s.get("endorsements", 0), reverse=True
    )[:3]
    skill_names = ", ".join(s["name"] for s in top_skills)
    current_company = candidate["profile"]["current_company"]
    yoe = candidate["profile"]["years_of_experience"]
    title = candidate["profile"]["current_title"]
    return (
        f"{title} at {current_company}, {yoe} yrs experience. "
        f"Verified strongest skills: {skill_names}."
    )
```

```python
def assemble_reasoning(candidate, rank, pre_verified_narrative):
    base = pre_verified_narrative
    if rank > 80:
        if candidate["_logistics_score"] < 0.6:
            base += " Concern: logistics mismatch (location/salary/notice period)."
        elif candidate["_core_fit_score"] < 0.5:
            base += " Concern: marginal production evidence depth."
    return base
```

---

## 12. Testing & Validation Plan

| Test | Purpose |
|---|---|
| `test_schema_compliance.py` | Every loaded candidate matches `candidate_schema.json` |
| `test_honeypot_detection.py` | Every named trap pattern + several unnamed ones is caught |
| `test_dedup_assembly.py` | Tier-5/top-95 overlap produces no duplicate IDs, exactly 100 rows |
| `test_no_network_in_rank.py` | `rank.py` and its imports contain no networking calls |
| `test_validator_compliance.py` | Output CSV passes `validate_submission.py` with zero errors |

Run the full suite, then run `python validate_submission.py <your_csv>` directly as the final gate before every one of your 3 submissions.

---

## 13. Compute Benchmarking & Sandbox Reproduction

Do not trust an estimate based on the 50-sample file — benchmark against the real 100K pool inside an environment matching the Stage 3 sandbox exactly (CPU-only, 16 GB cap, no network during the timed portion):

```
docker/Dockerfile  → CPU-only base image, 16GB memory limit set at `docker run --memory=16g --cpus=<N>`
```

```
python scripts/benchmark_compute.py --candidates ./data/raw/candidates.jsonl.gz
# Reports: load time, peak RSS, per-stage wall-clock (retrieval, gates, scoring,
# cross-encoder, tier-5, assembly), total wall-clock
```

Target: total wall-clock with comfortable headroom under 300s (aim for under ~150s to absorb sandbox-hardware variance — your dev machine and the Stage 3 container are not guaranteed to have identical CPU performance). Re-run this benchmark after any change to embedding dimensionality, cross-encoder top-K, or candidate pool sizes at each stage — these are the levers most likely to blow the budget.

**Acceptance:** `docker run --network=none --memory=16g <image> python rank.py --candidates ./candidates.jsonl.gz --out ./submission.csv` completes successfully end-to-end, with `--network=none` actually enforced (not just assumed), before you consider Phase 2 done.

---

## 14. Submission Checklist (per the README + problem-statement spec — verify both against `submission_spec.md`)

- [ ] CSV: exactly 100 data rows + header, correct column order, unique ranks 1–100, unique candidate_ids, non-increasing scores, all IDs exist in the candidate pool, UTF-8, `.csv` extension, filename = registered team ID
- [ ] `validate_submission.py` passes with zero errors
- [ ] GitHub repo: README with single reproduce command, full source, all artifacts (or generation script) committed, `requirements.txt` pinned, `submission_metadata.yaml` at root
- [ ] Sandbox link: hosted environment runnable on a small sample (HF Spaces / Streamlit Cloud / Replit / Colab / Docker registry / Binder)
- [ ] Portal metadata: team info, repo URL, sandbox link, reproduce command, compute environment details, AI tools declaration (be specific and honest — Stage 5 checks this against your actual code), ≤200-word methodology summary, all required declarations checked
- [ ] Deck/PDF — confirm this requirement and its exact content list against the live `submission_spec.md`, not just the problem-statement summary

---

## 15. Three-Submission Rollout

1. **Submission 1 (stripped baseline):** FAISS retrieval + hard gates + core fit scoring + offline LLM features, no cross-encoder, no Tier-5 finder. Goal: lock in a valid, format-compliant, reasonably strong submission early, so a late-stage bug never costs you a clean entry.
2. **Submission 2 (full pipeline test):** add cross-encoder rerank + Tier-5 finder pass. Compare qualitatively against submission 1's top-10/top-50 (no live NDCG feedback exists, so this is a manual sanity check, not a metric comparison).
3. **Submission 3 (final tuning):** fully calibrated thresholds from synthetic edge-case tuning, verified reasoning assembly active, dedup-safe assembly confirmed via test suite, full benchmark re-run inside the Docker sandbox one final time before upload.

---

## 16. Risk Register

| Risk | Mitigation in this plan |
|---|---|
| Duplicate candidate_id from Tier-5/top-95 overlap → Stage 1 auto-reject | §9 dedup-safe assembly + dedicated test |
| Honeypot rate > 10% → Stage 3 auto-disqualify regardless of score | §8 expanded detection rules + full-pool spot-check |
| Runtime/RAM estimate based on sample data, not real 100K pool | §13 mandatory Docker benchmark on real file before each submission |
| Live LLM calls reachable during Stage 3 (no-network) reproduction | §6.4 scoped LLM extraction + committed artifacts, no runtime fallback to live calls from `rank.py` |
| Stage 4 templated-reasoning penalty | §11 richer fallback template + fallback-rate tracking |
| Weights (education/logistics 50%) may contradict JD's stated philosophy | §10 sensitivity testing against sample candidates before locking |
| File-format assumptions (`.docx`/plain jsonl) don't match real bundle (`.md`/gzipped) | Task 0 — verify against README's actual file list before coding |

---

## 17. Build Order (literal task sequence for the AI IDE)

1. Task 0: read `job_description.md`, `submission_spec.md`, `redrob_signals_doc.md` in full; reconcile any conflicts with this plan
2. Scaffold repo structure (§3), `requirements.txt`, `config.py`
3. `data_loader.py` + `schema.py`, validate against `candidate_schema.json`, run `benchmark_compute.py --stage load`
4. `jd_parser.py` — structured extraction of must-haves/nice-to-haves/disqualifiers/ideal-profile
5. `embeddings.py` + FAISS index build (§6.1–6.2)
6. `skill_ontology.py` (§6.3)
7. `honeypot_detection.py` + full test suite (§8) — build and test this before scoring logic depends on it
8. `hard_gates.py` (title/domain/consulting-only checks, calls into honeypot_detection)
9. Reduced-pool LLM feature extraction + `narrative_generation.py` with verification loop (§6.4–6.5)
10. `scoring.py` (core fit / logistics / education / behavioral) (§10)
11. `cross_encoder_rerank.py`, `tier5_finder.py`
12. `assembly.py` dedup-safe top-100 + test (§9)
13. Reasoning assembly with anti-template fallback (§11)
14. `scripts/calibrate.py` — run sample + synthetic edge cases (§6.6), lock `calibration_weights.json`
15. `scripts/rank.py` end-to-end wiring, CLI matching the spec exactly
16. `scripts/benchmark_compute.py` full run inside `docker/Dockerfile` with `--network=none --memory=16g`
17. Full test suite green, `validate_submission.py` clean run
18. Submission 1 (stripped baseline) — lock in early
19. Add cross-encoder + Tier-5 finder, re-test, Submission 2
20. Final calibration pass, full re-benchmark, Submission 3
21. Repo README, sandbox deployment, portal metadata, methodology summary, deck

---

*This plan supersedes the file-format assumptions in the earlier "Problem Statement" and "Strategy v3.5" documents wherever they conflict with this README. Re-verify Task 0 against `submission_spec.md` and `job_description.md` before finalizing scoring weights or the deck requirement.*
