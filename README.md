# Redrob-Ranker: Intelligent Candidate Discovery & Ranking

This repository implements a 2-Phase Retrieve-and-Rerank candidate discovery and ranking system to match 100,000 candidate profiles against a Senior AI Engineer hiring requirement.

The ranking pipeline is optimized for CPU-only execution and completes in under **30 seconds** on the full 100K pool using **under 2 GB RAM**, satisfying all computational constraints (Budget: ≤ 5 min wall-clock, ≤ 16 GB RAM).

## 🚀 Setup & Execution

### 1. Environment Setup
Set up the Python virtual environment and install the pinned dependencies:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Phase 1: Pre-Computation (Offline, Network Allowed)
Run the pre-computation step to generate dense embeddings, build the HNSW FAISS search index, normalise skills, pre-cache model weights, and compute LLM-derived features on the top search candidates:
```bash
OMP_NUM_THREADS=1 PYTHONPATH=. python scripts/setup_artifacts.py
```
*Note: For faster local testing, you can use the `--limit-embed 3000` flag to embed a subset of the dataset and pad the rest.*

### 3. Phase 2: Timed Ranking (Stage 3 Sandboxed, No Network)
This is the single reproducible command to execute ranking and generate the final submission CSV file:
```bash
OMP_NUM_THREADS=1 PYTHONPATH=. python scripts/rank.py --candidates ./candidates.jsonl --out ./submission.csv
```

### 4. Running the Computational Benchmark
Measure wall-clock time and peak RSS memory utilization for every stage of the pipeline:
```bash
OMP_NUM_THREADS=1 PYTHONPATH=. python scripts/benchmark_compute.py --candidates ./candidates.jsonl
```

### 5. Running the Test Suite
Execute the test suite to verify schema compliance, honeypots, assembly uniqueness, and validator check:
```bash
OMP_NUM_THREADS=1 PYTHONPATH=. pytest
```

---

## 🏗️ Repository Architecture

- `src/config.py`: Core configurations, weights, thresholds, and path definitions.
- `src/schema.py`: Pydantic validation schemas matching `candidate_schema.json`.
- `src/data_loader.py`: High-speed generator-based streaming ingestion.
- `src/jd_parser.py`: Extracts requirements, preferred locations, and disqualifiers.
- `src/embeddings.py`: Profile concatenation recipes and BGE query/passage embeddings.
- `src/skill_ontology.py`: Cosine-similarity agglomerative skill clustering.
- `src/honeypot_detection.py`: Impossible dates, duration-proficiency clashes, and date sequencing rules to catch traps.
- `src/hard_gates.py`: In-office proximity, TITLES, and consulting-only (WITCH-tier) gates.
- `src/scoring.py`: Combines core fit, logistics, education, and behavioral multipliers.
- `src/cross_encoder_rerank.py`: Local cross-encoder model scoring.
- `src/tier5_finder.py`: Discovers hidden-gem recommendation/search developers.
- `src/assembly.py`: Overlap deduplication and score monotonicity reconciliation.

---

## 💡 Methodology Summary

Our solution implements an advanced retrieve-and-rerank model:
1. **Semantic Retrieval**: Top 4,000 candidates are retrieved from a HNSW index using `BAAI/bge-large-en-v1.5`.
2. **Hard & Soft Gates**: Strict heuristics filter out honeypot profiles, candidates from WITCH consulting firms, and non-engineering titles.
3. **Core Scorer**: Computes a base match using Core ML/Search Fit (70%), Logistics (20%), and Education (10%).
4. **Behavioral Multiplier**: Stated responsiveness, last active dates, and open-to-work flags modify scores with a 0.60–1.0 multiplier.
5. **Cross-Encoder Rerank**: The top 200 candidates are reranked via a locally cached `cross-encoder/ms-marco-MiniLM-L-6-v2`.
6. **Tier-5 Force-Includes**: Scans the dataset to identify developers with deep recommendation/search experience but zero buzzwords, force-including them at the end of the top-100 with reconciled non-increasing scores.
