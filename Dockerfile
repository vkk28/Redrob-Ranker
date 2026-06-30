FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ src/
COPY scripts/ scripts/
COPY validate_submission.py .
COPY job_description.md .
COPY redrob_signals_doc.md .
COPY candidate_schema.json .
COPY submission_metadata.yaml .

# Copy pre-computed artifacts (small ones — embeddings/index must be regenerated or mounted)
COPY artifacts/jd_embedding.npy artifacts/
COPY artifacts/candidate_id_map.json artifacts/
COPY artifacts/llm_features.jsonl artifacts/
COPY artifacts/manifest.json artifacts/
COPY artifacts/skill_ontology.json artifacts/
COPY artifacts/calibration_weights.json artifacts/

# Environment
ENV OMP_NUM_THREADS=1
ENV PYTHONPATH=/app

# Default command: rank a candidates file and produce submission.csv
# Usage: docker run -v /path/to/candidates.jsonl:/app/candidates.jsonl redrob-ranker
ENTRYPOINT ["python", "scripts/rank.py"]
CMD ["--candidates", "./candidates.jsonl", "--out", "./submission.csv"]
