import re
from pathlib import Path
from typing import Dict, Any, List
from src.utils import get_logger

logger = get_logger("jd_parser")

class JobDescriptionRequirements:
    def __init__(self):
        self.role_name = "Senior AI Engineer — Founding Team"
        self.experience_min = 5
        self.experience_max = 9
        
        # Must-have categories & keywords (lowercase)
        self.must_have_embeddings = ["embedding", "sentence-transformer", "bge", "e5", "dense retrieval", "vector representation"]
        self.must_have_vector_db = ["vector database", "pinecone", "weaviate", "qdrant", "milvus", "opensearch", "elasticsearch", "faiss", "hybrid search", "vector search"]
        self.must_have_eval = ["evaluation framework", "ndcg", "mrr", "map", "precision@", "recall@", "a/b test", "offline-to-online", "ranking metrics", "eval infrastructure"]
        self.must_have_python = ["python", "pyspark", "numpy", "pandas", "scikit-learn"]

        # Nice-to-have keywords (lowercase)
        self.nice_to_haves = [
            "fine-tuning", "lora", "qlora", "peft", "learning-to-rank", "xgboost", 
            "hr-tech", "recruiting tech", "marketplace", "distributed systems", 
            "inference optimization", "open-source"
        ]

        # Locations (lowercase)
        self.preferred_locations = ["pune", "noida", "delhi", "ncr", "hyderabad", "mumbai", "bangalore", "bengaluru"]

        # Consulting firms to exclude if ONLY at these
        self.witch_companies = {"tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini"}

def parse_job_description(jd_path: Path) -> JobDescriptionRequirements:
    logger.info(f"Parsing job description from {jd_path}...")
    reqs = JobDescriptionRequirements()
    
    if not jd_path.exists():
        logger.warning(f"JD file {jd_path} not found. Using pre-defined default requirements.")
        return reqs
        
    try:
        with open(jd_path, "r", encoding="utf-8") as f:
            content = f.read()
            
        # Extract experience range if present
        exp_match = re.search(r"(\d+)–(\d+)\s*years", content, re.IGNORECASE)
        if exp_match:
            reqs.experience_min = int(exp_match.group(1))
            reqs.experience_max = int(exp_match.group(2))
            logger.info(f"Parsed experience range: {reqs.experience_min} to {reqs.experience_max} years")
            
    except Exception as e:
        logger.error(f"Error parsing JD file: {e}. Falling back to default spec.")
        
    return reqs
