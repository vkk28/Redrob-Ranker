import json
from pathlib import Path
from typing import Dict, List, Any, Set
import numpy as np
from src.embeddings import embed_text
from src.utils import get_logger

logger = get_logger("skill_ontology")

class SkillOntology:
    def __init__(self, ontology_path: Path):
        self.ontology_path = ontology_path
        self.mapping: Dict[str, str] = {}  # skill_name -> canonical_name
        self.load()

    def load(self):
        if self.ontology_path.exists():
            try:
                with open(self.ontology_path, "r", encoding="utf-8") as f:
                    self.mapping = json.load(f)
                logger.info(f"Loaded skill ontology with {len(self.mapping)} mappings")
            except Exception as e:
                logger.error(f"Error loading skill ontology: {e}")
                self.mapping = {}

    def save(self):
        try:
            self.ontology_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.ontology_path, "w", encoding="utf-8") as f:
                json.dump(self.mapping, f, indent=2)
            logger.info(f"Saved skill ontology with {len(self.mapping)} mappings to {self.ontology_path}")
        except Exception as e:
            logger.error(f"Error saving skill ontology: {e}")

    def get_canonical_name(self, skill_name: str) -> str:
        """
        Resolve a skill name to its canonical representative.
        Returns the lowercase cleaned skill name if not mapped.
        """
        cleaned = skill_name.strip().lower()
        return self.mapping.get(cleaned, cleaned)

    def build_ontology(self, unique_skills: List[str]):
        """
        Build mapping by embedding unique skills and clustering similar ones.
        We group any skills with cosine similarity >= 0.82 together.
        """
        logger.info(f"Building skill ontology for {len(unique_skills)} unique skills...")
        if not unique_skills:
            return

        # Clean skills
        cleaned_skills = sorted(list({s.strip().lower() for s in unique_skills if s.strip()}))
        
        # Embed in batches to save time
        logger.info("Embedding unique skill strings...")
        embeddings = embed_text(cleaned_skills)
        
        # Simple threshold-based agglomerative clustering
        # mapping: cleaned_skill -> canonical_skill
        visited = set()
        mapping = {}

        for i, skill in enumerate(cleaned_skills):
            if skill in visited:
                continue
                
            # Current skill is the start of a new cluster
            cluster = [skill]
            visited.add(skill)
            
            # Find all other skills close to this one
            emb_i = embeddings[i]
            for j in range(i + 1, len(cleaned_skills)):
                other_skill = cleaned_skills[j]
                if other_skill in visited:
                    continue
                    
                emb_j = embeddings[j]
                # Cosine similarity
                sim = np.dot(emb_i, emb_j)
                if sim >= 0.82:
                    cluster.append(other_skill)
                    visited.add(other_skill)
                    
            # Choose the shortest skill name in the cluster as the canonical one
            canonical = min(cluster, key=len)
            for s in cluster:
                mapping[s] = canonical
                
        self.mapping = mapping
        self.save()
