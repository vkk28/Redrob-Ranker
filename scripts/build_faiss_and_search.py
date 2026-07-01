import faiss
import numpy as np
import json
from pathlib import Path

def main():
    artifacts_dir = Path("artifacts")
    print("Loading candidate embeddings...")
    embeddings = np.load(artifacts_dir / "candidate_embeddings.npy")
    print("Candidate embeddings loaded, shape:", embeddings.shape)
    
    print("Creating FAISS IndexFlatIP...")
    index = faiss.IndexFlatIP(384)
    index.add(embeddings)
    
    faiss.write_index(index, str(artifacts_dir / "faiss_index.bin"))
    print("Saved faiss_index.bin")
    
    print("Loading JD embedding...")
    jd_embedding = np.load(artifacts_dir / "jd_embedding.npy")
    print("JD embedding loaded, shape:", jd_embedding.shape)
    
    print("Searching Flat index...")
    distances, indices = index.search(jd_embedding.reshape(1, -1), 4000)
    
    retrieved = indices[0].tolist()
    with open(artifacts_dir / "retrieved_indices.json", "w") as f:
        json.dump(retrieved, f)
    print(f"Successfully saved {len(retrieved)} retrieved indices.")

if __name__ == '__main__':
    main()
