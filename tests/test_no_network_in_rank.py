import ast
from pathlib import Path
import pytest

# Core runtime modules that are imported during Phase 2 ranking
RUNTIME_FILES = [
    Path("scripts/rank.py"),
    Path("src/config.py"),
    Path("src/schema.py"),
    Path("src/data_loader.py"),
    Path("src/embeddings.py"),
    Path("src/skill_ontology.py"),
    Path("src/honeypot_detection.py"),
    Path("src/hard_gates.py"),
    Path("src/scoring.py"),
    Path("src/cross_encoder_rerank.py"),
    Path("src/tier5_finder.py"),
    Path("src/assembly.py"),
    Path("src/utils.py"),
    Path("src/llm_feature_extraction.py"),
]

BANNED_IMPORTS = {"requests", "httpx", "urllib", "openai", "anthropic", "google"}

def check_file_for_banned_imports(file_path: Path):
    if not file_path.exists():
        # Some files might not be created yet during early testing, skip
        return
        
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
        
    tree = ast.parse(content, filename=str(file_path))
    
    for node in ast.walk(tree):
        # Handle 'import banned'
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.name.split('.')[0]
                if name in BANNED_IMPORTS:
                    pytest.fail(f"Banned network import '{name}' found in runtime file {file_path}")
        # Handle 'from banned import ...'
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                name = node.module.split('.')[0]
                if name in BANNED_IMPORTS:
                    pytest.fail(f"Banned network import '{name}' found in runtime file {file_path}")

def test_no_network_in_runtime_path():
    """
    Ensure no runtime files contain imports of networking or LLM client libraries.
    """
    for file_path in RUNTIME_FILES:
        check_file_for_banned_imports(file_path)
