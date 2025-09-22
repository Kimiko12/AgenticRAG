import sys
from pathlib import Path
from dotenv import load_dotenv
from dataclasses import dataclass

sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent))

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent


@dataclass
class Config:
    data_path: Path = BASE_DIR / "data/row"
    output_data_path: Path = BASE_DIR / "data/clean"

    chunk_size: int = 700
    chunk_overlap: int = 100
    embedding_model: str = "BAAI/bge-m3"
    index_path: str = BASE_DIR / "FAISS/faiss_index.bin"
    meta_path: str = BASE_DIR / "FAISS/faiss_meta.json"
    graph_path: str = BASE_DIR / "graph.png"
    alpha: float = 0.7  # Weight for RRF
    M: int = 16
    efConstruction: int = 200