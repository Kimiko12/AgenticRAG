import sys
import faiss
import json
import logging
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional, Tuple
from FlagEmbedding import BGEM3FlagModel
from langchain_text_splitters import RecursiveCharacterTextSplitter

sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent))

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

load_dotenv()

from src.core.config.config import Config
from src.core.services.preprocessing_service import PreprocessingService


class FAISSIndexService:
    """
    This class encapsulates logic for working with FAISS index
    as a vector store for text embeddings.
    """
    def __init__(
        self,
        config: Config,
        load_index: bool = False,
        save_index: bool = True
    ) -> None:
        self.preprocessing_service = PreprocessingService(
            data_path=config.data_path
        )
        self.load_index = load_index
        self.save_index = save_index
        self.chunk_size = config.chunk_size
        self.chunk_overlap = config.chunk_overlap
        self.index_path = Path(config.index_path)
        self.meta_path = Path(config.meta_path)
        self.alpha = config.alpha
        self.M = config.M
        self.efConstruction = config.efConstruction
        self.embedding_model_name = config.embedding_model

        self.chunks: List[str] = []
        self.chunk_metadata: List[Dict[str, Any]] = []
        self.hnsw_index: Optional[faiss.Index] = None

        # Initialize multilingual embedding model for Ukrainian text
        self.embedding_model = BGEM3FlagModel(
            model_name_or_path=self.embedding_model_name
        )

        # Chunking text by using Recursive Character Text Splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " "]
        )

        documents = [doc["text"] for doc in self.preprocessing_service.cleaned_data]
        doc_ids = [idx for idx, _ in enumerate(self.preprocessing_service.cleaned_data)]

        if self.load_index:  # Load already existing index
            self.hnsw_index = self._load_index()
        elif self.load_index and self.hnsw_index:  # Reindex already existing index with new data
            self.hnsw_index = self.add_documents(documents, doc_ids)
        else:  # Create new index
            self.hnsw_index = self.create_index(documents, doc_ids)

        if self.save_index:  # Save created index locally in memory
            self._save_index()

    def chunk_documents(self, documents: List[str], doc_ids: Optional[List[str]]) -> Tuple[List[str], List[Dict[str, Any]]]:
        """
        Method for chunking entire text into smaller chunks with metadata.
        
        Args:
            documents: List[str] - list of documents to add to index.
            doc_ids: List[str] - list of document ids to add to index.
        Returns:
            Tuple[List[str], List[Dict[str, Any]]] - list of chunks and metadata.
        """
        all_chunks = []
        chunk_metadata = []

        for doc_idx, (doc, doc_id) in enumerate(zip(documents, doc_ids)):
            chunks = self.text_splitter.split_text(doc)

            for chunk_idx, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                chunk_metadata.append({
                    'doc_id': doc_id,
                    'chunk_index': chunk_idx,
                    'doc_index': doc_idx,
                    'chunk_text': chunk,
                    'chunk_length': len(chunk)
                })
        return all_chunks, chunk_metadata

    def create_index(self, documents: List[str], doc_ids: Optional[List[str]]) -> faiss.Index:
        """
        Method for indexing vector store in FAISS.
        
        Args:
            documents: List[str] - list of documents to add to index.
            doc_ids: List[str] - list of document ids to add to index.
        Returns:
            faiss.Index - reindexed vector store.
        """
        chunks, chunk_metadata = self.chunk_documents(documents, doc_ids)

        embeddings = self.embedding_model.encode(
            chunks,
            batch_size=32,
            max_length=512
        )["dense_vecs"]

        if not isinstance(embeddings, np.ndarray):
            embeddings = np.array(embeddings)

        hnsw_index = faiss.IndexHNSWFlat(
            embeddings.shape[1],
            self.M
        )
        hnsw_index.hnsw.efConstruction = self.efConstruction
        hnsw_index.add(embeddings.astype('float32'))
        self.chunks.extend(chunks)
        self.chunk_metadata.extend(chunk_metadata)

        return hnsw_index

    def _load_index(self) -> Optional[faiss.Index]:
        """Load index and metadata from disk"""
        try:
            index = faiss.read_index(str(self.index_path))

            with open(self.meta_path, 'r', encoding='utf-8') as file:
                metadata = json.load(file)

            self.chunks = metadata['chunks']
            self.chunk_metadata = metadata['chunk_metadata']
            return index
        except Exception as e:
            logger.error(f"Error loading index: {e}")
            return None

    def add_documents(self, documents: List[str], doc_ids: Optional[List[str]]) -> faiss.Index:
        """
        Method for reindexing vector store with new data.
        
        Args:
            documents: List[str] - list of documents to add to index.
            doc_ids: List[str] - list of document ids to add to index.
        Returns:
            faiss.Index - reindexed vector store.
        """
        new_chunks, new_metadata = self.chunk_documents(documents, doc_ids)

        if not new_chunks:
            logger.warning("No chunks sampled from documents")

        embeddings = self.embedding_model.encode(
            new_chunks,
            batch_size=32,
            max_length=512
        )['dense_vecs']

        if not isinstance(embeddings, np.ndarray):
            embeddings = np.array(embeddings)

        if self.hnsw_index:
            hnsw_index = self.hnsw_index
            hnsw_index.add(embeddings.astype('float32'))
            self.chunks.extend(new_chunks)
            self.chunk_metadata.extend(new_metadata)
        else:
            logger.warning("No index to add documents!")

        return hnsw_index

    def _save_index(self) -> None:
        """Save index and metadata to disk"""
        if self.hnsw_index is None:
            logger.warning("No index to save")
            return

        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.meta_path.parent.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.hnsw_index, str(self.index_path))

        metadata = {
            'chunks': self.chunks,
            'chunk_metadata': self.chunk_metadata,
            'chunk_size': self.chunk_size,
            'chunk_overlap': self.chunk_overlap,
            'total_chunks': len(self.chunks)
        }

        with open(str(self.meta_path), 'w', encoding='utf-8') as file:
            json.dump(metadata, file, ensure_ascii=False, indent=2)

        logger.info(f"Index saved to {self.index_path}")
        logger.info(f"Metadata saved to {self.meta_path}")

    def semantic_search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Method for sematic search in vector store by query.
        
        Args:
            query: str - input query for sematic search in index.
            top_k: int - number of relevant chunks to return.

        Returns:
            List[Dict] - list of the most relevant chunks of text.
        """
        if not self.hnsw_index or len(self.chunks) == 0:
            logger.warning("No index to search or no chunks available")
            return []

        query_embedding = self.embedding_model.encode(
            [query],
            batch_size=1,
            max_length=512
        )['dense_vecs']

        if not isinstance(query_embedding, np.ndarray):
            query_embedding = np.array(query_embedding)

        self.hnsw_index.hnsw.efSearch = self.efConstruction // 2
        scores, indices = self.hnsw_index.search(query_embedding.astype('float32'), min(top_k, len(self.chunks)))

        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx >= 0:
                result = {
                    'rank': i + 1,
                    'score': float(score),
                    'chunk_text': self.chunks[idx],
                    'metadata': self.chunk_metadata[idx]
                }
                results.append(result)

        return results

# TODO: (Improvments) Add Semantic Chunking to improve context quality inside every chunk
# TODO: (Improvments) Add Full Text Seach and Hybrid Search
# TODO: (Improvments) Experiment with different tokenizers & embedding models


def main() -> None:
    """Main function"""
    config = Config()
    faiss_service = FAISSIndexService(
        config,
        load_index=True,
        save_index=False
    )

    logger.info(f"Number of chunks:{len(faiss_service.chunks)}")

    results = faiss_service.semantic_search("– вдруге – у фазу початку викидання волотi 1,5 л/га.", top_k=3)
    for result in results:
        logger.info(f"Search result: {result['chunk_text']}")
        logger.info(len(result['chunk_text']))


if __name__ == "__main__":
    main()