import sys
import logging
from pathlib import Path
from dotenv import load_dotenv
from typing import List, Dict, Any

from langchain_core.tools import BaseTool

from langchain_core.tools import tool

sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent.parent))

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

load_dotenv()

from src.core.services.index_service import FAISSIndexService
from src.core.config.config import Config


def rag_tool(index_service: FAISSIndexService) -> BaseTool:
    """Create tool for semantic serch in vector store."""

    @tool
    def rag_semantic_search(query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Tool in general Agentic RAG pipeline which uses semantic search 
        algorithm for retrieve relevant information from vector store.

        Args:
            qurey: str - input query for sematic search in index.
            top_k: int - number of relevant chunks to return.

        Returns:
            List[Dict] - list of the most relevant chunks of text. 
        """
        try:
            results = index_service.semantic_search(query, top_k)

            formatted_results = []
            for result in results:
                formatted_results.append({
                    'text': result['chunk_text'],
                    'score': result['score'],
                    'metadata': result['metadata']
                })

            return formatted_results

        except Exception as e:
            logger.error(f"Error in semantic search: {e}")
            return [{"error": str(e)}]

    return rag_semantic_search


def main() -> None:
    """Main function"""
    config = Config()
    faiss_service = FAISSIndexService(
        config,
        load_index=True,
        save_index=False
    )
    logger.info(f"Number of chunks:{len(faiss_service.chunks)}")

    input_query: str = "Як впливає розміщенні кукурудзи після цукрових буряків на почву?"
    rag_tool_instance = rag_tool(faiss_service)

    logger.info(rag_tool_instance.invoke(input={"index": faiss_service, "query": input_query, "top_k": 3}))


if __name__ == "__main__":
    main()