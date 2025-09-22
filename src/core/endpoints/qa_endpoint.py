import sys
import uvicorn
from pathlib import Path
from dotenv import load_dotenv

from fastapi import (
    APIRouter,
    Depends,
    HTTPException, 
    FastAPI,
    status
)
import logging
from pydantic import BaseModel, Field

sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent))

from src.core.services.agent_service.graph import AgentGraph
from src.core.services.index_service import FAISSIndexService
from src.core.config.config import Config

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

load_dotenv()

app = FastAPI()

class QARequest(BaseModel):
    question: str = Field(..., description="User question")
    top_k: int = Field(default=3, description="Number of the most relevant chunks to return")
    max_iterations: int = Field(default=3, description="Maximum number of iterations")

class QAResponse(BaseModel):
    question: str
    answer: str

def get_config() -> Config:
    return Config()

def get_index_service(config: Config = Depends(get_config)) -> FAISSIndexService:
    return FAISSIndexService(
        config=config,
        load_index=True,
        save_index=False
    )

@app.post("/qa", tags=["QA"], response_model=QAResponse, status_code=status.HTTP_200_OK)
async def qa_endpoint(
    qa_request: QARequest,
    config: Config = Depends(get_config),
    index_service: FAISSIndexService = Depends(get_index_service),
) -> QAResponse:
    """QA endpoint which receives user prompt and returns agentic response."""
    try:
        agent = AgentGraph(
            config=config,
            top_k=qa_request.top_k,
            index_service=index_service,
            max_iterations=qa_request.max_iterations
        )
        
        response = agent.invoke(qa_request.question)
        return QAResponse(
            question=qa_request.question,
            answer=response["answer"]
        )
        
    except Exception as e:
        logger.error(f"Error in QA endpoint: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing request: {str(e)}"
        )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)