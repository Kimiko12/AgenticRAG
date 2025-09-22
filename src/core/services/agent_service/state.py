from typing import List
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage


class AgentState(BaseModel):
    user_question: str = Field(default="", description="Default user question")
    answer: str = Field(default="", description="Final answer")
    sources: List[dict] = Field(default=[], description="List of sources which were used as additional context")
    chat_history: List[BaseMessage] = Field(default=[], description="List of all messages inside agent flow")
    iterations: int = Field(default=0, description="Number of iterations")