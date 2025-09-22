import os
import sys
import json
import logging
from pathlib import Path
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langchain_core.runnables import Runnable
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent.parent))

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

load_dotenv()

from src.core.config.config import Config
from src.core.services.index_service import FAISSIndexService
from src.core.services.agent_service.state import AgentState
from src.core.services.agent_service.tools import rag_tool


class AgentGraph:
    """Agentic RAG workflow."""
    def __init__(
        self,
        top_k: int,
        max_iterations: int,
        config: Config,
        index_service: FAISSIndexService,
    ) -> None:
        self.top_k = top_k
        self.config = config
        self.index_service = index_service
        self.max_iterations = max_iterations

        self.tools = [rag_tool(index_service)]

        self.model = ChatOpenAI(
            model=os.getenv("OPENAI_MODEL_NAME"),
            api_key=os.getenv("OPENAI_API_KEY"),
            temperature=0
        )
        self.model_with_tools = self.model.bind_tools(self.tools)

        self.workflow = self._build_graph_workflow()

    def _build_graph_workflow(self, save_graph: bool = True) -> StateGraph:
        """Build the ReAct agent graph workflow."""
        workflow = StateGraph(state_schema=AgentState)

        workflow.add_node("agent", self.agent_node)
        workflow.add_node("action", self.tool_node)

        workflow.set_entry_point("agent")
        workflow.add_conditional_edges(
            "agent",
            self.should_continue,
            {
                "continue": "action",
                "end": END,
            }
        )

        workflow.add_edge("action", "agent")
        graph = workflow.compile()
        
        if save_graph:
            self._save_graph(graph)
        return graph
    
    def build_agent_chain(self) -> Runnable:
        """Build the agent chain with improved prompt engineering."""       
        system_prompt_template_2 = """
            You are a helpful agricultural assistant. Use the available tools to answer the user's question about agriculture.
            Always think step by step and explain your reasoning in Ukrainian.
            If you have enough information to answer the question, provide a complete answer in Ukrainian.
            If you need more information, use the available tools to gather it.

            Important instructions:
            - Provide complete, comprehensive answers based on the information you find
            - Do not ask follow-up questions at the end of your response
            - Focus on delivering the requested information clearly and concisely
            - If information is insufficient, use the search tool to get more context
            - Please write your answer in continuous text.

            Available tools: 
            - rag_semantic_search: Search for relevant information in the agricultural knowledge base.
        """
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt_template_2),
            MessagesPlaceholder(variable_name="messages"),
        ])

        chain = prompt | self.model_with_tools
        return chain

    def agent_node(self, state: AgentState) -> dict:
        """The agent node that decides what to do next."""
        messages = state.chat_history
        if not messages:
            messages = [HumanMessage(content=state.user_question)]

        chain = self.build_agent_chain()
        response = chain.invoke({"messages": messages})
        new_messages = messages + [response]

        return {"chat_history": new_messages}

    def tool_node(self, state: AgentState) -> dict:
        """The tools node that executes the selected tool."""
        last_message = state.chat_history[-1]
        tool_calls = last_message.additional_kwargs.get("tool_calls", [])

        tool_messages = []
        results = []

        for tool_call in tool_calls:
            tool_name = tool_call["function"]["name"]
            tool_args = json.loads(tool_call["function"]["arguments"])
            tool_call_id = tool_call["id"]

            for tool in self.tools:
                if tool.name == tool_name:
                    result = tool.invoke(tool_args)
                    if isinstance(result, list):
                        results.extend(result)
                    else:
                        results.append(result)

                    tool_message = ToolMessage(
                        content=json.dumps(result, ensure_ascii=False),
                        tool_call_id=tool_call_id,
                        name=tool_name
                    )
                    tool_messages.append(tool_message)
                    break

        new_sources = state.sources + results

        return {
            "chat_history": state.chat_history + tool_messages,
            "sources": new_sources,
            "iterations": state.iterations + 1
        }

    def should_continue(self, state: AgentState) -> str:
        last_message = state.chat_history[-1]
        tool_calls = last_message.additional_kwargs.get("tool_calls", [])

        if tool_calls:
            return "continue"

        if "final answer" in last_message.content.lower() or state.iterations >= self.max_iterations:
            return "end"

        return "continue"

    def invoke(self, question: str) -> dict:
        """Invoke the agent with a question."""
        initial_state = AgentState(user_question=question)

        final_state = self.workflow.invoke(initial_state)
        final_answer = self._extract_final_answer(final_state)

        return {
            "question": question,
            "answer": final_answer,
            "iterations": final_state["iterations"],
            "reasoning_history": [msg.content for msg in final_state["chat_history"] if isinstance(msg, AIMessage)]
        }

    def _extract_final_answer(self, state: dict) -> str:
        """Extract the final answer from the state."""
        for message in reversed(state["chat_history"]):
            if isinstance(message, AIMessage) and "final answer" in message.content.lower():
                return message.content

        for message in reversed(state["chat_history"]):
            if isinstance(message, AIMessage):
                return message.content

        return "No answer found"

    def _save_graph(self, graph) -> None:
        """Save graph schema as image to file."""
        png_bytes = graph.get_graph().draw_mermaid_png()
        with open(self.config.graph_path, "wb") as f:
            f.write(png_bytes)


def main() -> None:
    """Main function"""
    config = Config()

    faiss_service = FAISSIndexService(
        config,
        load_index=True,
        save_index=False
    )

    agent = AgentGraph(
        top_k=3,
        index_service=faiss_service,
        config=config,
        max_iterations=3
    )

    question="Як впливає розміщенні кукурудзи після цукрових буряків на почву?"

    result = agent.invoke(question)

    logger.info(f"Question: {question}")
    logger.info(f"Answer: {result['answer']}")
    logger.info(f"Number of iterations: {result['iterations']}")


if __name__ == "__main__":
    main()