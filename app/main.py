# app.py (New file, for API exposure)
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

from src.graph.agent_workflow import create_rag_agent_workflow
from src.models import AgentState
from src.config import ConfigurationManager

rag_app = create_rag_agent_workflow()

app = FastAPI(title="RAG Agent API", version="1.0.0")


class QueryRequest(BaseModel):
    query: str


@app.post("/query")
async def process_query(request: QueryRequest):
    """
    Endpoint to process a user query using the RAG agent.
    """
    print(f"Received query: {request.query}")

    config = ConfigurationManager()
    knowledge_base_config = config.get_knowledge_base_config()
    chroma_db_dir = knowledge_base_config["CHROMA_DB_DIR"]

    if not os.path.exists(chroma_db_dir) or not os.listdir(chroma_db_dir):
        raise HTTPException(
            status_code=500,
            detail="Knowledge base not found or empty. Please ensure it's mounted correctly.",
        )

    initial_state: AgentState = {
        "original_query": request.query,
        "sub_queries_list": [],
        "current_sub_query_index": 0,
        "current_sub_query": "",
        "retrieved_chunks": [],
        "evaluated_sufficiency": False,
        "evaluator_feedback": "",
        "retrieval_attempts": 0,
        "accumulated_relevant_chunks": [],
        "unanswerable_sub_queries": [],
        "final_answer_draft": "",
        "report_formatted": "",
        "next_agent_to_call": "research_agent",
    }

    try:
        final_state = rag_app.invoke(initial_state)
        if final_state.get("report_formatted"):
            return {"answer": final_state["report_formatted"]}
        else:
            raise HTTPException(
                status_code=500,
                detail="An error occurred and no final answer could be generated.",
            )
    except Exception as e:
        print(f"Error during agent execution: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")


@app.get("/health")
async def health_check():
    return {"status": "ok", "message": "RAG Agent API is running"}
