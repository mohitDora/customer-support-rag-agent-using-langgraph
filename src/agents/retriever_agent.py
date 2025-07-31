from typing import List

from langchain_core.documents import Document

from src.constants import DEFAULT_RETRIEVAL_K
from src.models import AgentState
from src.utils.db_utils import get_vector_db


class RetrieverAgent:
    """
    Agent responsible for retrieving relevant chunks from the vector database.
    """

    def __init__(self):
        self.vector_db = get_vector_db()
        self.retriever = self.vector_db.as_retriever(
            search_kwargs={"k": DEFAULT_RETRIEVAL_K}
        )

    def run(self, state: AgentState) -> AgentState:
        """
        Retrieves document chunks based on the current sub-query.
        """
        print("---RETRIEVER AGENT: Retrieving information---")
        current_sub_query = state["current_sub_query"]
        retrieval_attempts = state["retrieval_attempts"] + 1

        print(
            f"---RETRIEVER AGENT: Attempt {retrieval_attempts} for '{current_sub_query}'---"
        )

        try:
            retrieved_chunks: List[Document] = self.retriever.invoke(current_sub_query)
            if not retrieved_chunks:
                print(
                    f"---RETRIEVER AGENT: No chunks retrieved for '{current_sub_query}'---"
                )

            return {
                **state,
                "retrieved_chunks": retrieved_chunks,
                "retrieval_attempts": retrieval_attempts,
                "next_agent_to_call": "evaluator_agent",
            }
        except Exception as e:
            print(f"---ERROR: Retriever agent failed to retrieve chunks: {e}---")
            unanswerable_sub_queries = state.get("unanswerable_sub_queries", [])
            unanswerable_sub_queries.append(current_sub_query)
            return {
                **state,
                "unanswerable_sub_queries": unanswerable_sub_queries,
                "current_sub_query_index": state["current_sub_query_index"] + 1,
                "next_agent_to_call": "research_agent",
            }
