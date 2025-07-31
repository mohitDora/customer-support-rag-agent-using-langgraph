from src.graph.agent_workflow import create_rag_agent_workflow
from src.models import AgentState  # Import the AgentState TypedDict

# from src.config import RAW_DOCS_DIR, CHROMA_DB_DIR # Import directories for checks


def run_agent(query: str):
    """
    Initializes and runs the RAG agent workflow for a given query.
    """

    rag_app = create_rag_agent_workflow()

    # Define the initial state for the graph
    initial_state: AgentState = {
        "original_query": query,
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
        "next_agent_to_call": "research_agent",  # Initial state to start the process
    }

    print(f"\n--- Starting RAG Agent for query: '{query}' ---\n")

    # Run the graph
    # The 'stream' method is good for seeing intermediate steps, 'invoke' gets the final state.
    # We'll use invoke for simplicity here to get the final result.
    final_state = rag_app.invoke(initial_state, config={"recursion_limit": 200})

    print("\n--- RAG Agent Workflow Completed ---\n")

    if final_state.get("report_formatted"):
        print("\n--- FINAL ANSWER ---\n")
        print(final_state["report_formatted"])
        return final_state["report_formatted"]
    else:
        print("\n--- ERROR: Could not generate a final answer. ---\n")
        print(f"Last known state: {final_state}")
        return "An error occurred and no final answer could be generated."


if __name__ == "__main__":
    # Example usage:
    user_query = " What is the warranty period for the QuantumFlow QF-2025"
    # user_query = "Summarize the key differences between various types of large language models."
    # user_query = "Tell me about the history of artificial intelligence."

    run_agent(user_query)

    print(
        "\n\nRemember to define your GOOGLE_API_KEY in a .env file at the project root."
    )
    # print(f"Ensure your knowledge base documents are in '{RAW_DOCS_DIR}' and run 'python ingest.py' before querying.")
