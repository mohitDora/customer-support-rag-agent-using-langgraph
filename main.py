from src.graph.agent_workflow import create_rag_agent_workflow
from src.models import AgentState


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

    user_query = "What is the warranty period for the QuantumFlow QF-2025"

    run_agent(user_query)

    print(
        "\n\nRemember to define your GOOGLE_API_KEY in a .env file at the project root."
    )
