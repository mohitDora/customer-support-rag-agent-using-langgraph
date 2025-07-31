from langgraph.graph import END, StateGraph

from src.agents.evaluator_agent import EvaluatorAgent
from src.agents.formatter_agent import FormatterAgent
from src.agents.research_agent import ResearchAgent
from src.agents.retriever_agent import RetrieverAgent
from src.agents.supervisor_agent import SupervisorAgent
from src.agents.synthesizer_agent import SynthesizerAgent
from src.models import AgentState

# Initialize agent instances
research_agent = ResearchAgent()
retriever_agent = RetrieverAgent()
evaluator_agent = EvaluatorAgent()
synthesizer_agent = SynthesizerAgent()
formatter_agent = FormatterAgent()
supervisor_agent = SupervisorAgent()


def create_rag_agent_workflow():
    """
    Defines and compiles the LangGraph workflow for the RAG agent.
    """
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("supervisor", supervisor_agent.run)
    workflow.add_node("research_agent", research_agent.run)
    workflow.add_node("retriever_agent", retriever_agent.run)
    workflow.add_node("evaluator_agent", evaluator_agent.run)
    workflow.add_node("synthesizer_agent", synthesizer_agent.run)
    workflow.add_node("formatter_agent", formatter_agent.run)

    # Set entry point
    workflow.set_entry_point("supervisor")

    workflow.add_edge("research_agent", "supervisor")
    workflow.add_edge("retriever_agent", "supervisor")
    workflow.add_edge("evaluator_agent", "supervisor")
    workflow.add_edge("synthesizer_agent", "supervisor")
    workflow.add_edge("formatter_agent", "supervisor")

    workflow.add_conditional_edges(
        "supervisor",
        lambda state: state["next_agent_to_call"],
        {
            "research_agent": "research_agent",
            "retriever_agent": "retriever_agent",
            "evaluator_agent": "evaluator_agent",
            "synthesizer_agent": "synthesizer_agent",
            "formatter_agent": "formatter_agent",
            "END": END,
            "FATAL_ERROR": END,
            "end_workflow": END,
        },
    )

    app = workflow.compile()

    return app


if __name__ == "__main__":
    compiled_app = create_rag_agent_workflow()
