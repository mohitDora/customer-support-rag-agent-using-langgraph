from typing import List, TypedDict, Literal
from langchain_core.documents import Document

class AgentState(TypedDict):
    """
    Represents the state of our RAG agent's overall workflow.
    This state is passed between all nodes in the LangGraph.
    """
    original_query: str
    sub_queries_list: List[str]
    current_sub_query_index: int
    current_sub_query: str 
    retrieved_chunks: List[Document]
    evaluated_sufficiency: bool
    evaluator_feedback: str
    retrieval_attempts: int
    accumulated_relevant_chunks: List[Document] 
    unanswerable_sub_queries: List[str] 
    final_answer_draft: str 
    report_formatted: str 

    next_agent_to_call: Literal[
        'research_agent',
        'retriever_agent',
        'evaluator_agent',
        'synthesizer_agent',
        'formatter_agent',
        'END',              
        'FATAL_ERROR'
    ]