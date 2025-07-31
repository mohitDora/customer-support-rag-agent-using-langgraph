from pathlib import Path
from typing import List

from langchain.prompts import PromptTemplate
from langchain_core.documents import Document

from src.constants import BASE_DIR, MAX_RETRIEVAL_ATTEMPTS
from src.llm_config import LLM
from src.models import AgentState
from src.utils.common import read_txt


class EvaluatorAgent:
    """
    Agent responsible for evaluating the sufficiency of retrieved chunks
    to answer the current sub-query.
    """

    def __init__(self):
        self.llm = LLM
        raw_prompt = read_txt(Path(BASE_DIR) / "prompts" / "evaluator_agent_prompt.txt")
        self.prompt_template = PromptTemplate(
            template=raw_prompt,
            input_variables=[
                {
                    "current_sub_query": "current_sub_query",
                    "retrieved_chunks_content": "retrieved_chunks_content",
                }
            ],
        )

    def run(self, state: AgentState) -> AgentState:
        """
        Evaluates the retrieved chunks and decides whether they are sufficient.
        Manages retry logic.
        """
        print("---EVALUATOR AGENT: Evaluating retrieved chunks---")

        current_sub_query = state["current_sub_query"]
        retrieved_chunks: List[Document] = state["retrieved_chunks"]
        retrieval_attempts = state["retrieval_attempts"]
        accumulated_relevant_chunks = state.get("accumulated_relevant_chunks", [])
        unanswerable_sub_queries = state.get("unanswerable_sub_queries", [])

        if not retrieved_chunks:
            print(
                f"---EVALUATOR AGENT: No chunks to evaluate for '{current_sub_query}'. Marking as insufficient.---"
            )
            evaluated_sufficiency = False
            evaluator_feedback = "No relevant chunks were retrieved."
        else:
            retrieved_chunks_content = "\n\n".join(
                [chunk.page_content for chunk in retrieved_chunks]
            )
            try:
                chain = self.prompt_template | self.llm
                response = chain.invoke(
                    {
                        "current_sub_query": current_sub_query,
                        "retrieved_chunks_content": retrieved_chunks_content,
                    }
                )

                response_content = response.content.strip().upper()
                print(f"---EVALUATOR AGENT: LLM Response:\n{response_content}---")

                # Parse LLM response
                if "SUFFICIENCY: YES" in response_content:
                    evaluated_sufficiency = True
                    evaluator_feedback = ""
                else:
                    evaluated_sufficiency = False
                    # Extract feedback if available
                    feedback_line = [
                        line
                        for line in response_content.split("\n")
                        if "FEEDBACK:" in line
                    ]
                    evaluator_feedback = (
                        feedback_line[0].replace("FEEDBACK:", "").strip()
                        if feedback_line
                        else "Information insufficient."
                    )
                    if (
                        not evaluator_feedback
                    ):  # Ensure there's always some feedback if NO
                        evaluator_feedback = "Information insufficient."

            except Exception as e:
                print(f"---ERROR: Evaluator agent failed during LLM call: {e}---")
                evaluated_sufficiency = False
                evaluator_feedback = (
                    "LLM evaluation failed. Assuming insufficient for retry."
                )

        print(
            f"---EVALUATOR AGENT: Sufficiency: {evaluated_sufficiency}. Feedback: '{evaluator_feedback}'---"
        )

        if evaluated_sufficiency:
            print(
                f"---EVALUATOR AGENT: Chunks are sufficient for '{current_sub_query}'. Accumulating and moving to next sub-query.---"
            )
            # Accumulate chunks if sufficient
            accumulated_relevant_chunks.extend(retrieved_chunks)
            next_agent = "research_agent"  # Go back to research to pick next sub-query
            current_sub_query_index = state["current_sub_query_index"] + 1

        elif retrieval_attempts < MAX_RETRIEVAL_ATTEMPTS:
            print(
                f"---EVALUATOR AGENT: Chunks insufficient. Retrying retrieval for '{current_sub_query}'.---"
            )
            next_agent = "retriever_agent"  # Loop back to retriever
            current_sub_query_index = state[
                "current_sub_query_index"
            ]  # Stay on same sub-query
        else:
            print(
                f"---EVALUATOR AGENT: Max retrieval attempts reached for '{current_sub_query}'. Marking as unanswerable.---"
            )
            unanswerable_sub_queries.append(current_sub_query)
            next_agent = "research_agent"  # Move to next sub-query
            current_sub_query_index = state["current_sub_query_index"] + 1

        return {
            **state,
            "evaluated_sufficiency": evaluated_sufficiency,
            "evaluator_feedback": evaluator_feedback,
            "retrieval_attempts": retrieval_attempts,
            "accumulated_relevant_chunks": accumulated_relevant_chunks,
            "unanswerable_sub_queries": unanswerable_sub_queries,
            "current_sub_query_index": current_sub_query_index,
            "next_agent_to_call": next_agent,
        }
