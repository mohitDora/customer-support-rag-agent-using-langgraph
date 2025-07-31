from pathlib import Path
from typing import List

from langchain.prompts import PromptTemplate
from langchain_core.documents import Document

from src.constants import BASE_DIR
from src.llm_config import LLM
from src.models import AgentState
from src.utils.common import read_txt


class SynthesizerAgent:
    """
    Agent responsible for synthesizing a draft answer from accumulated relevant chunks.
    """

    def __init__(self):
        self.llm = LLM
        raw_prompt = read_txt(
            Path(BASE_DIR) / "prompts" / "synthesizer_agent_prompt.txt"
        )
        self.prompt_template = PromptTemplate(
            template=raw_prompt,
            input_variables=[
                {
                    "original_query": "original_query",
                    "accumulated_relevant_chunks_content": "accumulated_relevant_chunks_content",
                    "unanswerable_sub_queries_str": "unanswerable_sub_queries_str",
                }
            ],
        )

    def run(self, state: AgentState) -> AgentState:
        """
        Synthesizes the final answer draft from all accumulated relevant chunks.
        """
        print("---SYNTHESIZER AGENT: Generating final answer draft---")

        original_query = state["original_query"]
        accumulated_relevant_chunks: List[Document] = state.get(
            "accumulated_relevant_chunks", []
        )
        unanswerable_sub_queries: List[str] = state.get("unanswerable_sub_queries", [])

        if not accumulated_relevant_chunks:
            print(
                "---SYNTHESIZER AGENT: No relevant chunks accumulated. Cannot synthesize.---"
            )
            final_answer_draft = (
                "I could not find sufficient information in the knowledge base to answer your query: "
                + original_query
            )
            if unanswerable_sub_queries:
                final_answer_draft += f" (Specifically, could not answer sub-queries: {', '.join(unanswerable_sub_queries)})"
        else:
            accumulated_relevant_chunks_content = "\n\n".join(
                [chunk.page_content for chunk in accumulated_relevant_chunks]
            )
            unanswerable_sub_queries_str = (
                "\n".join([f"- {sq}" for sq in unanswerable_sub_queries])
                if unanswerable_sub_queries
                else "None"
            )

            try:
                chain = self.prompt_template | self.llm
                response = chain.invoke(
                    {
                        "original_query": original_query,
                        "accumulated_relevant_chunks_content": accumulated_relevant_chunks_content,
                        "unanswerable_sub_queries_str": unanswerable_sub_queries_str,
                    }
                )
                final_answer_draft = response.content
            except Exception as e:
                print(f"---ERROR: Synthesizer agent failed during LLM call: {e}---")
                final_answer_draft = "An error occurred during synthesis. I might not be able to provide a full answer."
                if accumulated_relevant_chunks:
                    final_answer_draft += (
                        "\n\nBased on available information, but potentially unrefined: "
                        + accumulated_relevant_chunks_content[:500]
                        + "..."
                    )
                elif unanswerable_sub_queries:
                    final_answer_draft += f"\n\nCould not answer specific parts: {', '.join(unanswerable_sub_queries)}"

        print("---SYNTHESIZER AGENT: Draft Answer Generated. Moving to formatting.---")

        return {
            **state,
            "final_answer_draft": final_answer_draft,
            "next_agent_to_call": "formatter_agent",
        }
