from pathlib import Path

from langchain.prompts import PromptTemplate

from src.constants import BASE_DIR
from src.llm_config import LLM
from src.models import AgentState
from src.utils.common import read_txt


class FormatterAgent:
    """
    Agent responsible for formatting and polishing the final answer report.
    """

    def __init__(self):
        self.llm = LLM
        raw_prompt = read_txt(Path(BASE_DIR) / "prompts" / "formatter_agent_prompt.txt")
        self.prompt_template = PromptTemplate(
            template=raw_prompt,
            input_variables=[{"final_answer_draft": "final_answer_draft"}],
        )

    def run(self, state: AgentState) -> AgentState:
        """
        Formats the final answer draft into a polished report.
        """
        print("---FORMATTER AGENT: Formatting final report---")

        final_answer_draft = state["final_answer_draft"]

        try:
            chain = self.prompt_template | self.llm
            response = chain.invoke({"final_answer_draft": final_answer_draft})
            report_formatted = response.content
        except Exception as e:
            print(f"---ERROR: Formatter agent failed during LLM call: {e}---")
            report_formatted = (
                "An error occurred during formatting. Here's the raw draft:\n\n"
                + final_answer_draft
            )

        print("---FORMATTER AGENT: Final Report Formatted. Workflow END.---")

        return {
            **state,
            "report_formatted": report_formatted,
            "next_agent_to_call": "END",
        }
