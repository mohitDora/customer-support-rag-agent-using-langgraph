import os
import pandas as pd
from typing import List, Dict, Any
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

from langchain.evaluation import EvaluatorType, load_evaluator

from src.graph.agent_workflow import create_rag_agent_workflow
from src.models import AgentState
from src.config import ConfigurationManager
from src.llm_config import LLM


class RAGEvaluator:
    """
    Class to orchestrate the evaluation of the RAG agent.
    """

    def __init__(self, chroma_db_dir=None):
        self.rag_app = create_rag_agent_workflow()
        self.evaluation_llm = LLM
        self.faithfulness_evaluator = load_evaluator(
            EvaluatorType.SCORE_STRING,
            criteria="faithfulness",
            llm=self.evaluation_llm,  # Pass the evaluation LLM
        )
        self.answer_relevance_evaluator = load_evaluator(
            EvaluatorType.SCORE_STRING,
            criteria="answer_relevance",
            llm=self.evaluation_llm,  # Pass the evaluation LLM
        )
        self.chroma_db_dir = chroma_db_dir

    def _run_agent_and_get_results(self, query: str) -> Dict[str, Any]:
        """Runs the RAG agent and returns the final state."""
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
            "next_agent_to_call": "research_agent",
        }
        final_state = self.rag_app.invoke(initial_state, config={"recursion_limit": 200})
        return final_state

    def evaluate_query(self, query: str) -> Dict[str, Any]:
        """
        Runs the agent for a single query and evaluates its output.
        """
        print(f"\n--- Evaluating query: '{query}' ---")
        agent_output = self._run_agent_and_get_results(query)
        generated_answer = agent_output.get("report_formatted", "")
        retrieved_contexts_content = "\n\n".join(
            [
                doc.page_content
                for doc in agent_output.get("accumulated_relevant_chunks", [])
            ]
        )

        results = {
            "query": query,
            "generated_answer": generated_answer,
            "retrieved_contexts_count": len(
                agent_output.get("accumulated_relevant_chunks", [])
            ),
            "unanswerable_sub_queries": ", ".join(
                agent_output.get("unanswerable_sub_queries", [])
            ),
            "faithfulness_score": None,
            "faithfulness_reasoning": None,
            "answer_relevance_score": None,
            "answer_relevance_reasoning": None,
            "status": "Success" if generated_answer else "Failed",
        }

        if not generated_answer:
            print("No answer generated for evaluation.")
            return results

        try:
            # Evaluate Faithfulness (Generated Answer vs. Retrieved Contexts)
            faithfulness_eval_result = self.faithfulness_evaluator.evaluate_strings(
                prediction=generated_answer,
                input=query,  # Pass the original query as input
                reference=retrieved_contexts_content,  # Crucially, pass the retrieved content here
            )
            results["faithfulness_score"] = faithfulness_eval_result.get("score")
            results["faithfulness_reasoning"] = faithfulness_eval_result.get(
                "reasoning"
            )
            print(f"  Faithfulness Score: {results['faithfulness_score']}")

            # Evaluate Answer Relevance (Generated Answer vs. Original Query)
            answer_relevance_eval_result = (
                self.answer_relevance_evaluator.evaluate_strings(
                    prediction=generated_answer, input=query  # Only need the query here
                )
            )
            results["answer_relevance_score"] = answer_relevance_eval_result.get(
                "score"
            )
            results["answer_relevance_reasoning"] = answer_relevance_eval_result.get(
                "reasoning"
            )
            print(f"  Answer Relevance Score: {results['answer_relevance_score']}")

        except Exception as e:
            print(f"Error during evaluation for query '{query}': {e}")
            results["status"] = "Evaluation Error"
            results["faithfulness_reasoning"] = f"Evaluation failed: {e}"
            results["answer_relevance_reasoning"] = f"Evaluation failed: {e}"

        return results

    def run_evaluation_suite(self, test_queries: List[str]) -> pd.DataFrame:
        """
        Runs the RAG agent and evaluates it across a suite of test queries.
        """
        if not os.path.exists(self.chroma_db_dir) or not os.listdir(self.chroma_db_dir):
            print(
                f"Error: ChromaDB directory '{self.chroma_db_dir}' is empty or does not exist."
            )
            print(
                "Please run `python ingest.py` first to build the knowledge base before evaluating."
            )
            return pd.DataFrame()

        all_results = []
        for query in tqdm(test_queries, desc="Running evaluations"):
            result = self.evaluate_query(query)
            all_results.append(result)

        df = pd.DataFrame(all_results)
        return df


if __name__ == "__main__":
    # Example Test Queries (In a real scenario, load from a JSON/CSV dataset)
    config = ConfigurationManager()
    knowledge_base_config = config.get_knowledge_base_config()
    chroma_db_dir = knowledge_base_config["CHROMA_DB_DIR"]

    test_queries = [
        # "What is the warranty period for the QuantumFlow QF-2025",
        "What is the storage capacity of the QuantumFlow?",
        "what is the probable cause and solutions for strange taste ot odor in purified water",
        # "what is the probable cause and solutions for water leakage",
    ]

    evaluator = RAGEvaluator(chroma_db_dir)
    evaluation_df = evaluator.run_evaluation_suite(test_queries)

    print("\n--- Evaluation Summary ---")
    print(
        evaluation_df[
            ["query", "status", "faithfulness_score", "answer_relevance_score"]
        ]
    )

    # Optionally save results to CSV
    output_filename = "rag_evaluation_results.csv"
    evaluation_df.to_csv(output_filename, index=False)
    print(f"\nDetailed evaluation results saved to {output_filename}")

    # Display average scores for successful runs
    successful_runs = evaluation_df[evaluation_df["status"] == "Success"]
    if not successful_runs.empty:
        avg_faithfulness = successful_runs["faithfulness_score"].mean()
        avg_answer_relevance = successful_runs["answer_relevance_score"].mean()
        print(f"\nAverage Faithfulness Score (Successful Runs): {avg_faithfulness:.2f}")
        print(
            f"Average Answer Relevance Score (Successful Runs): {avg_answer_relevance:.2f}"
        )
    else:
        print("\nNo successful runs to calculate average scores.")

    print(
        "\n\nRemember to check your traces and evaluation results on LangSmith: https://smith.langchain.com/projects"
    )
