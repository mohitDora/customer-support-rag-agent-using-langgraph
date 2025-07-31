from src.models import AgentState


class SupervisorAgent:
    """
    Agent responsible for synthesizing a draft answer from accumulated relevant chunks.
    """

    def run(self, state: AgentState) -> str:
        """
        The supervisor node orchestrates the flow based on the 'next_agent_to_call' in the state.
        It also checks for overall completion of research.
        """
        print(f"\n---SUPERVISOR: Directing flow to: {state['next_agent_to_call']}---")

        # If an agent signals END or FATAL_ERROR, the supervisor transitions to the graph END
        if state["next_agent_to_call"] in ["END", "FATAL_ERROR"]:
            print(
                "---SUPERVISOR: Workflow complete or fatal error detected. Ending workflow.---"
            )
            return {"next_agent_to_call": "end_workflow"}  # Transition to END

        # Otherwise, direct to the agent specified in next_agent_to_call
        return {"next_agent_to_call": state["next_agent_to_call"]}
