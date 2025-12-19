from memory.session_memory import SessionMemory
from memory.contradiction_handler import ContradictionHandler

from tools.eligibility import EligibilityEngine, UserProfile, EligibilityCriteria
from tools.scheme_retriever import SchemeRetriever
from tools.mock_gov_api import MockGovAPI

from agent.state_machine import AgentStateMachine, AgentState
from agent.planner import Planner
from agent.executor import Executor
from agent.executor import ExecutionStatus


def main():
    # Memory-related imports
    memory = SessionMemory()
    checker = ContradictionHandler()

    # Tool imports
    engine = EligibilityEngine()
    retriever = SchemeRetriever()
    api = MockGovAPI()

    # State machine import check
    state_machine = AgentStateMachine(initial_state=AgentState.IDLE)

    planner = Planner()
    plan = planner.plan_next_action()

    assert plan.action is not None
    print("Planner sanity check PASSED")

    print("All imports OK (including state_machine)")

    executor = Executor()

    # Minimal valid plan (no tools required)
    plan = {
        "action": "ask_missing_info",
        "parameters": {
            "missing_fields": ["age", "income"],
            "context": "eligibility_check"
        }
    }

    result = executor.execute_plan(plan)

    # Basic structural checks
    assert result is not None, "Executor returned None"
    assert result.action == "ask_missing_info", "Wrong action in result"
    assert result.status == ExecutionStatus.SUCCESS, "Unexpected execution status"
    assert isinstance(result.output, dict), "Output must be a dictionary"

    print("✅ Executor check passed (imports, execution, structure OK)")


if __name__ == "__main__":
    main()
