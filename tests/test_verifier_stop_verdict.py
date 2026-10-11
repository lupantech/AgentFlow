from unittest.mock import Mock

import pytest
from agentflow.models.memory import Memory
from agentflow.models.verifier import Verifier
from agentflow.solver import Solver


@pytest.fixture
def verifier():
    # Parsing does not need either of the model clients created by __init__.
    return Verifier.__new__(Verifier)


@pytest.mark.parametrize(
    "response",
    [
        "We should not stop until another source is checked.",
        "The stopwatch measurement still needs verification.",
        "Continue gathering evidence; stopping now would be premature.",
        "The prompt mentions STOP, but the evidence is incomplete.",
        "Conclusion: uncertain. We should not stop yet.",
        "",
    ],
)
def test_explanatory_prose_cannot_stop_the_agent(verifier, response):
    assert verifier.extract_conclusion(response) == (response, "CONTINUE")


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        ("STOP", "STOP"),
        (" stop. ", "STOP"),
        ("**STOP**", "STOP"),
        ("`CONTINUE`", "CONTINUE"),
        ("Enough evidence.\nConclusion: STOP", "STOP"),
        ("Do not stop yet.\nConclusion: CONTINUE", "CONTINUE"),
        ("Conclusion: STOP\nCorrection:\nConclusion: CONTINUE", "CONTINUE"),
        ('{"analysis": "Sufficient evidence", "stop_signal": true}', "STOP"),
        ('{"analysis": "Need another source", "stop_signal": false}', "CONTINUE"),
    ],
)
def test_explicit_and_structured_verdicts_are_preserved(verifier, response, expected):
    assert verifier.extract_conclusion(response)[1] == expected


def test_solver_keeps_searching_after_an_inconclusive_verifier_response(verifier):
    planner = Mock()
    planner.available_tools = ["search"]
    planner.toolbox_metadata = {"search": {}}
    planner.extract_context_subgoal_and_tool.return_value = ("context", "get evidence", "search")
    planner.generate_direct_output.return_value = "answer"
    executor = Mock()
    executor.extract_explanation_and_command.return_value = ("analysis", "explanation", "query")
    executor.execute_tool_command.return_value = "evidence"
    verifier.verificate_context = Mock(side_effect=[
        "We should not stop until another source is checked.",
        "Conclusion: STOP",
    ])
    solver = Solver(
        planner, verifier, Memory(), executor,
        output_types="direct", max_steps=3, verbose=False,
    )

    result = solver.solve("A question that needs two sources")

    assert result["step_count"] == 2
    assert len(result["memory"]) == 2
    assert executor.execute_tool_command.call_count == 2
    assert result["direct_output"] == "answer"
