"""A query can exhaust its budget before the first tool action."""

from unittest.mock import Mock

import pytest

from agentflow.models.memory import Memory
from agentflow.solver import Solver


@pytest.mark.parametrize("limits", [{"max_steps": 0}, {"max_time": 0}, {"max_time": 1}])
def test_budget_exhausted_before_first_action(monkeypatch, limits):
    planner = Mock()
    planner.analyze_query.return_value = "analysis"
    planner.generate_direct_output.return_value = "answer from available context"
    verifier = Mock()
    executor = Mock()
    solver = Solver(
        planner, verifier, Memory(), executor,
        output_types="direct", verbose=False, **limits
    )
    # Query analysis takes two seconds, exceeding even the positive time budget.
    clock = Mock(side_effect=[0.0, 2.0, 2.0, 2.0, 2.0])
    monkeypatch.setattr("agentflow.solver.time.time", clock)

    result = solver.solve("question")

    assert result["memory"] == {}
    assert result["step_count"] == 0
    assert result["direct_output"] == "answer from available context"
    planner.generate_next_step.assert_not_called()
    verifier.verificate_context.assert_not_called()
    executor.execute_tool_command.assert_not_called()
