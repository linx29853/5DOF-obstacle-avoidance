import numpy as np

from so101.benchmark import verify
from so101.planners import Budget, Problem, plan


class Free:
    def __call__(self, q):
        return True

    def repulsive_potential(self, q):
        return 0.0


def test_apf_reaches_goal_and_respects_budget():
    problem = Problem(np.zeros(5), np.ones(5) * 0.5, [[-1, 1]] * 5, Free())
    result = plan(problem, "apf", Budget(max_checks=5000))
    assert verify(problem, result.path)
    assert result.stats["potential_evaluations"] > 0
    exhausted = plan(problem, "apf", Budget(max_checks=5))
    assert exhausted.path is None
    assert exhausted.stats["pose_checks"] == 5


def test_apf_reports_stationary_potential():
    class Flat(Free):
        def repulsive_potential(self, q):
            return -0.5 * np.sum((q - 0.5) ** 2)

    problem = Problem(np.zeros(5), np.ones(5) * 0.5, [[-1, 1]] * 5, Flat())
    result = plan(problem, "apf", Budget(max_checks=5000))
    assert result.path is None
    assert result.reason == "local_minimum"
