"""
Standalone local environment wrapper for testing without the HTTP server.
Uses the same core logic as server/datacenter_environment.py
"""
from server.datacenter_environment import DataCenterEnvironment, grade
from models import DataCenterAction


class DataCenterEnv:
    """Simple synchronous wrapper around DataCenterEnvironment for local testing."""

    def __init__(self):
        self._env = DataCenterEnvironment()

    def reset(self, difficulty: str = "medium"):
        return self._env.reset(difficulty=difficulty)

    def step(self, cooling_level: int, workload_distribution: str,
             power_source: str, defer_non_critical: bool):
        action = DataCenterAction(
            cooling_level=cooling_level,
            workload_distribution=workload_distribution,
            power_source=power_source,
            defer_non_critical=defer_non_critical
        )
        return self._env.step(action)

    def state(self):
        return self._env.state

    def grade(self):
        return grade(self._env.state, self._env.state.difficulty)