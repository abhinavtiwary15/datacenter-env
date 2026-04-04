"""
OpenEnv Client for the Sustainable Data Center RL Environment.

Usage:
    import asyncio
    from client import DataCenterEnv

    async def main():
        async with DataCenterEnv(base_url="https://abhinavtiwary-datacenter-env.hf.space") as env:
            obs = await env.reset(difficulty="hard")
            while not obs.observation.done:
                # Smart agent: prioritize renewables and temperature control
                o = obs.observation
                cooling = 5 if o.avg_temperature > 60 else 3
                power = "solar" if o.solar_availability > 0.5 else "wind" if o.wind_availability > 0.5 else "hybrid"
                obs = await env.step(DataCenterAction(
                    cooling_level=cooling,
                    workload_distribution="balanced",
                    power_source=power,
                    defer_non_critical=o.time_of_day == "afternoon"
                ))
                print(f"Step {o.step_number} | Temp: {o.avg_temperature}°C | Carbon: {o.total_carbon_kg}kg")

    asyncio.run(main())
"""

from openenv.core.env_client import EnvClient
from openenv.core.client_types import StepResult
from models import DataCenterAction, DataCenterObservation, DataCenterState


class DataCenterEnv(EnvClient[DataCenterAction, DataCenterObservation, DataCenterState]):
    """WebSocket client for the Sustainable Data Center RL Environment."""

    def _step_payload(self, action: DataCenterAction) -> dict:
        return {
            "cooling_level":        action.cooling_level,
            "workload_distribution": action.workload_distribution,
            "power_source":         action.power_source,
            "defer_non_critical":   action.defer_non_critical,
        }

    def _parse_result(self, payload: dict) -> StepResult[DataCenterObservation]:
        obs_data = payload.get("observation", {})
        obs = DataCenterObservation(
            server_racks         = obs_data.get("server_racks", []),
            avg_temperature      = obs_data.get("avg_temperature", 25.0),
            avg_utilization      = obs_data.get("avg_utilization", 0.5),
            failed_racks         = obs_data.get("failed_racks", 0),
            power_consumption_kw = obs_data.get("power_consumption_kw", 0.0),
            renewable_percentage = obs_data.get("renewable_percentage", 0.0),
            carbon_emissions_kg  = obs_data.get("carbon_emissions_kg", 0.0),
            total_carbon_kg      = obs_data.get("total_carbon_kg", 0.0),
            weather              = obs_data.get("weather", "sunny"),
            solar_availability   = obs_data.get("solar_availability", 0.5),
            wind_availability    = obs_data.get("wind_availability", 0.5),
            outside_temp         = obs_data.get("outside_temp", 20.0),
            time_of_day          = obs_data.get("time_of_day", "morning"),
            incoming_workload    = obs_data.get("incoming_workload", 0.5),
            deferred_tasks       = obs_data.get("deferred_tasks", 0),
            tasks_completed      = obs_data.get("tasks_completed", 0),
            sla_violations       = obs_data.get("sla_violations", 0),
            pue                  = obs_data.get("pue", 1.5),
            efficiency_score     = obs_data.get("efficiency_score", 0.5),
            step_number          = obs_data.get("step_number", 0),
            done                 = payload.get("done", False),
            reward               = payload.get("reward", 0.0),
            success              = True,
        )
        return StepResult(
            observation = obs,
            reward      = payload.get("reward", 0.0),
            done        = payload.get("done", False),
        )

    def _parse_state(self, payload: dict) -> DataCenterState:
        return DataCenterState(
            episode_id      = payload.get("episode_id", ""),
            step            = payload.get("step", 0),
            total_carbon_kg = payload.get("total_carbon_kg", 0.0),
            total_power_kwh = payload.get("total_power_kwh", 0.0),
            tasks_completed = payload.get("tasks_completed", 0),
            sla_violations  = payload.get("sla_violations", 0),
            failed_racks    = payload.get("failed_racks", 0),
            deferred_tasks  = payload.get("deferred_tasks", 0),
            weather         = payload.get("weather", "sunny"),
            time_of_day     = payload.get("time_of_day", "morning"),
            difficulty      = payload.get("difficulty", "medium"),
            score_history   = payload.get("score_history", []),
            done            = payload.get("done", False),
        )