from pydantic import BaseModel, Field
from typing import Dict, List, Optional
from openenv.core.env_server.types import State


class DataCenterAction(BaseModel):
    """Agent's control decisions for one time step."""
    cooling_level: int = Field(
        ...,
        description="Cooling power level 1-5 (1=minimal, 5=maximum)",
        ge=1, le=5
    )
    workload_distribution: str = Field(
        ...,
        description="How to distribute workload: balanced/high_performance/eco_mode"
    )
    power_source: str = Field(
        ...,
        description="Primary power source: solar/wind/grid/hybrid"
    )
    defer_non_critical: bool = Field(
        ...,
        description="Whether to defer non-critical tasks to off-peak hours"
    )


class ServerRack(BaseModel):
    id: str
    temperature: float
    utilization: float
    is_overheated: bool = False
    is_failed: bool = False


class DataCenterObservation(BaseModel):
    """What the agent observes each step."""
    # Server status
    server_racks: List[Dict] = Field(description="Status of each server rack")
    avg_temperature: float = Field(description="Average temperature across all racks (Celsius)")
    avg_utilization: float = Field(description="Average CPU utilization 0.0-1.0")
    failed_racks: int = Field(description="Number of failed/overheated racks")

    # Power & energy
    power_consumption_kw: float = Field(description="Current power consumption in kW")
    renewable_percentage: float = Field(description="Percentage of power from renewables 0-100")
    carbon_emissions_kg: float = Field(description="Carbon emissions this step in kg CO2")
    total_carbon_kg: float = Field(description="Total carbon emissions this episode")

    # Environment conditions
    weather: str = Field(description="Current weather: sunny/cloudy/windy/stormy")
    solar_availability: float = Field(description="Solar power availability 0.0-1.0")
    wind_availability: float = Field(description="Wind power availability 0.0-1.0")
    outside_temp: float = Field(description="Outside temperature in Celsius")
    time_of_day: str = Field(description="morning/afternoon/evening/night")

    # Workload
    incoming_workload: float = Field(description="Incoming workload demand 0.0-1.0")
    deferred_tasks: int = Field(description="Number of deferred non-critical tasks")
    tasks_completed: int = Field(description="Total tasks completed this episode")
    sla_violations: int = Field(description="Service level agreement violations")

    # Performance metrics
    pue: float = Field(description="Power Usage Effectiveness (ideal=1.0, typical=1.5)")
    efficiency_score: float = Field(description="Overall efficiency score 0.0-1.0")
    step_number: int
    done: bool = False
    reward: float = 0.0
    success: bool = True


class DataCenterState(State):
    """Full internal state."""
    step: int = 0
    total_carbon_kg: float = 0.0
    total_power_kwh: float = 0.0
    tasks_completed: int = 0
    sla_violations: int = 0
    deferred_tasks: int = 0
    failed_racks: int = 0
    score_history: List[float] = []
    weather: str = "sunny"
    time_of_day: str = "morning"
    difficulty: str = "medium"
    done: bool = False