import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Optional
from server.datacenter_environment import DataCenterEnvironment, grade
from models import DataCenterAction

app = FastAPI(
    title="🌱 Sustainable Data Center RL Environment",
    description="""
## Sustainable Data Center Controller

AI agent learns to operate a large-scale data center while minimizing
carbon emissions and maximizing efficiency.

### Features
- 🌡️ **Temperature management** — prevent server overheating
- ⚡ **Renewable energy** — solar, wind, grid, hybrid power sources
- 🌍 **Carbon tracking** — real CO2 emissions per decision
- 💻 **Workload scheduling** — defer non-critical tasks smartly
- 🔥 **Server failure risk** — push too hard and racks fail
- 🌤️ **Dynamic weather** — affects solar/wind availability
- 📊 **PUE optimization** — Power Usage Effectiveness metric
- 3 difficulty levels — easy, medium, hard
    """,
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

env = DataCenterEnvironment()


class ResetRequest(BaseModel):
    difficulty: Optional[str] = "medium"


class StepRequest(BaseModel):
    cooling_level: int = 3
    workload_distribution: str = "balanced"
    power_source: str = "hybrid"
    defer_non_critical: bool = False


def obs_to_dict(obs):
    return {
        "server_racks": obs.server_racks,
        "avg_temperature": obs.avg_temperature,
        "avg_utilization": obs.avg_utilization,
        "failed_racks": obs.failed_racks,
        "power_consumption_kw": obs.power_consumption_kw,
        "renewable_percentage": obs.renewable_percentage,
        "carbon_emissions_kg": obs.carbon_emissions_kg,
        "total_carbon_kg": obs.total_carbon_kg,
        "weather": obs.weather,
        "solar_availability": obs.solar_availability,
        "wind_availability": obs.wind_availability,
        "outside_temp": obs.outside_temp,
        "time_of_day": obs.time_of_day,
        "incoming_workload": obs.incoming_workload,
        "deferred_tasks": obs.deferred_tasks,
        "tasks_completed": obs.tasks_completed,
        "sla_violations": obs.sla_violations,
        "pue": obs.pue,
        "efficiency_score": obs.efficiency_score,
        "step_number": obs.step_number,
        "done": obs.done,
        "reward": obs.reward
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def root():
    path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "dashboard.html")
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        content = content.replace(
            "const API = 'http://127.0.0.1:8000'",
            "const API = window.location.origin"
        )
        return HTMLResponse(content=content)
    except Exception as e:
        return HTMLResponse(f"<h1>Error loading dashboard: {e}</h1>")


@app.post("/reset")
def reset(request: ResetRequest = None):
    if request is None:
        request = ResetRequest()
    obs = env.reset(difficulty=request.difficulty or "medium")
    return {
        "observation": obs_to_dict(obs),
        "message": f"Data center episode started | Difficulty: {request.difficulty}"
    }


@app.post("/step")
def step(request: StepRequest):
    action = DataCenterAction(
        cooling_level=request.cooling_level,
        workload_distribution=request.workload_distribution,
        power_source=request.power_source,
        defer_non_critical=request.defer_non_critical
    )
    obs = env.step(action)
    response = {
        "observation": obs_to_dict(obs),
        "reward": obs.reward,
        "done": obs.done,
        "info": {}
    }
    if obs.done:
        response["final_grade"] = grade(env.state, env.state.difficulty)
    return response


@app.get("/state")
def state():
    s = env.state
    return {
        "step": s.step,
        "total_carbon_kg": s.total_carbon_kg,
        "total_power_kwh": s.total_power_kwh,
        "tasks_completed": s.tasks_completed,
        "sla_violations": s.sla_violations,
        "failed_racks": s.failed_racks,
        "deferred_tasks": s.deferred_tasks,
        "weather": s.weather,
        "time_of_day": s.time_of_day,
        "difficulty": s.difficulty,
        "score_history": s.score_history,
        "done": s.done
    }


@app.get("/grade")
def grade_env():
    return grade(env.state, env.state.difficulty)


@app.get("/dashboard", response_class=HTMLResponse, include_in_schema=False)
def dashboard():
    path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "dashboard.html")
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        content = content.replace(
            "const API = 'http://127.0.0.1:8000'",
            "const API = window.location.origin"
        )
        return HTMLResponse(content=content)
    except Exception as e:
        return HTMLResponse(f"<h1>Error: {e}</h1>")
    
@app.get("/openapi-schema", include_in_schema=False)
def openapi_schema():
    return {
        "action": {
            "type": "object",
            "properties": {
                "cooling_level": {"type": "integer", "minimum": 1, "maximum": 5},
                "workload_distribution": {"type": "string", "enum": ["eco_mode", "balanced", "high_performance"]},
                "power_source": {"type": "string", "enum": ["solar", "wind", "hybrid", "grid"]},
                "defer_non_critical": {"type": "boolean"}
            }
        },
        "observation": {"type": "object"},
        "state": {"type": "object"}
    }


@app.post("/mcp", include_in_schema=False)
async def mcp_endpoint(request: Request):
    return {
        "jsonrpc": "2.0",
        "id": 1,
        "result": {
            "tools": [
                {"name": "reset", "description": "Reset the data center environment"},
                {"name": "step", "description": "Take one control action"},
                {"name": "state", "description": "Get full environment state"},
                {"name": "grade", "description": "Get performance grade 0.0-1.0"}
            ]
        }
    }