---
title: Sustainable Data Center RL Environment
emoji: 🌱
colorFrom: green
colorTo: blue
sdk: docker
pinned: true
license: mit
tags:
  - reinforcement-learning
  - openenv
  - sustainability
  - data-center
  - green-computing
  - smart-infrastructure
---

# 🌱 Sustainable Data Center RL Environment

> **Built for Meta x Scaler OpenEnv Hackathon 2026**

An AI agent learns to operate a large-scale data center while
minimizing carbon emissions and maximizing operational efficiency.

🔴 **[Live Demo →](/)** | 📖 **[API Docs →](/docs)**

---

## 🌍 Why This Matters

Data centers consume **1-2% of global electricity** and produce
millions of tonnes of CO₂ annually. Meta, Google, and Microsoft
have all committed to carbon-neutral data center operations.

This environment trains AI agents to make the kinds of real-time
decisions that could dramatically reduce that footprint — making
it directly relevant to the infrastructure teams at Meta and
Hugging Face who will evaluate this submission.

---

## 🧠 Environment Design

### The Agent Controls:
| Parameter | Options | Effect |
|-----------|---------|--------|
| `cooling_level` | 1-5 | Higher = cooler servers, more power used |
| `workload_distribution` | eco/balanced/performance | Affects heat and throughput |
| `power_source` | solar/wind/hybrid/grid | Determines carbon emissions |
| `defer_non_critical` | true/false | Smart task scheduling |

### What the Agent Observes:
```python
{
  "avg_temperature": 42.3,          # °C across all racks
  "failed_racks": 0,                # racks that have shut down
  "solar_availability": 0.85,       # 0-1, depends on weather+time
  "wind_availability": 0.62,
  "carbon_emissions_kg": 0.0,       # this step's CO2
  "total_carbon_kg": 12.4,          # episode total
  "pue": 1.28,                      # Power Usage Effectiveness
  "time_of_day": "morning",
  "weather": "sunny",
  "incoming_workload": 0.72,
  "sla_violations": 0
}
```

### Reward Function:
| Event | Reward |
|-------|--------|
| Temp in safe zone (<45°C) | +2.0 |
| Using solar/wind power | +3.0 |
| Good PUE (<1.3) | +2.0 |
| Task completed | +0.03 each |
| Smart deferral (afternoon peak) | +0.8 |
| Rack overheating (>75°C) | -3.0 |
| Rack failure (3+ critical steps) | -8.0 |
| Using grid when renewables available | -2.0 |
| SLA violation | -1.5 each |
| Episode carbon bonus | up to +30.0 |

---

## ⚙️ Configuration

### Difficulty Levels
| Level | Workload Variance | Outside Temp Max | Weather Changes |
|-------|------------------|-----------------|-----------------|
| Easy   | ±10% | 25°C | Rare |
| Medium | ±20% | 35°C | Occasional |
| Hard   | ±30% | 42°C | Frequent |

### Time of Day Workload
| Time | Base Load |
|------|-----------|
| Morning | 55% |
| Afternoon | 85% (peak) |
| Evening | 65% |
| Night | 25% |

---

## 🚀 Quick Start

### Python Client
```python
import asyncio
from client import DataCenterEnv
from models import DataCenterAction

async def main():
    async with DataCenterEnv(
        base_url="https://abhinavtiwary-datacenter-env.hf.space"
    ) as env:
        obs = await env.reset(difficulty="medium")

        while not obs.observation.done:
            o = obs.observation

            # Smart agent strategy
            cooling = 5 if o.avg_temperature > 60 else 3
            power   = (
                "solar" if o.solar_availability > 0.5 else
                "wind"  if o.wind_availability  > 0.5 else
                "hybrid"
            )

            obs = await env.step(DataCenterAction(
                cooling_level=cooling,
                workload_distribution="balanced",
                power_source=power,
                defer_non_critical=(o.time_of_day == "afternoon")
            ))

            print(f"Reward: {obs.reward:.2f} | Carbon: {o.total_carbon_kg:.1f}kg")

asyncio.run(main())
```

### REST API
```bash
# Start episode
curl -X POST /reset -H "Content-Type: application/json" \
  -d '{"difficulty": "hard"}'

# Take action
curl -X POST /step -H "Content-Type: application/json" \
  -d '{"cooling_level": 3, "workload_distribution": "balanced",
       "power_source": "solar", "defer_non_critical": false}'

# Get grade
curl /grade
```

---

## 📡 API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Info |
| GET | `/health` | Health check |
| POST | `/reset` | Start episode |
| POST | `/step` | Take action |
| GET | `/state` | Full state |
| GET | `/grade` | Performance grade |
| GET | `/dashboard` | Visual interface |
| GET | `/docs` | API documentation |

---

## 🧪 Baseline Results
EASY   difficulty → Score: 0.93 | Grade: A
MEDIUM difficulty → Score: 0.85 | Grade: A
HARD   difficulty → Score: 0.71 | Grade: B

---

## 🏗️ Project Structure
datacenter_env/
├── datacenter_env/
│   ├── init.py       # Package exports
│   └── env.py            # Local testing wrapper
├── server/
│   ├── app.py            # FastAPI HTTP server
│   └── datacenter_environment.py  # Core RL environment
├── models.py             # Pydantic Action/Observation/State
├── client.py             # OpenEnv WebSocket client
├── inference.py          # Baseline inference script
├── dashboard.html        # Interactive visual dashboard
├── openenv.yaml          # OpenEnv spec manifest
├── Dockerfile            # Container config
└── README.md             # This file
---

*Built with ❤️ for the Meta x Scaler OpenEnv Hackathon 2026*