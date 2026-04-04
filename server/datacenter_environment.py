import random
import uuid
from typing import Optional
from openenv.core.env_server import Environment
from models import DataCenterAction, DataCenterObservation, DataCenterState


class DataCenterEnvironment(Environment):
    """
    Sustainable Data Center RL Environment.

    PHYSICS MODEL:
    - Each rack generates heat proportional to utilization
    - Cooling actively reduces temperature each step
    - Racks only fail if temperature stays critically high for multiple steps
    - PUE = total_power / it_power (realistic range 1.1 - 2.5)
    - Carbon emissions depend purely on power source choice
    """

    NUM_RACKS = 12
    MAX_STEPS = 72

    # Realistic temperature thresholds
    TEMP_SAFE     = 45.0   # normal operating range
    TEMP_WARN     = 60.0   # warning — reduce workload
    TEMP_CRITICAL = 75.0   # throttle servers
    TEMP_FAILURE  = 85.0   # rack shuts down

    DIFFICULTY_SETTINGS = {
        "easy":   {"workload_spike": 0.1, "weather_change": 0.1, "outside_temp_max": 25},
        "medium": {"workload_spike": 0.2, "weather_change": 0.2, "outside_temp_max": 35},
        "hard":   {"workload_spike": 0.3, "weather_change": 0.3, "outside_temp_max": 42},
    }

    # Carbon intensity kg CO2 per kWh
    CARBON_INTENSITY = {
        "solar":  0.00,
        "wind":   0.00,
        "hybrid": 0.12,
        "grid":   0.45,
    }

    # Cooling efficiency per level
    # level 1 = barely cooling, level 5 = aggressive cooling
    COOLING_EFFECT = {1: -2, 2: 0, 3: 4, 4: 8, 5: 14}  # degrees removed per step

    # Cooling power consumption in kW per level
    COOLING_POWER = {1: 2, 2: 5, 3: 10, 4: 18, 5: 30}

    WORKLOAD_BY_TIME = {
        "morning":   0.55,
        "afternoon": 0.85,
        "evening":   0.65,
        "night":     0.25,
    }

    def __init__(self):
        super().__init__()
        self._state = DataCenterState(episode_id=str(uuid.uuid4()))
        self._racks = []
        self._outside_temp = 20.0
        self._solar_avail = 0.7
        self._wind_avail  = 0.5
        self._overheat_streak = {}  # rack_id -> consecutive overheat steps

    def reset(self, difficulty: str = "medium", **kwargs) -> DataCenterObservation:
        diff = difficulty if difficulty in self.DIFFICULTY_SETTINGS else "medium"

        self._state = DataCenterState(
            episode_id=str(uuid.uuid4()),
            step=0,
            total_carbon_kg=0.0,
            total_power_kwh=0.0,
            tasks_completed=0,
            sla_violations=0,
            deferred_tasks=0,
            failed_racks=0,
            score_history=[],
            weather="sunny",
            time_of_day="morning",
            difficulty=diff,
            done=False
        )

        # All racks start healthy at room temperature
        self._racks = [
            {
                "id": f"rack_{i:02d}",
                "temperature": random.uniform(20.0, 26.0),  # safe starting temp
                "utilization": random.uniform(0.2, 0.4),    # low initial load
                "is_overheated": False,
                "is_failed": False,
            }
            for i in range(self.NUM_RACKS)
        ]

        self._overheat_streak = {f"rack_{i:02d}": 0 for i in range(self.NUM_RACKS)}
        settings = self.DIFFICULTY_SETTINGS[diff]
        self._outside_temp = random.uniform(18, settings["outside_temp_max"])
        self._solar_avail = 0.7
        self._wind_avail  = 0.5

        return self._build_observation(reward=0.0, done=False, extra={})

    def step(self, action: DataCenterAction) -> DataCenterObservation:
        if self._state.done:
            return self._build_observation(reward=0.0, done=True, extra={})

        # ── Validate & clamp action ──────────────────────────
        cooling   = max(1, min(5, int(action.cooling_level)))
        w_mode    = action.workload_distribution
        power_src = action.power_source
        defer     = bool(action.defer_non_critical)

        if w_mode not in ["balanced", "high_performance", "eco_mode"]:
            w_mode = "balanced"
        if power_src not in ["solar", "wind", "hybrid", "grid"]:
            power_src = "hybrid"

        reward = 0.0
        settings = self.DIFFICULTY_SETTINGS[self._state.difficulty]

        # ── Advance time of day ──────────────────────────────
        step_in_day = self._state.step % 24
        if   step_in_day < 6:  self._state.time_of_day = "night"
        elif step_in_day < 12: self._state.time_of_day = "morning"
        elif step_in_day < 18: self._state.time_of_day = "afternoon"
        else:                  self._state.time_of_day = "evening"

        # ── Update weather realistically ──────────────────────
        self._update_weather(settings)

        # ── Incoming workload ────────────────────────────────
        base = self.WORKLOAD_BY_TIME[self._state.time_of_day]
        spike = random.uniform(-settings["workload_spike"], settings["workload_spike"])
        incoming_workload = max(0.1, min(1.0, base + spike))

        # Workload multiplier by mode
        wl_mult = {"eco_mode": 0.6, "balanced": 1.0, "high_performance": 1.4}[w_mode]
        effective_workload = min(1.0, incoming_workload * wl_mult)

        # ── Process each rack ────────────────────────────────
        tasks_this_step = 0
        sla_violations  = 0
        active_racks    = 0
        total_it_power  = 0.0

        cooling_effect = self.COOLING_EFFECT[cooling]

        for rack in self._racks:
            rid = rack["id"]

            if rack["is_failed"]:
                # Failed rack — slowly cooling down, might recover
                rack["temperature"] = max(20.0, rack["temperature"] - 3.0)
                continue

            active_racks += 1

            # ── Update utilization ───────────────────────────
            rack["utilization"] = max(0.05, min(1.0,
                rack["utilization"] * 0.6 + effective_workload * 0.4
                + random.uniform(-0.03, 0.03)
            ))

            # ── Temperature physics ──────────────────────────
            # Heat generated by computation
            heat_from_compute = rack["utilization"] * 30.0  # max 30°C rise at 100% load

            # Ambient heat from outside
            ambient_heat = max(0, (self._outside_temp - 20) * 0.15)

            # Cooling removes heat
            # cooling_effect is positive = degrees removed per step
            net_temp_change = (heat_from_compute + ambient_heat - cooling_effect) * 0.15

            rack["temperature"] = max(18.0, rack["temperature"] + net_temp_change + random.uniform(-0.5, 0.5))

            # ── Overheat detection ────────────────────────────
            if rack["temperature"] >= self.TEMP_CRITICAL:
                rack["is_overheated"] = True
                self._overheat_streak[rid] = self._overheat_streak.get(rid, 0) + 1
            else:
                rack["is_overheated"] = False
                self._overheat_streak[rid] = 0

            # ── Rack failure — ONLY after 3+ consecutive critical steps ──
            if self._overheat_streak.get(rid, 0) >= 3:
                rack["is_failed"] = True
                self._state.failed_racks += 1
                self._overheat_streak[rid] = 0
                reward -= 8.0
                sla_violations += 2
                continue

            # ── IT power consumption (realistic: 0.5-5kW per rack) ──────
            it_power = 0.5 + rack["utilization"] * 4.5
            total_it_power += it_power

            # ── Count tasks completed ────────────────────────
            if not rack["is_overheated"]:
                tasks_this_step += int(rack["utilization"] * 12)

        # ── Deferred tasks logic ─────────────────────────────
        if defer and self._state.time_of_day in ["afternoon"]:
            deferred_now = int(incoming_workload * 8)
            self._state.deferred_tasks += deferred_now
            reward += 0.8  # good decision to defer during peak

        elif not defer and self._state.deferred_tasks > 0:
            # Process deferred tasks during off-peak
            process = min(self._state.deferred_tasks, 15)
            tasks_this_step += process
            self._state.deferred_tasks = max(0, self._state.deferred_tasks - process)

        if self._state.deferred_tasks > 60:
            sla_violations += 1
            reward -= 0.5

        self._state.tasks_completed += tasks_this_step
        self._state.sla_violations  += sla_violations

        # ── Power calculations ────────────────────────────────
        cooling_power_kw  = self.COOLING_POWER[cooling]
        overhead_power_kw = active_racks * 0.3  # lighting, networking etc
        total_power_kw    = total_it_power + cooling_power_kw + overhead_power_kw

        # PUE = total facility power / IT equipment power
        pue = total_power_kw / max(total_it_power, 0.1)
        pue = max(1.0, min(3.0, pue))

        # ── Renewable availability ────────────────────────────
        renewable_pct = self._get_renewable_pct(power_src)

        # ── Carbon emissions (per 30-min step = 0.5 hours) ───
        carbon_intensity = self.CARBON_INTENSITY[power_src]
        carbon_this_step = total_power_kw * 0.5 * carbon_intensity
        self._state.total_carbon_kg  += carbon_this_step
        self._state.total_power_kwh  += total_power_kw * 0.5

        # ── Reward shaping ────────────────────────────────────

        # 1. Temperature reward — keep racks in safe zone
        avg_temp = (
            sum(r["temperature"] for r in self._racks if not r["is_failed"]) /
            max(active_racks, 1)
        )
        if avg_temp < self.TEMP_SAFE:
            reward += 2.0   # perfect operating temperature
        elif avg_temp < self.TEMP_WARN:
            reward += 1.0   # acceptable
        elif avg_temp < self.TEMP_CRITICAL:
            reward -= 1.0   # getting dangerous
        else:
            reward -= 3.0   # critical — should have cooled more!

        # 2. Carbon reward — strongly incentivize renewables
        if power_src in ["solar", "wind"] and renewable_pct > 50:
            reward += 3.0   # great green choice
        elif power_src == "hybrid":
            reward += 1.5
        elif power_src == "grid":
            # Extra penalty if renewables were available
            if self._solar_avail > 0.5 or self._wind_avail > 0.5:
                reward -= 2.0  # wasted renewable opportunity!
            else:
                reward -= 0.5  # no choice but grid

        # 3. PUE reward
        if pue < 1.3:
            reward += 2.0
        elif pue < 1.6:
            reward += 1.0
        elif pue > 2.0:
            reward -= 1.0

        # 4. Task throughput reward
        reward += tasks_this_step * 0.03

        # 5. SLA violation penalty
        reward -= sla_violations * 1.5

        # 6. Overcooling penalty — using cooling=5 when temp is low wastes energy
        if cooling == 5 and avg_temp < 35:
            reward -= 1.0  # wasteful

        # ── Efficiency score ──────────────────────────────────
        temp_score    = max(0, 1.0 - max(0, avg_temp - 20) / 65)
        carbon_score  = max(0, 1.0 - carbon_this_step / 20)
        pue_score     = max(0, 1.0 - (pue - 1.0) / 2.0)
        util_score    = sum(r["utilization"] for r in self._racks if not r["is_failed"]) / max(active_racks, 1)
        efficiency    = temp_score*0.3 + carbon_score*0.3 + pue_score*0.2 + util_score*0.2

        self._state.step += 1
        self._state.step_count = self._state.step
        self._state.score_history.append(round(reward, 2))

        # ── Episode completion ────────────────────────────────
        done = False
        if self._state.step >= self.MAX_STEPS:
            done = True
            self._state.done = True
            # Bonus based on total carbon saved vs worst case
            worst_case_carbon = self.MAX_STEPS * 0.5 * 100 * 0.45  # all grid all the time
            carbon_saved = worst_case_carbon - self._state.total_carbon_kg
            bonus = max(0, carbon_saved / worst_case_carbon) * 30
            reward += bonus

        return self._build_observation(
            reward=round(reward, 2),
            done=done,
            extra={
                "power_kw":          total_power_kw,
                "it_power":          total_it_power,
                "renewable_pct":     renewable_pct,
                "carbon_step":       carbon_this_step,
                "pue":               pue,
                "efficiency":        efficiency,
                "tasks":             tasks_this_step,
                "sla_violations":    sla_violations,
                "incoming_workload": incoming_workload,
                "avg_temp":          avg_temp,
            }
        )

    def _update_weather(self, settings):
        """Realistic weather transitions."""
        change_prob = settings["weather_change"]
        if random.random() < change_prob * 0.3:
            self._state.weather = random.choice(["sunny", "sunny", "cloudy", "windy", "stormy"])

        tod = self._state.time_of_day
        if self._state.weather == "sunny":
            self._solar_avail = 0.85 if tod in ["morning", "afternoon"] else 0.0
            self._wind_avail  = random.uniform(0.2, 0.5)
        elif self._state.weather == "windy":
            self._solar_avail = 0.4 if tod in ["morning", "afternoon"] else 0.0
            self._wind_avail  = random.uniform(0.7, 1.0)
        elif self._state.weather == "cloudy":
            self._solar_avail = random.uniform(0.1, 0.3)
            self._wind_avail  = random.uniform(0.3, 0.6)
        elif self._state.weather == "stormy":
            self._solar_avail = 0.0
            self._wind_avail  = random.uniform(0.2, 0.4)

        # Outside temperature drifts slowly
        self._outside_temp += random.uniform(-1.5, 1.5)
        self._outside_temp = max(5, min(
            self.DIFFICULTY_SETTINGS[self._state.difficulty]["outside_temp_max"],
            self._outside_temp
        ))

    def _get_renewable_pct(self, power_src):
        if power_src == "solar":
            return self._solar_avail * 100
        elif power_src == "wind":
            return self._wind_avail * 100
        elif power_src == "hybrid":
            return (self._solar_avail + self._wind_avail) * 50
        else:
            return 5.0  # grid has tiny renewable mix

    @property
    def state(self) -> DataCenterState:
        return self._state

    def _build_observation(self, reward, done, extra) -> DataCenterObservation:
        active = [r for r in self._racks if not r["is_failed"]]
        avg_temp = extra.get("avg_temp",
            sum(r["temperature"] for r in active) / max(len(active), 1)
        )
        avg_util = sum(r["utilization"] for r in active) / max(len(active), 1)

        return DataCenterObservation(
            server_racks        = self._racks,
            avg_temperature     = round(avg_temp, 1),
            avg_utilization     = round(avg_util, 3),
            failed_racks        = sum(1 for r in self._racks if r["is_failed"]),
            power_consumption_kw= round(extra.get("power_kw", 0), 1),
            renewable_percentage= round(extra.get("renewable_pct", 0), 1),
            carbon_emissions_kg = round(extra.get("carbon_step", 0), 3),
            total_carbon_kg     = round(self._state.total_carbon_kg, 2),
            weather             = self._state.weather,
            solar_availability  = round(self._solar_avail, 2),
            wind_availability   = round(self._wind_avail, 2),
            outside_temp        = round(self._outside_temp, 1),
            time_of_day         = self._state.time_of_day,
            incoming_workload   = round(extra.get("incoming_workload", 0.5), 2),
            deferred_tasks      = self._state.deferred_tasks,
            tasks_completed     = self._state.tasks_completed,
            sla_violations      = self._state.sla_violations,
            pue                 = round(extra.get("pue", 1.5), 3),
            efficiency_score    = round(extra.get("efficiency", 0.5), 3),
            step_number         = self._state.step,
            done                = done,
            reward              = reward,
            success             = True
        )


# ── Grader ────────────────────────────────────────────────────
def grade(state: DataCenterState, difficulty: str = "medium") -> dict:
    steps = state.step
    if steps == 0:
        return {"score": 0.0, "grade": "F", "feedback": "Episode not started"}

    carbon   = state.total_carbon_kg
    tasks    = state.tasks_completed
    sla      = state.sla_violations
    failed   = state.failed_racks
    history  = state.score_history

    # Carbon target — how much CO2 should a well-run DC produce?
    carbon_target = {"easy": 500, "medium": 350, "hard": 200}.get(difficulty, 350)
    carbon_score  = max(0.0, 1.0 - carbon / carbon_target)

    # Task throughput — expect ~40 tasks/step across 12 racks
    expected_tasks = steps * 40
    task_score = min(tasks / max(expected_tasks, 1), 1.0)

    # Reliability — penalize failures and SLA violations
    reliability = max(0.0, 1.0 - (sla * 0.02 + failed * 0.08))

    # Consistency of rewards
    if history:
        avg_r = sum(history) / len(history)
        consistency = min(max(avg_r / 8.0, 0.0), 1.0)
    else:
        consistency = 0.0

    score = round(
        carbon_score  * 0.35 +
        task_score    * 0.25 +
        reliability   * 0.25 +
        consistency   * 0.15,
        2
    )

    if score >= 0.85:   g, f = "A", "🌱 Excellent! Near-zero carbon, fully sustainable!"
    elif score >= 0.70: g, f = "B", "✅ Good sustainability. Minor inefficiencies."
    elif score >= 0.55: g, f = "C", "⚠️ Average. Moderate carbon waste detected."
    elif score >= 0.40: g, f = "D", "❌ Poor. High emissions and reliability issues."
    else:               g, f = "F", "🔥 Critical. Unsustainable and unreliable operations."

    return {
        "score":      score,
        "grade":      g,
        "feedback":   f,
        "difficulty": difficulty,
        "stats": {
            "total_carbon_kg":   round(carbon, 2),
            "tasks_completed":   tasks,
            "sla_violations":    sla,
            "failed_racks":      failed,
            "carbon_score":      round(carbon_score, 2),
            "task_score":        round(task_score, 2),
            "reliability_score": round(reliability, 2),
            "consistency_score": round(consistency, 2),
        }
    }