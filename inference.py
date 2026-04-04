"""
Inference Script — Sustainable Data Center RL Environment
=========================================================
Meta x Scaler OpenEnv Hackathon 2026

STDOUT FORMAT:
  [START] task=<task> env=<benchmark> model=<model>
  [STEP]  step=<n> action=<action> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...>
"""

import asyncio
import os
import json
import textwrap
from typing import List, Optional
import requests
from openai import OpenAI

# ── Config ────────────────────────────────────────────────────
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "hf_placeholder")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
ENV_URL      = os.getenv("ENV_URL", "https://abhinavtiwary-datacenter-env.hf.space")

TASK_NAME  = os.getenv("DC_TASK", "sustainable-datacenter-control")
BENCHMARK  = os.getenv("DC_BENCHMARK", "datacenter-env")
DIFFICULTY = os.getenv("DC_DIFFICULTY", "medium")

MAX_STEPS             = 72
TEMPERATURE           = 0.3
MAX_TOKENS            = 100
SUCCESS_SCORE_THRESHOLD = 0.5

SYSTEM_PROMPT = textwrap.dedent("""
    You are an AI agent managing a sustainable data center.
    
    Your goals (in priority order):
    1. Prevent server overheating (avg temp must stay below 65°C)
    2. Minimize carbon emissions (prefer solar/wind over grid)
    3. Maintain SLA (complete tasks on time)
    4. Optimize PUE (Power Usage Effectiveness — lower is better)
    
    Each step you must respond with a JSON object:
    {
      "cooling_level": <1-5>,
      "workload_distribution": "<eco_mode|balanced|high_performance>",
      "power_source": "<solar|wind|hybrid|grid>",
      "defer_non_critical": <true|false>
    }
    
    Rules:
    - If avg_temperature > 70: set cooling_level to 5
    - If solar_availability > 0.6: use solar
    - If wind_availability > 0.6: use wind  
    - During afternoon with high workload: defer_non_critical = true
    - Never use grid if renewables > 50% available
    
    Reply with ONLY the JSON object, no explanation.
""").strip()


def log_start(task, env, model):
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step, action, reward, done, error=None):
    err = error if error else "null"
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={err}", flush=True)

def log_end(success, steps, score, rewards):
    rstr = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rstr}", flush=True)


def build_prompt(step, obs, last_reward, history):
    return textwrap.dedent(f"""
        Step {step}/{MAX_STEPS} | Time: {obs.get('time_of_day')} | Weather: {obs.get('weather')}
        
        Server Status:
        - Avg Temperature: {obs.get('avg_temperature')}°C
        - Avg Utilization: {(obs.get('avg_utilization',0)*100):.1f}%
        - Failed Racks: {obs.get('failed_racks')}
        - PUE: {obs.get('pue')}
        
        Energy:
        - Solar availability: {(obs.get('solar_availability',0)*100):.0f}%
        - Wind availability: {(obs.get('wind_availability',0)*100):.0f}%
        - Current renewable %: {obs.get('renewable_percentage')}%
        - Total carbon so far: {obs.get('total_carbon_kg')} kg
        
        Workload:
        - Incoming workload: {(obs.get('incoming_workload',0)*100):.0f}%
        - Tasks completed: {obs.get('tasks_completed')}
        - SLA violations: {obs.get('sla_violations')}
        - Deferred tasks: {obs.get('deferred_tasks')}
        
        Last reward: {last_reward:.2f}
        
        What action do you take?
    """).strip()


def smart_fallback(obs):
    """Rule-based fallback if LLM fails."""
    cooling = 3
    if obs.get('avg_temperature', 30) > 70: cooling = 5
    elif obs.get('avg_temperature', 30) > 55: cooling = 4
    elif obs.get('avg_temperature', 30) < 30: cooling = 2

    if obs.get('solar_availability', 0) > 0.6: power = 'solar'
    elif obs.get('wind_availability', 0) > 0.6: power = 'wind'
    elif obs.get('solar_availability', 0) > 0.3 or obs.get('wind_availability', 0) > 0.3: power = 'hybrid'
    else: power = 'grid'

    if obs.get('avg_temperature', 30) > 65 or obs.get('failed_racks', 0) > 2:
        workload = 'eco_mode'
    elif obs.get('incoming_workload', 0.5) > 0.8:
        workload = 'high_performance'
    else:
        workload = 'balanced'

    defer = obs.get('time_of_day') == 'afternoon' and obs.get('incoming_workload', 0) > 0.7

    return {
        "cooling_level": cooling,
        "workload_distribution": workload,
        "power_source": power,
        "defer_non_critical": defer
    }


def get_action(client, step, obs, last_reward, history):
    try:
        prompt = build_prompt(step, obs, last_reward, history)
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
        # Parse JSON response
        if '{' in text:
            text = text[text.index('{'):text.rindex('}')+1]
        action = json.loads(text)
        # Validate
        action['cooling_level'] = max(1, min(5, int(action.get('cooling_level', 3))))
        if action.get('workload_distribution') not in ['eco_mode','balanced','high_performance']:
            action['workload_distribution'] = 'balanced'
        if action.get('power_source') not in ['solar','wind','hybrid','grid']:
            action['power_source'] = 'hybrid'
        action['defer_non_critical'] = bool(action.get('defer_non_critical', False))
        return action
    except Exception as e:
        print(f"[DEBUG] LLM failed: {e}", flush=True)
        return smart_fallback(obs)


def env_reset():
    r = requests.post(f"{ENV_URL}/reset", json={"difficulty": DIFFICULTY}, timeout=30)
    r.raise_for_status()
    return r.json()

def env_step(action):
    r = requests.post(f"{ENV_URL}/step", json=action, timeout=30)
    r.raise_for_status()
    return r.json()

def env_grade():
    r = requests.get(f"{ENV_URL}/grade", timeout=30)
    r.raise_for_status()
    return r.json()


async def main():
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    rewards, history = [], []
    steps_taken, score, success = 0, 0.0, False
    last_reward, obs = 0.0, {}

    log_start(TASK_NAME, BENCHMARK, MODEL_NAME)

    try:
        data = env_reset()
        obs = data.get("observation", {})
        done = False

        for step in range(1, MAX_STEPS + 1):
            if done: break

            action = get_action(client, step, obs, last_reward, history)
            action_str = f"cooling={action['cooling_level']},workload={action['workload_distribution']},power={action['power_source']},defer={action['defer_non_critical']}"
            error = None

            try:
                result = env_step(action)
                obs = result.get("observation", {})
                reward = float(result.get("reward", 0.0))
                done = bool(result.get("done", False))
            except Exception as e:
                reward, done, error = 0.0, False, str(e)

            rewards.append(reward)
            steps_taken = step
            last_reward = reward

            log_step(step, action_str, reward, done, error)

            history.append(
                f"Step {step}: cooling={action['cooling_level']} "
                f"power={action['power_source']} → reward {reward:+.2f} "
                f"temp={obs.get('avg_temperature','?')}°C "
                f"carbon={obs.get('carbon_emissions_kg','?')}kg"
            )

            if done: break

        try:
            g = env_grade()
            score = float(g.get("score", 0.0))
        except:
            max_r = MAX_STEPS * 8.0
            score = min(max(sum(rewards) / max_r, 0.0), 1.0)

        score = min(max(score, 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD

    finally:
        log_end(success, steps_taken, score, rewards)


if __name__ == "__main__":
    asyncio.run(main())