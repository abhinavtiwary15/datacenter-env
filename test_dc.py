import requests

BASE = 'http://127.0.0.1:8000'

r = requests.post(f'{BASE}/reset', json={'difficulty': 'medium'})
obs = r.json()['observation']
print(f'Start: temp={obs["avg_temperature"]}C weather={obs["weather"]}')
print()

total_reward = 0
for step in range(1, 30):
    cooling = 5 if obs['avg_temperature'] > 55 else 3 if obs['avg_temperature'] > 40 else 2
    power = 'solar' if obs['solar_availability'] > 0.5 else 'wind' if obs['wind_availability'] > 0.5 else 'hybrid'
    workload = 'eco_mode' if obs['avg_temperature'] > 60 else 'balanced'
    defer = obs['time_of_day'] == 'afternoon'

    r = requests.post(f'{BASE}/step', json={
        'cooling_level': cooling,
        'workload_distribution': workload,
        'power_source': power,
        'defer_non_critical': defer
    })
    data = r.json()
    obs = data['observation']
    total_reward += data['reward']

    print(f'Step {step:02d} | temp={obs["avg_temperature"]:5.1f}C | pue={obs["pue"]:.2f} | carbon={obs["carbon_emissions_kg"]:.3f}kg | reward={data["reward"]:+.1f} | failed={obs["failed_racks"]}')

    if obs['done']:
        break

g = requests.get(f'{BASE}/grade').json()
print()
print(f'Score: {g["score"]} | Grade: {g["grade"]} | {g["feedback"]}')
print(f'Stats: {g["stats"]}')