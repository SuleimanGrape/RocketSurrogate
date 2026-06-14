"""Analyze health CSV for degradation trends."""
import csv

rows = []
with open('outputs/rocket_data_5k_health.csv') as f:
    reader = csv.DictReader(f)
    for r in reader:
        rows.append(r)

print(f'Total health samples: {len(rows)}')
print()

for label, i in [('First', 0), ('~5min', min(60, len(rows)-1)),
                  ('Midway', len(rows)//2), ('Last', len(rows)-1)]:
    r = rows[i]
    es = r.get('elapsed_s', '?')
    print(f'--- {label} (t={es}s) ---')
    print(f'  CPU: {r["cpu_percent"]}%  RAM: {r["ram_percent"]}%  '
          f'GPU: {r["gpu_temp_c"]}C / {r["gpu_power_w"]}W')
print()

# RAM and GPU temp trends
n = len(rows)
ram_early = [float(r['ram_percent']) for r in rows[:n//10] if r.get('ram_percent')]
ram_late = [float(r['ram_percent']) for r in rows[-n//10:] if r.get('ram_percent')]
gpu_power_early = [float(r['gpu_power_w']) for r in rows[:n//10] if r.get('gpu_power_w')]
gpu_power_late = [float(r['gpu_power_w']) for r in rows[-n//10:] if r.get('gpu_power_w')]
cpu_early_vals = [float(r['cpu_percent']) for r in rows[:n//10] if r.get('cpu_percent')]
cpu_late_vals = [float(r['cpu_percent']) for r in rows[-n//10:] if r.get('cpu_percent')]

print(f'RAM: early avg={sum(ram_early)/len(ram_early):.1f}%  late avg={sum(ram_late)/len(ram_late):.1f}%')
print(f'CPU: early avg={sum(cpu_early_vals)/len(cpu_early_vals):.1f}%  late avg={sum(cpu_late_vals)/len(cpu_late_vals):.1f}%')
print(f'GPU power: early avg={sum(gpu_power_early)/len(gpu_power_early):.1f}W  late avg={sum(gpu_power_late)/len(gpu_power_late):.1f}W')

# Print some samples
print(f'\nFirst 3 entries:')
for r in rows[:3]:
    print(f'  t={r["elapsed_s"]:>6s}s  CPU={r["cpu_percent"]:>4s}%  RAM={r["ram_percent"]:>4s}%  GPU={r["gpu_temp_c"]:>2s}C / {r["gpu_power_w"]:>5s}W')

print(f'\nLast 3 entries:')
for r in rows[-3:]:
    print(f'  t={r["elapsed_s"]:>6s}s  CPU={r["cpu_percent"]:>4s}%  RAM={r["ram_percent"]:>4s}%  GPU={r["gpu_temp_c"]:>2s}C / {r["gpu_power_w"]:>5s}W')

# Check for memory growth pattern - RAM increase rate
ram_vals = [(i, float(r['ram_percent'])) for i, r in enumerate(rows) if r.get('ram_percent')]
if len(ram_vals) > 10:
    first_ram = ram_vals[0][1]
    last_ram = ram_vals[-1][1]
    print(f'\nRAM leak check: {first_ram:.1f}% -> {last_ram:.1f}% over {len(ram_vals)} samples')