"""Summarize bench_memory CSVs: steady-state RSS slope (leak rate) per run."""


def load(p):
    rows = []
    for ln in open(p):
        a = ln.strip().split(",")
        if not a[0] or a[0] == "i":
            continue
        rows.append((int(a[0]), float(a[2]), int(a[3])))
    return rows


def slope(idx, val):
    k = len(idx)
    if k < 2:
        return 0.0
    mx = sum(idx) / k
    my = sum(val) / k
    num = sum((x - mx) * (y - my) for x, y in zip(idx, val))
    den = sum((x - mx) ** 2 for x in idx)
    return num / den if den else 0.0


RUNS = [
    ("BASELINE  (1 proc, thread timeout)", "outputs/bench_baseline.csv"),
    ("FIXED w=4 (process isolation)", "outputs/bench_fixed.csv"),
    ("FIXED w=1 (process isolation)", "outputs/bench_fixed_w1.csv"),
]

for name, path in RUNS:
    r = load(path)
    idx = [x[0] for x in r]
    rss = [x[1] for x in r]
    thr = [x[2] for x in r]
    h = len(r) // 2
    s = slope(idx[h:], rss[h:])
    print(name)
    print(f"   RSS min / max     : {min(rss):.0f} / {max(rss):.0f} MiB")
    print(f"   steady slope      : {s:+.2f} MiB/sim (sims {idx[h]}-{idx[-1]})")
    print(f"   projected +10k    : {s*10000:+.0f} MiB at steady-state rate")
    print(f"   threads peak      : {max(thr)}")
    print()
