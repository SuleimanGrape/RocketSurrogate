# RocketSurrogate Data Quality Review — v2 Analysis

**Date:** 2026-06-10
**Review of:** 2000-sample balanced dataset v2 (`outputs/rocket_data_2k_v2.jsonl`)
**Simulation time:** 22,646s (~6.3 hours) — **47% faster** than v1 (42,821s / 11.9h)
**Previous review:** `data-quality-review.md` (v1, 2026-06-06)

---

## 1. Generation Summary: v2 vs v1

| Metric | v1 (seed=42) | v2 (seed=123) | Δ |
|---|---|---|---|
| Sampled | 6,000 | 6,000 | same |
| Pre-validated | 5,607 (93.4%) | **4,243 (70.7%)** | ↓ new constraints |
| Simulated | ~4,000 | 4,000 | same |
| **Valid** | **2,000** | **2,000** | same |
| Total time | 42,821s (11.9h) | **22,646s (6.3h)** | **47% faster** |
| Workers | 8 | 8 | same |

The pre-validation rate dropped from 93.4% to 70.7% because of the two new constraints:
- **Rail exit velocity ≥10 m/s** (estimated constant-acceleration model)
- **Slenderness ratio ≤80:1** (length/diameter)

These reject ~1,364 additional designs that would have failed post-validation or produced physically unrealistic results.

Simulation time nearly halved thanks to per-class timeouts (small motors timeout faster) and the pre-validation catching unstable designs earlier.

---

## 2. ✅ Critical Bug Fixed: Burnout Extraction

**v1 status:** 100% zeroed out — `burnout_altitude_m`, `burnout_velocity_mps`, `time_to_apogee_s` were ALL ZERO for every record.

**v2 status: ALL NON-ZERO ✓**

| Field | v1 Zero | v2 Zero | v2 Min | v2 Max | v2 Median | v2 Mean |
|---|---|---|---|---|---|---|
| `burnout_altitude_m` | 2,000/2,000 | **0/2,000** | 76.8 | 8,418 | 2,218 | 2,314 |
| `burnout_velocity_mps` | 2,000/2,000 | **0/2,000** | 40.7 | 1,738 | 255 | 405 |
| `time_to_apogee_s` | 2,000/2,000 | **0/2,000** | 5.3 | 72.2 | 19.6 | 22.3 |

**Root cause:** `_find_burnout_state()` was using `flight.solution.t` / `flight.solution.y` which don't exist in RocketPy 1.12.1. The `_safe()` wrapper silently caught the AttributeError and returned 0.0.

**Fix (applied):** Replaced with `flight.z.x_array` / `flight.z.y_array` — raw numpy arrays with no string headers.

*Note: `landing_velocity_mps` was intentionally removed — the simulation terminates at apogee, so no landing data is available.*

---

## 3. 🔶 Motor Class Balance — Improved, Further Fix Applied

### v1 Distribution (PRE-fix)

| Motor | Count | % | Issue |
|---|---|---|---|
| D | 262 | 13.1% | Over |
| E | 369 | 18.4% | Over |
| F | 307 | 15.3% | Over |
| G | 223 | 11.2% | OK |
| H | 217 | 10.8% | OK |
| I | 220 | 11.0% | OK |
| J | 246 | 12.3% | OK |
| K | 118 | 5.9% | Low |
| L | 34 | **1.7%** | **Severely low** |
| M | 4 | **0.2%** | **Severely low** |

### v2 Distribution (after balanced_sample() added motor_class as 4th dimension)

| Motor | Count | % | vs Target 10% |
|---|---|---|---|
| D | 346 | 17.3% | +7.3% |
| E | 372 | 18.6% | +8.6% |
| F | 249 | 12.4% | +2.4% |
| G | 122 | **6.1%** | **−3.9%** |
| H | 130 | **6.5%** | **−3.5%** |
| I | 195 | 9.8% | −0.2% |
| J | 239 | 11.9% | +1.9% |
| K | 175 | 8.8% | −1.2% |
| L | 117 | **5.9%** | **−4.1%** |
| M | 55 | **2.8%** | **−7.2%** |

**Analysis:**
- Major wins: L went 1.7%→5.9% (+3.5x), M went 0.2%→2.8% (+14x)
- G and H **decreased** because they primarily pair with 38mm/54mm diameters, which fail slenderness and rail-exit constraints at higher rates
- D and E are **overrepresented** because small motors on 24mm/29mm tubes easily pass all constraints

**Root cause — unequal post-validation survival rates:**

| Class | Final % | Survival Rate (final / 10% initial) |
|---|---|---|
| D | 17.3% | 1.73× |
| E | 18.6% | 1.86× |
| F | 12.4% | 1.24× |
| G | 6.1% | 0.61× |
| H | 6.5% | 0.65× |
| I | 9.8% | 0.98× |
| J | 11.9% | 1.19× |
| K | 8.8% | 0.88× |
| L | 5.9% | 0.59× |
| M | 2.8% | 0.28× |

### Fix Applied (2026-06-10)

**Problem:** `balanced_sample()` allocated motor classes equally (600 each), but post-validation disproportionately rejects G, H, L, M due to slenderness and rail-exit constraints.

**Solution — weighted allocation:**

Added `PER_CLASS_ALLOCATION_WEIGHTS` in `config.py` — weights are proportional to the inverse of observed survival rates. Classes with lower pass rates get more initial samples:

| Class | Weight | Why |
|---|---|---|
| D | 0.50 | High survival → fewer needed |
| E | 0.47 | High survival → fewer needed |
| F | 0.70 | Average survival |
| G | 1.43 | Low survival (slenderness failures) |
| H | 1.34 | Low survival (rail exit on 54mm) |
| I | 0.89 | Near-average survival |
| J | 0.73 | Above-average survival |
| K | 0.99 | Average survival |
| L | 1.49 | Low survival |
| M | 2.72 | Very low survival (9.2%) |

**Also changed:**
- Default oversample factor: 3.0 → **4.0** (more total samples to compensate for shifting budget to low-survival classes)
- Simulation cap: `count × 2` → `count × 3` (simulate more to hit 2,000 target)

**Expected result per class (n=8,000 sampled, based on v2 survival rates):**

| Class | Allocated | Expected Final |
|---|---|---|
| D | ~355 | ~205 |
| E | ~333 | ~206 |
| F | ~497 | ~206 |
| G | ~1,015 | ~206 |
| H | ~952 | ~207 |
| I | ~632 | ~205 |
| J | ~518 | ~206 |
| K | ~703 | ~205 |
| L | ~1,058 | ~206 |
| M | ~1,937 | ~178 |

M still slightly below target (178 vs 200), but dramatically improved from 55.

---

## 4. ✅ Validation Constraints Working

### Rail Exit Velocity (min 10 m/s)

| Metric | v1 | v2 | Verdict |
|---|---|---|---|
| Below 10 m/s | 155 (7.8%) | **0 (0%)** | ✅ **Fixed** |
| Below 5 m/s | 16 (0.8%) | **0 (0%)** | ✅ **Fixed** |
| Minimum value | 0.17 m/s | **10.01 m/s** | ✅ **Fixed** |
| Mean | — | 22.68 m/s | Physically realistic |

### Slenderness Ratio (max 80:1)

| Metric | v1 | v2 | Verdict |
|---|---|---|---|
| Over 80:1 | 370 (12.8%) | **0 (0%)** | ✅ **Fixed** |
| Over 100:1 | 37 (1.8%) | **0 (0%)** | ✅ **Fixed** |
| Maximum | 249.5:1 | **79.2:1** | ✅ **Fixed** |

---

## 5. 🚀 Performance Comparison

| Metric | v1 (old) | v2 (new) | Improvement |
|---|---|---|---|
| Total time | 11.9 hours | **6.3 hours** | **47% faster** |
| Workers | 8 | 8 | Same (the cap was always min(workers, cpu-2, 8) — never changed to 4) |
| Pre-val pass rate | 93.4% | 70.7% | Lower (stricter) |
| Sim→valid rate | 50% | 50% | Unchanged |

**Why 47% faster:**
1. Per-class timeouts (D=15s → M=120s instead of uniform 60s) — small motors finish faster
2. The two new pre-validation constraints catch unstable designs before they reach RocketPy, reducing simulation failures on expensive motors

---

## 6. Output Distribution Comparison

Values shifted slightly upward because the slenderness constraint eliminated very long/thin designs that produced lower-performance flights:

| Field | v1 Median | v2 Median | v1 Mean | v2 Mean | Shift |
|---|---|---|---|---|---|
| `apogee_m` | 3,686 | **3,864** | 4,574 | **5,908** | +4.8% med |
| `max_velocity_mps` | 240 | **265** | 329 | **415** | +10% med |
| `max_mach` | 0.72 | **0.78** | 0.97 | **1.23** | +8% med |
| `max_acceleration_mps2` | 136 | **150** | 173 | **202** | +10% med |
| `flight_time_s` | 19.5 | **19.6** | 20.4 | **22.3** | — |
| `max_dynamic_pressure_pa` | 30,791 | **36,360** | 93,071 | **156,484** | +18% med |

### v2 Distribution Details (for reference)

| Output | Range | Median | Mean | Heavy tail? |
|---|---|---|---|---|
| apogee_m | 197–37,997 | 3,864 | 5,908 | Yes |
| max_velocity_mps | 50.6–1,750 | 265 | 415 | Yes |
| max_mach | 0.15–4.96 | 0.78 | 1.23 | Yes |
| max_acceleration_mps2 | 48–1,974 | 150 | 202 | Yes |
| flight_time_s | 5.3–72.2 | 19.6 | 22.3 | Slight |
| max_dynamic_pressure_pa | 1,385–1.56M | 36,360 | 156,484 | Heavy |
| stability_margin_calibers | 0.62–4.00 | 2.83 | 2.77 | — |
| burnout_altitude_m | 77–8,418 | 2,218 | 2,314 | Moderate |
| burnout_velocity_mps | 41–1,738 | 255 | 405 | Yes |

---

## 7. Categorical Balance (v2)

| diameter_mm | nose_type | fin_count |
|---|---|---|
| 24mm: 16.2% | conical: 24.0% | 3: 49.5% |
| 29mm: 20.5% | ogive: 24.6% | 4: 50.4% |
| 38mm: 19.0% | von_karman: 25.2% | |
| 54mm: 9.4% | elliptical: 26.2% | |
| 75mm: 15.3% | | |
| 98mm: 19.4% | | |

---

## 8. Implementation Status

### Code Changes Applied (v1 fixes, 2026-06-06)

| Fix | File | Status |
|---|---|---|
| Burnout extraction — fixed API | `outputs.py` | ✅ Applied — uses `.x_array`/`.y_array` |
| Per-class simulation timeouts | `config.py` | ✅ Added `SIM_TIMEOUT_BY_CLASS` |
| Per-class timeout usage | `simulator.py` | ✅ Applied |
| Rail exit & slenderness validation | `validator.py` | ✅ Both checks in `is_valid()` and `prevalidate()` |
| Worker count reduction (6→4) | `generator.py` | ❌ **Not applied** — still caps at 8 (correctly so) |
| Explicit split fractions | `generator.py` | ✅ Passes 70/15/15 |
| Split defaults | `splitter.py` | ✅ Changed to 70/15/15 |

### Code Changes Applied (v2 motor balance fix, 2026-06-10)

| Fix | File | Status |
|---|---|---|
| Per-class allocation weights | `config.py` | ✅ Added `PER_CLASS_ALLOCATION_WEIGHTS` |
| Weighted motor allocation | `parameters.py` | ✅ Uses weights instead of equal split |
| Oversample 3.0 → 4.0 | `generator.py` | ✅ Default changed, simulation cap upped to `count×3` |

### What to do next

1. **Run a 500-sample smoke test** to verify the weighted allocation fix:
   ```
   python -m src.rocket_sim.generator --count 500 --workers 8 --output outputs/test_balance.jsonl
   ```
   Check that all 10 motor classes have ≥40 records (8% each).

2. **Run a full 2000-sample generation** after the smoke test passes.

3. **Post-generation checks:**
   - Verify burnout fields are non-zero (should pass — same code as v2)
   - Verify rail exit ≥10 m/s and slenderness ≤80:1 (should pass — same constraints)
   - Check motor class balance improved (target: all classes between 7-13%)

---

*See also: `IMPLEMENTATION.md`, `memory/simulation-run.md`*