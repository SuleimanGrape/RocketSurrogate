# RocketSurrogate — Synthetic Rocket Design Data Generator

## Overview

RocketSurrogate generates synthetic rocket design data using [RocketPy](https://github.com/RocketPy-Team/RocketPy) 6-DOF flight simulations. The project produces large-scale datasets of realistic rocket designs paired with their simulated flight outcomes, then trains a **differentiable neural surrogate** that replaces expensive simulations with fast inference. Because the surrogate is differentiable end-to-end, designs can be optimized directly by gradient ascent; the surrogate is further extended with a **closed-loop active learning framework** in which an LLM proposer and the neural surrogate evaluator co-evolve to discover high-performing rocket designs.

**Research paper:** *"Closing the Loop: Active Exploration of Rocket Design Space with LLM Policies and Neural Surrogates"* — see `docs/paper/` for the full draft and formatted document.

## Architecture

### Surrogate Pipeline

```
RocketPy 6-DOF Simulations (expensive, small)
    │     ~7s per simulation, ~30% computable
    ▼
Neural Surrogate (fast, differentiable)          ← trained directly on the corpus
    │     mean test R² ≈ 0.99, 100,000+ pred/sec  + Barrowman/Sobolev formula distillation
    ├──────────────▶ Gradient-based design optimization (autograd through the surrogate)
    │                     maximize apogee s.t. stability / Mach / accel / T-W constraints
    ▼
LLM Fine-Tuning (generative, LoRA on AMD GPU)
    │     Llama 3 7B, QLoRA, $100 budget
    ▼
Closed-Loop Active Learning (co-evolution)
          LLM proposes → NN evaluates → Select → Retrain both
```

A separate XGBoost **feasibility classifier** predicts whether a candidate design is computable at all (`within_bounds`), gating the proposer away from infeasible regions.

### Data Generation Pipeline

The data generation pipeline has five stages:

```
Sample → Pre-validate → Simulate → Post-validate → Save
```

Designs that fail pre-validation, time out, or fail post-validation are **not discarded** — the resilient runner (`run_with_monitor.py`) saves them as `within_bounds=false` records (input only, no numeric targets). Computable designs are saved as `within_bounds=true` with their full flight metrics. This yields a labelled feasibility dataset (the negative class) alongside the regression corpus. On recent balanced runs ~30% of attempted designs end up computable.

### 1. Parameter Sampling (`parameters.py`)

Three sampling strategies: random, Latin Hypercube (LHS) [McKay 1979], and Sobol sequence. A `balanced_sample()` variant ensures equal representation of discrete categories across all combinations of diameter, nose type, fin count, and motor class.

Parameters are constrained at sampling time to avoid physically impossible designs:
- Body length bounded per (diameter, motor) for realistic mass ratios (dry/propellant mass ∈ [1.5, 10.0])
- Fin span constrained via Barrowman stability equations (CP – CG)
- Motor class restricted per body diameter (e.g., motors D–G for 24 mm, D–I for 29 mm, etc.)
- Rail length proportional to rocket size
- Launch angle 85–90° (near-vertical)

### 2. Pre-validation (`validator.py`)

A fast, simulation-free pass that rejects ~94% of invalid designs before they reach the ODE solver. Checks include:

- Fin geometry validity (positive chord lengths, span < body length)
- Mass ratio bounds (dry_mass / propellant_mass ≥ 1.5)
- Thrust-to-weight ratio minimum (T/W ≥ 3.0)
- Barrowman static stability margin (0.5–4.0 calibers)
- Propellant volume constraints (fit within body tube)
- Rail length proportionality (rail ≥ 10× diameter)

### 3. Flight Simulation (`simulator.py`)

RocketPy 6-DOF flight simulation with:
- Custom atmospheric model (elevation-aware temperature and pressure)
- Synthetic thrust curves per motor class (triangular profile, peak = 1.3 × average thrust)
- 60-second wall-clock timeout enforced through threading to prevent hangs
- Rail buttons, parachute deployment at apogee, and landing dynamics

### 4. Post-validation

Filters simulation outputs for physical plausibility:
- Apogee ≤ 100 km
- Mach ≤ 5
- Max acceleration < 200 g

### 5. Output (`outputs.py`, `splitter.py`, `plotter.py`)

- JSONL serialisation (one record per line), each tagged with the `within_bounds` computability label
- Metadata recording (random seed, timing statistics, acceptance rates per stage, `within_bounds` true/false counts)
- Train/val/test splits (70/15/15, stratified)
- Summary distribution plots (histograms, pairwise scatter matrices)

## Current Status

**Implemented:**
- RocketPy 6-DOF simulation pipeline with timeout protection, process isolation, and memory self-regulation (RSS recycling + RAM backpressure)
- Constrained parameter sampling (random, LHS, Sobol, balanced)
- Two-stage validation, with the rejected (negative) class captured as `within_bounds=false`
- JSONL dataset generation with metadata, `within_bounds` labels, and distribution plots
- Burnout altitude/velocity extraction via flight.solution ODE trajectory
- Shared feature engineering (`common/features.py`): aspect ratio, motor impulse, T/W ratio, fin area ratio, nose-body ratio, slenderness, ballistic coefficient, fin loading, and exact closed-form Barrowman cg/cp/stability-margin features
- Neural surrogate (ResMLP + categorical embeddings) trained directly on the full computable corpus — mean test R² ≈ 0.99 (stability margin 0.999, cg/cp ≈ 0.999; `max_acceleration_mps2` ≈ 0.85, a heavy-tail plateau judged by MAE/MAPE)
- Differentiable surrogate (torch reimplementation incl. Barrowman) + gradient-based design optimizer with real-simulator validation
- Sobolev formula distillation + `class1_exact` hybrid for exact cg/cp/stability value **and** gradient
- `within_bounds` feasibility classifier (XGBoost binary) — test ROC-AUC ≈ 0.99, PR-AUC ≈ 0.99
- Sample-complexity studies for the neural surrogate and the classifier
- Timing benchmark suite
- Research paper draft and formatted .docx

**In Progress:**
- Large-scale dataset generation (consolidated ~62k records; a 30,000-valid run underway)
- Neural surrogate training (MLP, Residual MLP, Feature Transformer; pipeline rebuilt for feature parity, trains on the ROCm machine)
- Closed-loop active learning loop implementation

**Planned:**
- LLM fine-tuning via LoRA/QLoRA on AMD GPU (ROCm)
- Bayesian optimisation baseline comparison
- Multi-objective active learning (apogee + stability + structural constraints)
- Ensemble surrogate uncertainty estimation
- Open-source release of all code, checkpoints, and synthetic datasets

## Physics Model

### Center of Gravity

CG is computed from a seven-component mass model: nose cone, body tube, fins, recovery system, electronics/avionics, motor casing, and propellant. The model is body-length-aware — short rockets receive proportionally more mass in the nose/payload section to maintain realistic CG positions. CG shifts during flight as propellant is consumed (linear depletion assumed).

### Center of Pressure

CP is computed using the Barrowman equations: nose cone contribution (shape-dependent, different factors for conical, tangent ogive, von Karman, elliptical) and fin contribution (trapezoidal geometry with body interference effects).

### Stability Margin

Static margin = (CP – CG) / body diameter, measured in calibers. Target range: **0.5–4.0 calibers**. Below 0.5: unstable (prone to weathercocking). Above 4.0: overstable (excessive drag, poor altitude performance).

## Data Format

Records are stored as JSONL, one per line:

```json
{
  "input": {
    "diameter_mm": 54,
    "length_m": 2.800,
    "nose_type": "von_karman",
    "nose_length_m": 0.450,
    "fin_count": 4,
    "fin_root_chord_m": 0.2200,
    "fin_tip_chord_m": 0.1100,
    "fin_span_m": 0.1200,
    "fin_sweep_m": 0.0800,
    "fin_thickness_mm": 6.0,
    "dry_mass_kg": 5.400,
    "motor_class": "J",
    "propellant_mass_kg": 1.2000,
    "burn_time_s": 3.50,
    "avg_thrust_N": 1500.0,
    "wind_speed_mps": 4.1,
    "wind_direction_deg": 180.0,
    "elevation_m": 300.0,
    "temperature_c": 20.0,
    "rail_length_m": 3.0,
    "launch_angle_deg": 89.0
  },
  "output": {
    "apogee_m": 4218.0,
    "max_velocity_mps": 318.0,
    "max_mach": 0.920,
    "max_acceleration_mps2": 124.0,
    "burnout_altitude_m": 682.0,
    "burnout_velocity_mps": 185.0,
    "time_to_apogee_s": 18.0,
    "stability_margin_calibers": 1.80,
    "rail_exit_velocity_mps": 22.5,
    "max_dynamic_pressure_pa": 45000.0,
    "cg_m": 1.4500,
    "cp_m": 1.5500,
    "motor_class": "J",
    "within_bounds": true
  }
}
```

A not-computable design carries only the label and no numeric targets:

```json
{ "input": { "...": "..." }, "output": { "within_bounds": false } }
```

## Input Parameters

### Discrete Choices

| Parameter | Options |
|-----------|---------|
| Body diameter | 24, 29, 38, 54, 75, 98 mm |
| Nose type | Conical, Tangent Ogive, Von Karman, Elliptical |
| Fin count | 3, 4 |
| Motor class | D, E, F, G, H, I, J, K, L, M |

### Continuous Parameters

| Parameter | Range | Unit |
|-----------|-------|------|
| Body length | 0.5 – 6.0 | m |
| Nose length | 0.5 – 5.0 × diameter | m |
| Fin root chord | 1.0 – 2.5 × diameter | m |
| Fin tip chord | 20 – 100% of root chord | m |
| Fin span | 0.5 – 2.0 × diameter | m |
| Fin sweep | 0 – root chord | m |
| Fin thickness | 2.0 – 12.0 | mm |
| Dry mass | k × d² × L, k ∈ [200, 600] | kg |
| Propellant mass | Per motor class specs | kg |
| Burn time | Per motor class specs | s |
| Average thrust | Per motor class specs | N |
| Elevation | 0 – 3000 | m |
| Temperature | -10 – 40 | °C |
| Wind speed | 0 – 15 | m/s |
| Wind direction | 0 – 360 | ° |
| Rail length | 1.0 – 8.0 | m |
| Launch angle | 85 – 90 | ° |

## Output Metrics

| Metric | Description | Unit |
|--------|-------------|------|
| `apogee_m` | Maximum altitude above launch | m |
| `max_velocity_mps` | Maximum speed during flight | m/s |
| `max_mach` | Maximum Mach number | — |
| `max_acceleration_mps2` | Maximum acceleration | m/s² |
| `burnout_altitude_m` | Altitude at motor burnout | m |
| `burnout_velocity_mps` | Velocity at motor burnout | m/s |
| `time_to_apogee_s` | Time from launch to apogee | s |
| `stability_margin_calibers` | Static margin (CP – CG) / diameter | calibers |
| `rail_exit_velocity_mps` | Speed at end of launch rail | m/s |
| `max_dynamic_pressure_pa` | Maximum aerodynamic dynamic pressure | Pa |
| `cg_m` | Center of gravity from nose tip | m |
| `cp_m` | Center of pressure from nose tip (Barrowman) | m |

These 12 metrics are the regression targets. In addition, every record carries a
`within_bounds` boolean (the binary classification label) and a `motor_class`
passthrough echo.

> The canonical input/output field lists and categorical encodings live in
> `src/common/schema.py` (the single source of truth, verified by
> `tests/test_schema.py`). The simulation terminates at apogee, so there is no
> `landing_velocity_mps`, and `flight_time_s` was dropped because it is identical
> to `time_to_apogee_s` (the flight ends at apogee). `time_to_apogee_s` is a
> modelled target.

## Motor Specifications

| Class | Propellant (kg) | Burn time (s) | Avg thrust (N) |
|-------|-----------------|---------------|----------------|
| D | 0.012 – 0.025 | 1.2 – 2.0 | 18 – 35 |
| E | 0.025 – 0.050 | 1.3 – 2.2 | 35 – 65 |
| F | 0.050 – 0.100 | 1.5 – 2.5 | 65 – 130 |
| G | 0.100 – 0.200 | 1.8 – 3.0 | 130 – 260 |
| H | 0.200 – 0.400 | 2.0 – 3.5 | 260 – 520 |
| I | 0.400 – 0.800 | 2.2 – 4.0 | 520 – 1,100 |
| J | 0.800 – 1.600 | 2.5 – 4.5 | 1,100 – 2,200 |
| K | 1.600 – 3.200 | 3.0 – 5.0 | 2,200 – 4,400 |
| L | 3.200 – 6.400 | 3.5 – 6.0 | 4,400 – 8,800 |
| M | 6.400 – 12.80 | 4.0 – 7.0 | 8,800 – 17,600 |

## Closed-Loop Active Learning

The closed-loop active learning framework is the core innovation described in the research paper. It enables the LLM and neural surrogate to co-evolve over multiple cycles:

### Loop Steps

1. **Propose (Step A):** The current LLM generates B = 100 rocket design texts with temperature 0.8. Each text is parsed into structured parameter vectors.

2. **Evaluate (Step B):** The neural surrogate predicts flight metrics for all 100 designs in < 1 ms (100,000+ predictions/sec on GPU).

3. **Select (Step C):** Designs are ranked by objective (e.g., predicted apogee). Top K = 10 are retained. 5% of retained designs are optionally validated with RocketPy to detect surrogate drift.

4. **Retrain (Step D):** Selected designs join the training set. The neural surrogate is fine-tuned for 5–10 epochs (reduced LR). The LLM is fine-tuned via LoRA for 1 epoch on the selected designs as preferred completions.

5. **Loop (Step E):** Repeat for N = 20 cycles or until the objective plateaus (ε = 0.01 over 3 cycles).

### Surrogate Error Amplification Mitigation

If the neural surrogate overestimates performance in unexplored regions, the selector preferentially retains those designs, reinforcing bias. Two mitigations:
- **RocketPy validation:** 5% of selected designs run through RocketPy. If mean prediction error > 20%, pause and retrain.
- **Ensemble uncertainty:** Multiple neural surrogates with different seeds; reject designs with high prediction variance.

### Comparison Baselines

- **Static LLM:** Fine-tuned once on NN-generated data, then generates 1,000 designs without retraining.
- **Bayesian Optimisation:** Uses the neural surrogate as the objective with expected improvement acquisition (20 iterations).

Expected results from the paper: the active loop achieves ~41% higher apogee than static LLM and ~19% higher than Bayesian optimisation, reaching 95% of maximum objective in ~8 cycles.

## Usage

### Generate Data

Resilient runner (recommended — incremental flush, resume, health log, and it captures the `within_bounds=false` negative class):

```bash
python run_with_monitor.py --count 30000 --workers 14 --seed 2030 \
    --output outputs/rocket_data_30k_s2030.jsonl --exclude outputs/rocket_data_full.jsonl
```

Bare generator (positives only) and its options:

```bash
python -m src.rocket_sim.generator --count 2000 --method random --workers 6 --output outputs/rocket_data.jsonl
```

- `--method {random,lhs,sobol}` — sampling strategy
- `--no-balanced` — disable balanced category sampling
- `--workers N` — parallel simulation workers
- `--oversample F` — oversample factor for rejections
- `--splits-dir DIR` / `--plots-dir DIR` — output directories for splits / plots
- `--exclude FILES` — de-duplicate inputs against prior runs (run_with_monitor.py)

### Train the Surrogate + Classifier + Optimize

```bash
python src/neural_surrogate/train_surrogate.py --data outputs/rocket_data_full.jsonl --save-dir models/neural --device auto
python src/gbt/train_classifier.py --data outputs/rocket_data_full.jsonl             # within_bounds feasibility
python src/neural_surrogate/optim/design_optimizer.py --diameter 54 --motor K --validate   # gradient-based design optimization
```

### Sample-Complexity Studies

```bash
python learning_curve.py --mode nn         --inputs outputs/rocket_data_full.jsonl   # neural R² vs N
python learning_curve.py --mode classifier --inputs outputs/rocket_data_full.jsonl   # classifier AUC vs N
```

### Run Timing Benchmark

```bash
python tools/run_ten_rockets.py --seed 42 --output outputs/ten_rocket_results.json
```

## Performance

Based on a 10-rocket benchmark (seed=42, single-threaded) and a 100-valid-sample generation run:

| Metric | Value |
|--------|-------|
| Pre-validation pass rate | ~94% |
| Simulation success rate | 80% (8/10; 2 hit 60s timeout) |
| Mean simulation time | ~7 s |
| Median simulation time | ~7 s |
| Fastest | 0.1 s (29 mm D-class) |
| Slowest successful | 28.8 s (29 mm H-class, 12 km apogee) |
| 100-valid-sample generation | 300 sampled → 281 pre-valid → 200 simulated → 100 valid |
| Total generation time (100 valid) | 4,856 s (~81 min) |

Simulation time is bimodal: small rockets (D–G motors) finish in under a second, while high-power rockets (H–M motors) can take 7–60+ seconds depending on apogee.

## Project Structure

```
RocketSurrogate/
├── README.md                     # Project overview and quick start
├── ROCKET.md                     # This file — detailed technical documentation
├── requirements.txt
├── run_generation.ps1            # Data-generation launcher (PowerShell)
├── run_with_monitor.py           # Resilient generator: health log + resume + negative-class capture
├── consolidate_dataset.py        # Merge + dedup runs into one corpus
├── backfill_within_bounds.py     # Stamp within_bounds=true on legacy data
├── learning_curve.py             # Sample-complexity study: --mode {nn,classifier}
├── tools/                        # Timing + memory benchmarks
├── src/
│   ├── common/                   # Shared single-source-of-truth modules
│   │   ├── schema.py             # SINGLE SOURCE OF TRUTH for input/output fields
│   │   ├── features.py           # Engineered features (incl. Barrowman) + LOG1P_TARGETS, shared
│   │   ├── scalers.py            # StandardScaler / MinMaxScaler (shared)
│   │   └── dataio.py             # load_jsonl
│   ├── rocket_sim/               # Data generation pipeline
│   │   ├── config.py             # Parameter ranges, motor specs, validation bounds
│   │   ├── parameters.py         # Sampling strategies
│   │   ├── rocket_builder.py     # RocketPy Rocket/Motor construction
│   │   ├── simulator.py          # Synchronous flight solve (simulate_flight)
│   │   ├── gen_worker.py         # Process-isolated pool: hard-kill timeout + recycling
│   │   ├── generator.py          # Generation orchestrator (streaming JSONL + resume)
│   │   ├── validator.py          # Two-stage validation
│   │   ├── outputs.py            # Output extraction + JSONL serialization
│   │   ├── utils.py              # CG, Barrowman CP, stability
│   │   ├── splitter.py           # Train/val/test splits
│   │   └── plotter.py            # Distribution plots
│   ├── gbt/                      # within_bounds feasibility classifier (XGBoost)
│   │   ├── data_loader.py        # JSONL loading (label-aware: regression / classification)
│   │   └── train_classifier.py   # within_bounds feasibility classifier entry point
│   └── neural_surrogate/         # Neural surrogate + differentiable design optimization
│       ├── data/, models/, training/   # dataset, architectures, trainer
│       ├── optim/                # diff_features, diff_surrogate, design_optimizer
│       ├── train_surrogate.py    # surrogate training (--save-dir bundle)
│       ├── eval_gradients.py     # gradient-quality eval vs ground truth
│       └── train_distill_class1.py  # Sobolev formula distillation
├── tests/
│   ├── test_schema.py            # Schema consistency checks
│   ├── test_diff_features.py     # Differentiable-feature correctness
│   ├── test_design_optimizer.py  # Optimizer feasibility/improvement checks
│   └── debug_params.py
├── models/                       # Trained models (gitignored)
├── plots/                        # Evaluation plots (gitignored)
├── outputs/                      # Generated data (gitignored)
├── requirements-rocm.txt         # ROCm/CUDA PyTorch + LLM deps
└── docs/
    └── paper/                    # Research paper drafts and assets
        ├── paper-full-draft.txt
        ├── research-paper-formatted.docx
        ├── IMPLEMENTATION.md
        └── build_paper.py
```

## Key Design Principles

1. **Pre-validation is essential** — Catching invalid parameters before simulation prevents ODE solver hangs. ~94% pass rate.
2. **Motor-to-diameter matching** — Impossible motor/body combinations are excluded at the sampling stage.
3. **Mass-ratio constraints** — Body length bounded per (diameter, motor) to ensure dry_mass / propellant_mass ∈ [1.5, 10.0].
4. **Stability-constrained fin geometry** — Barrowman equations determine valid fin span ranges during sampling (0.5–4.0 cal).
5. **Process isolation + memory self-regulation** — Each sim runs in a worker with a hard-kill timeout; the pool recycles workers on RSS growth and throttles under RAM pressure, so leaks/hangs cannot crash multi-day runs.
6. **Capture the negative class** — Rejected designs are saved as `within_bounds=false` so a feasibility classifier can flag designs that are not computable.
7. **Burnout extraction from ODE trajectory** — Uses flight.solution arrays (t, y) rather than RocketPy's unreliable `.burn_out_altitude` API.
8. **Shared, differentiable feature engineering** — Domain features (ballistic coefficient, fin loading, aspect ratio, T/W) plus exact closed-form Barrowman cg/cp/stability live once in `common/features.py`; the neural surrogate and the classifier share them, and `optim/diff_features.py` reimplements them in torch so they are autograd-differentiable.
9. **Single source of truth for the schema** — `src/common/schema.py` defines all fields/encodings once; every package imports it.
10. **Closed-loop co-evolution** — The LLM proposer and neural surrogate evaluator improve together through iterative active learning, outperforming static pipelines.

## License

MIT