# RocketSurrogate

Surrogate modeling for rocket flight simulation — cascaded surrogates (XGBoost → Neural Network → LLM) with a closed-loop active learning framework for data-efficient rocket design exploration.

This project uses [RocketPy](https://github.com/RocketPy-Team/RocketPy) 6-DOF simulations to generate realistic rocket designs paired with their flight outcomes, trains physics surrogate models, and fine-tunes LLMs for generative design. The cascaded pipeline converts a small set of expensive simulations into an arbitrarily large synthetic corpus, and the closed-loop active learning system enables the LLM and neural surrogate to co-evolve toward better designs.

**Associated paper:** *"Closing the Loop: Active Exploration of Rocket Design Space with LLM Policies and Neural Surrogates"* — see [`docs/paper/`](docs/paper/) for the full draft and formatted document.

## Cascaded Surrogate Pipeline

```
RocketPy (expensive, small)
    ↓
XGBoost Teacher (accurate, discrete)
    ↓
Neural Network Student (fast, differentiable)
    ↓
LLM Fine-Tuning (generative, LoRA on AMD GPU)
    ↓
Closed-Loop Active Learning (co-evolution)
```

## Research Questions

1. **Surrogate accuracy (baseline):** How accurately does the distilled neural surrogate predict RocketPy flight outcomes?
2. **Closed-loop vs. static:** Does iterative active learning discover higher-performing designs than static LLM fine-tuning or Bayesian optimisation?
3. **Data efficiency:** How many active learning cycles are needed to reach 95% of the maximum achievable objective, and at what cost?

## Current Status

**Implemented:**
- RocketPy 6-DOF simulation pipeline with timeout protection
- Constrained parameter sampling (random, LHS, Sobol, balanced)
- Two-stage validation (pre/post simulation) — ~94% pre-validation pass rate
- JSONL dataset generation with metadata and distribution plots
- Feature engineering (aspect ratio, thrust-to-weight, ballistic coefficient, fin loading, etc.)
- XGBoost training pipeline with native categorical feature support
- Train/val/test splitting and evaluation with metrics tables

**In Progress:**
- 2000-sample balanced dataset generation (running)
- Neural surrogate training and knowledge distillation
- Closed-loop active learning loop implementation

**Planned:**
- LLM fine-tuning via LoRA/QLoRA on AMD GPU (ROCm)
- Bayesian optimisation baseline comparison
- Multi-objective active learning
- Open-source release of all code and synthetic datasets

## Computational Requirements

RocketPy simulations take 0.1–60+ seconds per design depending on motor class and apogee. The 2000-sample generation uses 6 parallel workers and takes ~2–3 hours on a 16-core CPU. Neural surrogate training requires a GPU (AMD with ROCm or CUDA). LLM fine-tuning targets a single AMD GPU with a $100 cloud budget.

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Generate Synthetic Data

```bash
python -m src.rocket_sim.generator --count 2000 --method random --workers 6 --oversample 3.0 --output outputs/rocket_data.jsonl
```

### Train XGBoost Surrogate

```bash
python -m src.gbt.train --data outputs/rocket_data.jsonl --no-tune
```

### Train Neural Surrogate

```bash
python -m src.neural_surrogate.train_surrogate
```

### Run Timing Benchmark

```bash
python run_ten_rockets.py --seed 42
```

## Project Structure

```
RocketSurrogate/
├── README.md                     # This file
├── ROCKET.md                     # Detailed technical documentation
├── requirements.txt
├── run_ten_rockets.py            # Timing benchmark
├── run_sim_and_train.py          # End-to-end simulation → training
├── src/
│   ├── rocket_sim/               # Data generation pipeline
│   │   ├── config.py             # Parameter ranges, motor specs, validation bounds
│   │   ├── parameters.py         # Sampling strategies (random, LHS, Sobol, balanced)
│   │   ├── rocket_builder.py     # RocketPy Rocket/Motor construction
│   │   ├── simulator.py          # Flight simulation with threading timeout
│   │   ├── validator.py          # Two-stage validation (pre/post simulation)
│   │   ├── outputs.py            # Output extraction and JSONL serialization
│   │   ├── utils.py              # CG estimation, Barrowman CP, stability margin
│   │   ├── splitter.py           # Train/val/test splitting
│   │   └── plotter.py            # Distribution plots
│   ├── gbt/                      # XGBoost surrogate models
│   │   ├── data_loader.py        # JSONL loading with categorical DataFrames
│   │   ├── preprocess.py         # Feature engineering + scaling
│   │   ├── model.py              # XGBoost training (categorical support)
│   │   ├── evaluate.py           # Metrics, plots, feature importance
│   │   └── train.py              # Main training entry point
│   └── neural_surrogate/         # Neural network surrogate models
├── tests/
│   └── debug_params.py
├── models/                       # Trained models (gitignored)
├── outputs/                      # Generated data (gitignored)
└── docs/
    └── paper/                    # Research paper drafts and assets
```

## Data Format

JSONL (JSON Lines), one record per line. Each record has `input` (21 design parameters) and `output` (13 flight metrics). See [ROCKET.md](ROCKET.md) for the full schema.

## Key Design Principles

1. **Pre-validation is essential** — Catching invalid parameters before simulation prevents ODE solver hangs. ~94% of randomly sampled designs pass.
2. **Motor-to-diameter matching** — Impossible motor/body combinations are excluded at the sampling stage.
3. **Mass-ratio constraints** — Body length is bounded per (diameter, motor) to ensure realistic mass ratios.
4. **Stability-constrained fin geometry** — Barrowman equations determine valid fin span ranges during sampling (0.5–4.0 caliber stability margin).
5. **Simulation timeout is mandatory** — 60-second threading-based wall-clock limit prevents hangs.
6. **Closed-loop co-evolution** — The LLM proposer and neural surrogate evaluator improve together through iterative active learning.

## Documentation

- **[ROCKET.md](ROCKET.md)** — Detailed technical docs: parameter ranges, physics model, motor specs, data schema, performance benchmarks, active learning loop design.

## Related Work

This project builds on recent advances in surrogate-assisted aerospace design:
- **Chen et al. (2025)** — Multi-fidelity neural network surrogate for rocket aerodynamic shape optimisation (30% drag reduction) [J. Phys.: Conf. Ser. 3109]
- **Wu et al. (2026)** — EEFO-KELM surrogate for sounding rocket tailfin design (12.5% drag reduction) [Aerospace Science & Technology 170]
- **Separovic & Conti (2025)** — Neural surrogates replacing legacy liquid rocket propulsion design software (10⁴× speedup) [POLITesi, Politecnico di Milano]
- **Zhang et al. (2026)** — LLM as meta-surrogate for offline many-task optimisation [Information Sciences 726]

## License

MIT