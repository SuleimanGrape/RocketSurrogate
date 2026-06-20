# RocketSurrogate

Surrogate modeling for rocket flight simulation — cascaded surrogates (XGBoost → Neural Network → LLM) with a closed-loop active learning framework for data-efficient rocket design exploration.

This project uses [RocketPy](https://github.com/RocketPy-Team/RocketPy) 6-DOF simulations to generate realistic rocket designs paired with their flight outcomes, trains physics surrogate models, and fine-tunes LLMs for generative design. The cascaded pipeline converts a small set of expensive simulations into an arbitrarily large synthetic corpus, and the closed-loop active learning system enables the LLM and neural surrogate to co-evolve toward better designs.



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
- RocketPy 6-DOF simulation pipeline with process-isolated workers (hard-kill timeouts, worker recycling) and **memory self-regulation** (per-worker RSS recycling + system-RAM backpressure) — survives long multi-day runs without OOM/hangs at 12–14 workers
- Resilient generation runner with incremental JSONL flushing, resume-on-restart, and RSS health logging
- Constrained parameter sampling (random, LHS, Sobol, balanced)
- Two-stage validation (pre/post simulation) with a drag-aware performance gate that skips designs certain to bust the Mach/apogee caps before the expensive solve
- **Feasibility (`within_bounds`) label** — every record is tagged computable/not-computable, and the generator captures the rejected (negative) class instead of discarding it, so the surrogate can learn what is *not* computable
- JSONL dataset generation with metadata and distribution plots
- Single-source-of-truth schema (`src/common/schema.py`) shared across simulation, XGBoost, and neural packages
- Feature engineering (aspect ratio, thrust-to-weight, ballistic coefficient, fin loading, and exact closed-form **Barrowman cg/cp/stability-margin** features shared by both model families)
- **XGBoost surrogate** (12 numeric targets, native categorical support) trained on ~38.6k computable samples: test R² ≥ 0.996 on 11/12 targets (stability margin 0.999, cg/cp 1.000)
- **`within_bounds` feasibility classifier** (XGBoost binary): test ROC-AUC ≈ 0.99, PR-AUC ≈ 0.99
- **Neural surrogate** (ResMLP + categorical embeddings, Huber loss, log1p on the heavy-tailed accel target): trains on the full computable corpus to mean test R² ≈ 0.99; canonical bundle saved to `models/neural/`
- **Differentiable surrogate + gradient-based design optimization** — the whole pipeline (raw params → engineered features incl. Barrowman → scaling → network → natural-unit metrics) is reimplemented in torch (`src/neural_surrogate/optim/`), so `d(metric)/d(design)` flows by autograd; `eval_gradients.py` scores gradient quality against ground truth (analytic Barrowman for Class-1, simulator finite-difference for Class-2)
- **Sobolev formula distillation** (`train_distill_class1.py`) — fine-tunes the surrogate on unlimited exact synthetic Barrowman data with value + derivative matching, repairing the closed-form targets' gradient fields while real-corpus replay preserves the flight-dynamics targets
- Sample-complexity tooling: one learning-curve study with `learning_curve.py --mode {surrogate,classifier,nn}`
- Train/val/test splitting and evaluation with metrics tables

**In Progress:**
- Large-scale dataset generation (consolidated corpus ~62k records = ~38.6k computable + ~23.5k not-computable; a 30,000-valid run is underway)
- Neural surrogate training (pipeline rebuilt for feature parity with XGBoost; trains on the ROCm machine)
- Closed-loop active learning loop implementation

**Planned:**
- LLM fine-tuning via LoRA/QLoRA on AMD GPU (ROCm)
- Bayesian optimisation baseline comparison
- Multi-objective active learning
- Open-source release of all code and synthetic datasets

## Computational Requirements

RocketPy simulations take 0.1–60+ seconds per design depending on motor class and apogee. Generation runs with up to 14 process-isolated parallel workers on a 16-core CPU; the self-regulating pool recycles workers on RSS growth and throttles dispatch under system-RAM pressure, so worker count is a throughput knob rather than a safety knob. The resilient runner (`run_with_monitor.py`) flushes incrementally and resumes after interruption, so large runs scaling to tens of thousands of samples can proceed in stages. Neural surrogate training requires a GPU — install ROCm-compatible PyTorch from `requirements-rocm.txt` (latest is torch 2.9.1 + rocm6.4, Linux-only) on an AMD machine, or a CUDA wheel (cu128+ for Blackwell) on NVIDIA. LLM fine-tuning targets a single AMD GPU with a $100 cloud budget.

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Generate Synthetic Data

Resilient runner (recommended for large runs — incremental flush, resume, health logging). It captures both the computable (`within_bounds=true`) and rejected (`within_bounds=false`) classes; `--exclude` de-duplicates against prior runs:

```bash
python run_with_monitor.py --count 30000 --workers 14 --seed 2030 \
    --output outputs/rocket_data_30k_s2030.jsonl \
    --exclude outputs/rocket_data_full.jsonl
```

Or the bare generator for quick runs (positives only):

```bash
python -m src.rocket_sim.generator --count 2000 --method random --workers 6 --oversample 3.0 --output outputs/rocket_data.jsonl
```

### Consolidate Runs

Merge multiple runs into one deduplicated corpus (exact + near-duplicate passes):

```bash
python consolidate_dataset.py --inputs outputs/run_a.jsonl outputs/run_b.jsonl \
    --output outputs/rocket_data_full.jsonl --near-dist 0
```

### Train XGBoost Surrogate + Feasibility Classifier

```bash
python src/gbt/train.py --data outputs/rocket_data_full.jsonl --no-tune \
    --output-dir models/surrogate --plots-dir plots/surrogate
python src/gbt/train_classifier.py --data outputs/rocket_data_full.jsonl \
    --output-dir models/classifier --plots-dir plots/classifier
```

### Train Neural Surrogate

Trains on CPU in minutes (a GPU only pays off at the millions-of-rows scale of formula distillation). `--save-dir` writes a portable, self-contained bundle (weights + config + scalers + metadata) mirroring `models/surrogate/`:

```bash
python src/neural_surrogate/train_surrogate.py --data outputs/rocket_data_full.jsonl \
    --model resmlp --loss huber --lr 5e-4 --train-frac 0.70 --val-frac 0.15 --test-frac 0.15 \
    --save-dir models/neural --device auto
```

### Gradient-Based Design Optimization

Score the surrogate's gradient quality against ground truth (analytic Barrowman for cg/cp/stability; add `--with-sim` to finite-difference the simulator for the flight-dynamics targets):

```bash
python src/neural_surrogate/eval_gradients.py --bundle models/neural \
    --data outputs/rocket_data_full.jsonl --n-class1 400 --with-sim --n-sim 3
```

Repair the closed-form targets' gradients with Sobolev (value + derivative) distillation on exact synthetic data — scales to millions of rows / the $100 AMD GPU via `--synth` / `--device`:

```bash
python src/neural_surrogate/train_distill_class1.py --bundle models/neural \
    --data outputs/rocket_data_full.jsonl --synth 150000 --epochs 8 \
    --save-dir models/neural_distilled
```

### Sample-Complexity Studies

```bash
python learning_curve.py --mode surrogate  --inputs outputs/rocket_data_full.jsonl --out-dir outputs/learning_curve
python learning_curve.py --mode classifier --inputs outputs/rocket_data_full.jsonl --out-dir outputs/classifier_curve
python learning_curve.py --mode nn         --inputs outputs/rocket_data_full.jsonl --device auto
```

### Run Timing Benchmark

```bash
python tools/run_ten_rockets.py --seed 42
```

## Project Structure

```
RocketSurrogate/
├── README.md                     # This file
├── ROCKET.md                     # Detailed technical documentation
├── requirements.txt
├── run_with_monitor.py           # Resilient generation runner (flush/resume/health log, captures within_bounds=false)
├── run_generation.ps1            # Convenience launcher for run_with_monitor.py
├── run_sim_and_train.py          # End-to-end simulation → training
├── consolidate_dataset.py        # Merge + dedup multiple runs into one corpus
├── backfill_within_bounds.py     # Stamp within_bounds=true on legacy data files
├── learning_curve.py             # Sample-complexity study: --mode {surrogate,classifier,nn}
├── tools/                        # Benchmarks & diagnostics
│   ├── run_ten_rockets.py        #   Timing benchmark
│   ├── bench_compare.py          #   Benchmark comparison
│   └── bench_memory.py           #   Memory benchmark
├── src/
│   ├── common/                   # Shared single-source-of-truth package (pure numpy)
│   │   ├── schema.py             # Canonical input/target fields, encodings, cardinalities
│   │   ├── scalers.py            # StandardScaler / MinMaxScaler
│   │   └── dataio.py             # Tolerant JSONL loader
│   ├── rocket_sim/               # Data generation pipeline
│   │   ├── config.py             # Parameter ranges, motor specs, validation bounds
│   │   ├── parameters.py         # Sampling strategies (random, LHS, Sobol, balanced)
│   │   ├── rocket_builder.py     # RocketPy Rocket/Motor construction
│   │   ├── simulator.py          # Single-flight simulation
│   │   ├── gen_worker.py         # Process-isolated worker pool (hard-kill timeouts, recycling)
│   │   ├── generator.py          # Streaming JSONL generation (resume-capable)
│   │   ├── validator.py          # Two-stage validation (pre/post simulation)
│   │   ├── outputs.py            # Output extraction and JSONL serialization
│   │   ├── utils.py              # CG estimation, Barrowman CP, stability margin
│   │   ├── splitter.py           # Train/val/test splitting
│   │   └── plotter.py            # Distribution plots
│   ├── gbt/                      # XGBoost surrogate models
│   │   ├── data_loader.py        # JSONL loading (label-aware: regression vs classification)
│   │   ├── preprocess.py         # Feature engineering (incl. Barrowman) + scaling
│   │   ├── synthetic_data.py     # Synthetic corpus generation via gen_worker
│   │   ├── model.py              # XGBoost training (categorical support)
│   │   ├── evaluate.py           # Metrics, plots, feature importance
│   │   ├── train.py              # Regression surrogate entry point (one model per target)
│   │   └── train_classifier.py   # within_bounds feasibility classifier entry point
│   └── neural_surrogate/         # Neural network surrogate + LLM distillation
│       ├── data/dataset.py       # Label-aware loading + engineered-feature parity with gbt
│       ├── models/surrogate.py   # MLP / ResMLP / Feature-Transformer
│       ├── training/trainer.py   # Training loop (CUDA/ROCm auto-device)
│       ├── optim/                # Differentiable surrogate for gradient-based design optimization
│       │   ├── diff_features.py  #   Torch reimpl. of the 14 engineered features (incl. Barrowman)
│       │   └── diff_surrogate.py #   End-to-end differentiable raw-params → metrics wrapper
│       ├── train_surrogate.py    # Neural surrogate entry point (--loss, --save-dir bundle)
│       ├── eval_gradients.py     # Gradient-quality eval vs analytic + simulator ground truth
│       └── train_distill_class1.py  # Sobolev formula distillation for the closed-form targets
├── tests/
│   ├── test_schema.py            # Schema consistency checks
│   ├── test_diff_features.py     # Differentiable features match numpy + autograd correctness
│   └── debug_params.py
├── models/                       # Trained models (gitignored)
├── plots/                        # Evaluation plots (gitignored)
├── outputs/                      # Generated data (gitignored)
├── requirements.txt              # Core deps
├── requirements-rocm.txt         # ROCm/CUDA PyTorch + LLM deps
└── docs/
    └── paper/                    # Research paper drafts and assets
```

## Data Format

JSONL (JSON Lines), one record per line. Each record has `input` (21 design parameters) and `output`. Computable designs carry 12 numeric flight metrics plus `within_bounds: true`; not-computable designs carry only `within_bounds: false` (no numeric targets). The schema is defined once in `src/common/schema.py` and consumed by every package. See [ROCKET.md](ROCKET.md) for the full schema.

## Key Design Principles

1. **Pre-validation is essential** — Catching invalid parameters before simulation prevents ODE solver hangs. ~67% of randomly sampled designs pass.
2. **Motor-to-diameter matching** — Impossible motor/body combinations are excluded at the sampling stage.
3. **Mass-ratio constraints** — Body length is bounded per (diameter, motor) to ensure realistic mass ratios.
4. **Stability-constrained fin geometry** — Barrowman equations determine valid fin span ranges during sampling (0.5–4.0 caliber stability margin).
5. **Cheap performance gate before simulation** — A closed-form drag-aware boost+coast estimate rejects designs whose predicted Mach/apogee clearly exceed the caps, removing the dominant source of wasted simulations. Thresholds are calibrated against real outcomes for zero false-rejections (verified 0/5000 on the seed-2026 run).
6. **Process isolation + memory self-regulation** — Each simulation runs in a separate worker with a hard-kill timeout; the pool also recycles workers on RSS growth and throttles dispatch under system-RAM pressure, so a leaking sim cannot stall or OOM a multi-day run.
7. **Capture the negative class** — Rejected designs are saved as `within_bounds=false` rather than discarded, so a feasibility classifier can tell the downstream LLM which designs are not computable.
8. **Single source of truth for the schema** — `src/common/schema.py` defines all input/target fields and encodings once; the simulator, XGBoost, and neural packages import it rather than redefining their own.
9. **Feature parity across model families** — XGBoost and the neural surrogate share the same engineered features (incl. closed-form Barrowman cg/cp/stability), so accuracy comparisons isolate the model, not the inputs.
10. **Closed-loop co-evolution** — The LLM proposer and neural surrogate evaluator improve together through iterative active learning.

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
