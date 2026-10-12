# CardioTwin-FM

This repository contains the code for a longitudinal multimodal cardiac digital twin. The twin does four things:

- It assimilates asynchronous ECG, PCG, echocardiography, EHR and wearable records into a latent state, using precision weights that account for how old each record is.
- It rolls that state forward under factual exposure trajectories, and under intervened ones for counterfactual estimates.
- It produces multi-horizon discrete-time cumulative incidence of major adverse cardiovascular events.
- It calibrates horizon-specific risks into prediction sets using censoring-weighted, availability-stratified (Mondrian) conformal prediction.

Everything runs on CPU with JAX. A GPU is not required.

## Files (single flat folder)

| File | Purpose |
|---|---|
| `cardiosim.py` | Mechanistic six-site multimodal simulator with exact Monte Carlo counterfactual risks |
| `data_sim.py` | Builds the padded simulator tensors and counterfactual ground truth (`cardiosim.npz`) |
| `cohort.py` | Unified cohort containers for the simulator, the Framingham teaching cohort and the aortic-valve echo cohort |
| `nets.py` | Minimal JAX layers and modality encoders |
| `pretrain.py` | Self-supervised pretraining: masked-modality InfoNCE with cross- and intra-modal reconstruction |
| `twin.py` | CardioTwin-FM: assimilation, structural dynamics, roll-out, hazard head, soft monotonicity penalty |
| `causal.py` | Stabilised marginal-structural-model weights for treatment initiation |
| `conformal.py` | IPCW label x availability Mondrian conformal sets, with optional covariate-shift weights |
| `baselines.py` | Cox, random survival forest, DeepSurv, DeepHit, Dynamic-DeepHit and multimodal transformer |
| `features.py` | Engineered landmark and record features used by the baselines |
| `metrics.py` | Uno C-index, time-dependent AUC, (integrated) Brier score, KM calibration error, net benefit, bootstrap tests |
| `run_sim.py`, `run_real.py` | Run one (model, seed) or (model, fold) job and save its predictions |
| `analyze_all.py` | Aggregates every reported number into `results.json` |
| `fig_arch.py`, `fig_traj.py`, `fig_results.py` | Manuscript figures (grayscale) |
| `run_queue.sh`, `run_queue_real.sh`, `run_queue_conf.sh` | Full experiment queues |
| `run_all.sh` | End-to-end reproduction |

## Setup

```bash
pip install -r requirements.txt
mkdir -p ../data_raw ../runs/sim ../runs/real ../paper/figs
curl -L -o ../data_raw/framingham.rda https://raw.githubusercontent.com/cran/riskCommunicator/master/data/framingham.rda
curl -L -o ../data_raw/heart.valve.rda https://raw.githubusercontent.com/cran/joineR/master/data/heart.valve.rda
```

The Framingham file is the anonymised teaching release distributed with the riskCommunicator R package. Results from it describe the behaviour of the methods, not Framingham epidemiology. The valve cohort is the `heart.valve` dataset distributed with the joineR R package.

## Reproduce

```bash
bash run_all.sh
```

`run_all.sh` does the following in order:

1. Simulates the cohort.
2. Runs the main comparison (5 seeds), the ablations (3 seeds), the real-cohort cross-validation and the unmeasured-confounding sensitivity runs.
3. Aggregates the results.
4. Draws the figures.

On a two-core CPU the whole pipeline takes several hours.
