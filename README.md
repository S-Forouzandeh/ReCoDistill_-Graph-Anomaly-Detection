# ReCoDistill: Resolving the Fidelity-Awareness Dilemma in Graph Anomaly Detection via Co-Evolutionary Contrastive Distillation

<p align="center">
  <img src="https://img.shields.io/badge/python-3.8%2B-green?style=flat-square" />
  <img src="https://img.shields.io/badge/pytorch-2.0%2B-orange?style=flat-square" />
  <img src="https://img.shields.io/badge/license-MIT-lightgrey?style=flat-square" />
</p>

> **Saman Forouzandeh, Kamal Berahmand, Mahdi Jalili**  
> School of Engineering, RMIT University, Melbourne, Australia  
> *Under review at IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)*

---

## Overview

ReCoDistill is a unified graph anomaly detection (GAD) framework that resolves a fundamental tension in knowledge distillation — the **Fidelity-Awareness Dilemma** — through three tightly integrated, theoretically grounded components:

| Component | Description | Paper Reference |
|---|---|---|
| **Multi-Scale Structured Perturbations** | Variance-scaled node noise, similarity-weighted edge flipping, spectral graph rewiring | Eqs. 3–6 |
| **Bidirectional Co-Evolutionary Distillation** | MAML-inspired meta-gradient path reshapes teacher geometry via student feedback | Eqs. 7–13 |
| **Progressive Curriculum via Adaptive Checkpoint Selection** | Compatibility–complexity criterion annealed by student capacity κt | Eq. 14 |

At inference time, the teacher is **discarded entirely**. Only the lightweight student Sϕ is deployed, scoring nodes via dual reconstruction and Mahalanobis signals aggregated to graph level via learned attention.

```
Phase 1  →  Teacher pre-training + checkpoint collection
Phase 2  →  Co-evolutionary distillation (bidirectional InfoNCE + FOMAML meta-update)
Phase 3  →  Student-only inference  [teacher discarded]
```

---

## The Fidelity-Awareness Dilemma

A teacher trained exclusively on clean graphs produces faithful representations of normal structure — but a student that merely imitates it cannot learn to recognize anomalies. Conversely, exposing the teacher to anomalous data during pre-training corrupts the clean representations that make distillation valuable.

**Theorem 1** (proved in paper) shows this is *structurally irreducible* under any frozen-teacher distillation scheme. ReCoDistill escapes this class via a meta-gradient path that actively reshapes the teacher's embedding geometry during distillation, provably widening the anomaly embedding margin by Ω(β₀Δ²cn / (τ(Δcn + B))) over any unidirectional baseline (Theorem 2).

---

## Results Summary

Performance on 14 benchmark datasets spanning node-, edge-, and graph-level tasks.

### Node / Edge-Level (AUROC %)

| Method | Reddit | Weibo | Amazon | Yelp | T-Finance |
|---|---|---|---|---|---|
| UniGAD-GCN | 71.7 | 86.1 | 80.9 | 84.1 | 82.7 |
| SCRD4AD | 73.5 | 87.5 | 86.1 | 87.3 | 85.1 |
| DiffGAD | 56.3 | 88.4 | 66.4 | 71.6 | 85.9 |
| **ReCoDistill** | **81.2** | 86.3 | **88.9** | 86.3 | **87.2** |

### Graph-Level (AUROC %)

| Method | BM-MN | BM-MT | MUTAG | T-Group |
|---|---|---|---|---|
| UniGAD-GCN | 86.4 | 86.6 | 85.2 | 81.7 |
| SCRD4AD | 84.7 | 87.1 | 83.7 | 81.4 |
| DiffGAD | 85.7 | **89.7** | 84.5 | 84.8 |
| **ReCoDistill** | **89.8** | **89.7** | **86.5** | 84.8 |

### Efficiency

| Method | Inference (ms/graph) | Peak GPU Memory |
|---|---|---|
| DiffGAD | 148.6 | > 24 GB (OOM on T-Group) |
| SCRD4AD | 31.2 | 10.4 GB |
| UniGAD | 24.7 | 9.1 GB |
| **ReCoDistill** | **9.3** | **6.8 GB** |

> ReCoDistill achieves **up to 16× inference speedup** and **3× memory reduction** versus SCRD4AD, while matching or exceeding detection performance on 13 of 14 benchmarks.

---

## Installation

### Requirements

```
Python  ≥ 3.8
PyTorch ≥ 2.0
torch-geometric ≥ 2.3
scikit-learn ≥ 1.2
scipy ≥ 1.10
networkx ≥ 3.0
numpy ≥ 1.24
matplotlib ≥ 3.7   # for figure generation only
```

### Step-by-Step

```bash
# 1. Clone the repository
git clone https://github.com/S-Forouzandeh/ReCoDistill.git
cd ReCoDistill

# 2. Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate          # Linux / macOS
# venv\Scripts\activate           # Windows

# 3. Install PyTorch (choose your CUDA version at https://pytorch.org)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 4. Install PyTorch Geometric
pip install torch-geometric
pip install pyg-lib torch-scatter torch-sparse -f \
    https://data.pyg.org/whl/torch-2.0.0+cu118.html

# 5. Install remaining dependencies
pip install -r requirements.txt
```

### `requirements.txt`

```
torch>=2.0.0
torch-geometric>=2.3.0
scikit-learn>=1.2.0
scipy>=1.10.0
networkx>=3.0
numpy>=1.24.0
matplotlib>=3.7.0
psutil>=5.9.0
```

---

## Repository Structure

```
ReCoDistill/
│
├── recodistill.py              # Full model: perturbations, losses, training, inference
├── figures/
│   ├── fig2_scalability_efficiency.pdf
│   ├── fig3_hyperparameter_sensitivity.pdf
│   ├── fig4_robustness.pdf
│   ├── fig5_convergence.pdf
│   ├── fig6_representation_quality.pdf
│   └── fig7_statistical_complexity.pdf
├── plot_figures.py             # Reproduces all paper figures (Figs. 2–7)
├── synthetic_graph_datasets/   # Auto-generated .pt dataset files
├── models/                     # Saved model checkpoints per trial
├── results/
│   ├── recodistill_results.json
│   ├── prodigy_zero_shot_results.json
│   └── prodigy_comprehensive_results.json
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Quick Start

### Run Full Evaluation (Standard + Zero-Shot + Performance)

```python
from recodistill import main_with_comprehensive_evaluation
results = main_with_comprehensive_evaluation()
```

### Run Standard Evaluation Only

```python
from recodistill import main
results = main()
```

This trains and evaluates ReCoDistill on all seven synthetic benchmark configurations (BM-MS, BM-MN, BM-MT, MUTAG, MNIST0, MNIST1, T-Group) with 3 random seeds each, and saves results to `recodistill_results.json`.

### Run Zero-Shot Transfer Evaluation

```python
from recodistill import zero_shot_experiment, get_prodigy_dataset_configs
configs = get_prodigy_dataset_configs()
results = zero_shot_experiment(configs, num_trials=5)
```

### Train on Your Own Dataset

```python
from recodistill import train_recodistill_pipeline
from torch_geometric.data import Data
import torch

# Build your graph list (list of torch_geometric.data.Data objects)
# Each Data object must have: x, edge_index, y, node_label, edge_label
graph_list = [...]   # your dataset here
input_dim  = graph_list[0].x.shape[1]

trained = train_recodistill_pipeline(
    graph_list   = graph_list,
    input_dim    = input_dim,
    num_epochs   = 50,
    beta0        = 0.5,       # initial teacher regularisation weight
    tau          = 0.1,       # InfoNCE temperature
    lambda_recon = 0.1,       # reconstruction loss weight
    lambda_reg   = 1e-4,      # L2 regularisation weight
    eta_S        = 0.01,      # student learning rate
    rho_ema      = 0.99,      # EMA decay for μk, Σk statistics
    seed         = 42,
)

# Access trained components
student   = trained['student']     # deploy at inference
decoder   = trained['decoder']
attention = trained['attention']
ema_stats = trained['ema_stats']
```

### Run Anomaly Scoring Only (Inference)

```python
from recodistill import detect_anomalies, evaluate

results  = detect_anomalies(
    graph_list    = test_graphs,
    teacher       = trained['teacher'],
    student_model = trained['student'],
    decoder       = trained['decoder'],
    ema_stats     = trained['ema_stats'],
    attention     = trained['attention'],
    alpha_model   = trained['alpha_model'],
    lambda_score  = 0.5,    # balance between srecon and sdist
)
metrics = evaluate(results)
print(metrics)
# {'node_auroc': 0.xx, 'node_auprc': 0.xx, 'graph_auroc': 0.xx, ...}
```

---

## Architecture

### Model Components

| Class | Role | Paper |
|---|---|---|
| `GCNEncoder` | Multi-layer GCN teacher Tθ | §III-A |
| `StudentGCN` | Single-layer student Sϕ (h′ ≪ h) | §III-A |
| `GCNDecoder` | MLP decoder Gϕ for reconstruction | Eq. 15, 17 |
| `AlphaWeights` | Learnable fusion weights αk (softmax) | Eq. 6 |
| `AlignmentProjection` | Wproj ∈ R^{h′×h} for Compat | Def. 2 |
| `AttentionVector` | Learned u ∈ R^{h′} for graph aggregation | Eq. 20 |
| `TeacherWithCheckpoints` | Checkpoint manager C = {Tθ1,…,TθM} | Algo. 1 |
| `EMAStatistics` | Tracks {μk, Σk} via exponential moving averages | Algo. 1, lines 37–40 |

### Perturbation Operators

```python
# Eq. 3 — Variance-scaled Gaussian node noise
node_level_perturbation(data, sigma_N=0.1)

# Eq. 4 — Similarity-weighted edge flipping
edge_level_perturbation(data, p_E=0.1, tau_E=1.0)

# Eq. 5 — Spectral Laplacian rewiring (top-r eigenvectors)
graph_level_perturbation(data, r=5)
```

### Loss Functions

```
Ltotal = Lbidirect + λrecon · Lrecon + λreg ‖ϕ‖²      (Eq. 16)

Lbidirect = Σk αk [ L_student^(k) + βt · L_teacher^(k) ]  (Eq. 13)
  where βt = β0(1 − κt)                                     (Eq. 13)
        κt = 1 − L_student^(t) / L_student^(0)              (Eq. 7)

L_student^(k) = InfoNCE(H_S^(k) → clean, away from noisy)  (Eq. 11)
L_teacher^(k) = InfoNCE(H_C^(k) → student, away from noisy)(Eq. 12)

Lrecon = ‖Gϕ(HS) − H_C^(t*k)‖²_F                          (Eq. 15)
```

### Checkpoint Selection

```
t*k = argmax_i [ Compat(H_S^(k), H_C^(i,k)) − λreg · Complexity(H_C^(i,k)) · (1−κt) ]
```

where `Compat` is the Frobenius cosine similarity after projection Wproj, and `Complexity` is the spectral entropy of the checkpoint's representation.

### Anomaly Scoring (Teacher-Free Inference)

```
srecon(v) = ‖Gϕ(HS(v)) − HS(v)‖²                          (Eq. 17)
sdist(v)  = Σk αk · (H_S^(k)(v) − μk)ᵀ Σk⁻¹ (H_S^(k)(v) − μk)  (Eq. 18)
s(v)      = λ srecon(v) + (1−λ) sdist(v)                   (Eq. 19)
s(G)      = Σv wv · s(v),   wv ∝ exp(HS(v)ᵀ u)             (Eq. 20)
```

---

## Theoretical Guarantees

| Theorem | Statement | Location |
|---|---|---|
| **T1** — Unidirectional limitation | Anomaly margin is upper-bounded by a term decreasing monotonically with teacher fidelity ϵT | Appendix A |
| **T2** — Margin superiority | γ(S^bi) ≥ γ(S^uni) + Ω(β₀Δ²cn / (τ(Δcn+B))) | Appendix B |
| **T3** — Curriculum optimality | Adaptive schedule improves cumulative alignment over any fixed schedule by Ω(TΔ²sim/τ) | Appendix D |
| **T4** — SNR-optimal attention | α*k ∝ (μ_a^(k) − μ_n^(k))² / (σ²k + τ²/4) (Fisher discriminant ratio) | Appendix E |
| **T5** — Convergence | (1/T) Σt E[‖∇ϕL‖² + ‖∇θLmeta‖²] = O(1/T) | Appendix F |
| **T6** — Score equivalence | \|sstudent(v) − steacher(v)\| ≤ O(√λrecon + σnoise + τ⁻¹) | Appendix G |

---

## Hyperparameter Guide

| Hyperparameter | Default | Range Searched | Sensitivity |
|---|---|---|---|
| Temperature τ | 0.1 | {0.05, 0.1, 0.2} | ±1.7% AUROC across grid |
| Teacher weight β₀ | 0.5 | {0.3, 0.5, 0.7} | ±1.6% AUROC across grid |
| Reconstruction weight λrecon | 0.1 | {0.01, 0.1, 0.5} | ±2.5% AUROC across grid |
| EMA decay ρ | 0.99 | — | Stable for ρ ∈ [0.95, 0.999] |
| Spectral rewiring rank r | 5 | — | Low sensitivity for r ∈ [3, 10] |
| Score balance λ | 0.5 | — | Balanced combination recommended |

All three hyperparameters exhibit **broad optima** — performance rankings over baselines are preserved throughout each grid, confirming gains reflect architectural advantages rather than hyperparameter over-fitting.

---

## Reproducing Paper Figures

```bash
# Generates Figs. 2–7 as PDFs in ./figures/
python plot_figures.py
```

Figures produced:

| File | Content |
|---|---|
| `fig2_scalability_efficiency.pdf` | AUROC vs. graph size, inference latency, efficiency frontier |
| `fig3_hyperparameter_sensitivity.pdf` | Sensitivity to τ, β, λrecon |
| `fig4_robustness.pdf` | Robustness under anomaly ratio, feature noise, edge perturbation |
| `fig5_convergence.pdf` | Validation AUROC curves, loss decomposition, checkpoint quality |
| `fig6_representation_quality.pdf` | Score distributions, ROC curves, score separability gap |
| `fig7_statistical_complexity.pdf` | 95% CI bars, training time scalability, peak GPU memory |

---

## Datasets

ReCoDistill is evaluated on **14 benchmark datasets** across three task levels.

### Single-Graph Benchmarks (Node / Edge Level)

| Dataset | Nodes | Edges | Anomaly Ratio |
|---|---|---|---|
| Reddit | 10,984 | 168,016 | 3.3% |
| Weibo | 8,405 | 407,963 | 6.8% |
| Amazon | 11,944 | 4,398,392 | 6.9% |
| Yelp | 45,954 | 3,846,979 | 14.5% |
| Tolokers | 11,758 | 519,000 | 21.8% |
| Questions | 48,921 | 153,540 | 3.0% |
| T-Finance | 39,357 | 21,222,543 | 4.6% |

### Multi-Graph Benchmarks (Graph Level)

| Dataset | Graphs | Avg. Nodes | Anomaly Ratio |
|---|---|---|---|
| BM-MN | 700 | 18 | 48.9% (node), 14.3% (graph) |
| BM-MS | 700 | 14 | 32.0% (node), 14.3% (graph) |
| BM-MT | 700 | 17 | 34.5% (node), 14.3% (graph) |
| MUTAG | 2,951 | 30 | 4.8% (node), 34.4% (graph) |
| MNIST0 | 1,000 | 70 | 35.5% (node), 9.9% (graph) |
| MNIST1 | 1,000 | 70 | 35.5% (node), 11.3% (graph) |
| T-Group | 1,000 | 300 | 0.6% (node), 4.3% (graph) |

Synthetic dataset generation matching these statistics is built into the code via `generate_synthetic_anomaly_graphs()` and `get_prodigy_dataset_configs()`.

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## Contact

For questions about the paper or code, please open a GitHub issue or contact:

**Saman Forouzandeh** — `saman.forouzandeh@rmit.edu.au`  
School of Engineering, RMIT University, Melbourne, Australia

---

<p align="center">
  <sub>
    ReCoDistill · RMIT University · 2025
  </sub>
</p>
