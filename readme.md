# AGOTMVC

**Adaptive Granular-Ball and Optimal-Transport Enhanced Multi-View Clustering**

## Repository Structure
```text
AGOTMVC/
├── train.py
│   └── Main training entry for one dataset
│
├── run_pipeline.py
│   └── Automatic dataset discovery and multi-dataset training pipeline
│
├── ablation.py
│   └── Automated ablation experiments on Cora
│
├── ablation.md
│   └── Commands for the full model and individual ablations
│
├── metric.py
│   └── Clustering evaluation and representation selection
│
├── dataset/
│   └── dataloader.py
│       └── Dynamic MAT/NPY multi-view dataset loader
│
├── networks/
│   └── network.py
│       ├── Encoder / Decoder
│       ├── Private and noise representation modules
│       └── Multi-head cross-view attention
│
├── utils/
│   ├── granular.py
│   │   └── Adaptive granular-ball generation
│   ├── granular_loss.py
│   │   ├── Optimal-transport matching
│   │   ├── Weighted multi-positive contrastive loss
│   │   ├── Uncertainty weighting
│   │   └── Temporal prototype memory / drift loss
│   ├── train_epoches.py
│   │   ├── SplitExtract
│   │   └── ViewsFusion
│   ├── loss.py
│   │   └── Auxiliary L2 and orthogonality losses
│   └── tools.py
│       └── Utility functions and logging
│
├── data/                 # User-provided datasets
├── models/               # Generated checkpoints and result JSON files
├── logs/                 # Training logs
├── tSNE/                 # Exported features for visualization
└── ablation_results/     # Archived ablation results
```

The last five directories are data/output directories and may be created automatically during execution.

---

## Environment

Python **3.10+** is recommended.

The current execution path mainly depends on:

```text
PyTorch
NumPy
SciPy
scikit-learn
h5py
```

A minimal installation is:

```bash
pip install numpy scipy scikit-learn h5py
pip install torch
```

Please install the PyTorch build appropriate for your CUDA environment.

The previous development environment of this project used approximately:

```text
numpy   1.23.5
torch   2.4.1 + CUDA 12.4
```

Exact versions are not strictly hard-coded by the current implementation.

---

## Dataset Preparation

Create a directory named `data` under the repository root:

```text
AGOTMVC/
└── data/
    ├── WebKB.mat
    ├── Cora.mat
    ├── ...
    └── <your_dataset>.mat
```

### Automatic dataset discovery

`dataset/dataloader.py` automatically discovers supported `.mat` files inside `data/`.

The dataset name passed to the training script is simply the filename without `.mat`.

For example:

```text
data/WebKB.mat  ->  WebKB
data/Cora.mat   ->  Cora
```

You can check all currently detected datasets with:

```bash
python run_pipeline.py
```

## Quick Start

### 1. List available datasets

```bash
python run_pipeline.py 
```

### 2. Train one dataset

Recommended:

```bash
python run_pipeline.py --only WebKB
```

You can also invoke the training script directly:

```bash
python train.py --dataset WebKB
```

### 3. Train selected datasets

```bash
python run_pipeline.py --only WebKB Cora
```

### 4. Train every discovered dataset

```bash
python run_pipeline.py
```

### 5. Resume a multi-dataset experiment

Existing successful result files can be skipped:

```bash
python run_pipeline.py --skip_completed
```

Arguments not belonging to `run_pipeline.py` are automatically forwarded to `train.py`.

For example:

```bash
python run_pipeline.py \
    --only Cora \
    --ori_epochs 100 \
    --epochs 100 \
    --iteration 4 \
    --batch_size 256
```

---

## Default Training Configuration

Important defaults in `train.py` are:

| Parameter          | Default | Description                                |
| ------------------ | ------: | ------------------------------------------ |
| `backbone`         |    `AE` | Autoencoder backbone (`AE` or `DAE`)       |
| `ori_epochs`       |     100 | Split-Extract Stage epochs                 |
| `epochs`           |     100 | VFS epochs per iteration                   |
| `iteration`        |       4 | Number of VFS training intervals           |
| `batch_size`       |     256 | Training batch size                        |
| `learning_rate`    |  `5e-5` | Adam learning rate                         |
| `weight_decay`     |   `0.0` | Adam weight decay                          |
| `feature_dim`      |     256 | Encoder representation dimension           |
| `high_feature_dim` |      64 | Projected/private representation dimension |
| `temperature`      |     0.2 | Contrastive temperature                    |
| `lambda1`          |     0.3 | Noise/private orthogonality weight         |
| `lambda2`          |    0.01 | L2 regularization weight                   |

With the default configuration, VFS performs:

```text
epochs × iteration = 100 × 4 = 400 epochs
```

in addition to the 100 SES epochs.

For the current attention implementation, using a `high_feature_dim` that divides `feature_dim` is recommended.

---

## Granular-Ball and OT Parameters

The full configuration uses:

| Parameter                   | Default |
| --------------------------- | ------: |
| `gb_min_samples`            |       8 |
| `gb_max_balls`              |      64 |
| `gb_min_gain`               |    0.08 |
| `gb_min_separation`         |    0.20 |
| `ot_epsilon`                |    0.10 |
| `ot_iters`                  |      30 |
| `ot_semantic_weight`        |    0.55 |
| `ot_overlap_weight`         |    0.40 |
| `gb_positive_topk`          |       3 |
| `gb_positive_rel_threshold` |    0.25 |
| `gb_drift_weight`           |    0.05 |
| `gb_drift_warmup`           |       5 |
| `gb_memory_size`            |      32 |
| `gb_memory_momentum`        |    0.95 |

The remaining OT cost weight is automatically assigned to the granular-radius discrepancy term.

---

## Full Model

The default full granular-learning configuration is:

```bash
python run_pipeline.py \
    --only Cora \
    --gb_generation adaptive \
    --matching ot \
    --positive_mode multi \
    --use_uncertainty 1 \
    --gb_drift_weight 0.05
```

This corresponds to:

```text
Adaptive Granular Balls
+ Optimal Transport
+ Weighted Multi-Positive Learning
+ Uncertainty-Aware Weighting
+ Temporal Drift Control
```

---

## Ablation Studies

The implementation exposes each major granular-learning component through command-line switches.

### Without Adaptive Granular-Ball Generation

Replace adaptive granular balls with K-Means partitions:

```bash
python run_pipeline.py \
    --only Cora \
    --gb_generation kmeans \
    --matching ot \
    --positive_mode multi \
    --use_uncertainty 1 \
    --gb_drift_weight 0.05
```

### Without Optimal Transport

Replace OT soft correspondence with hard matching:

```bash
python run_pipeline.py \
    --only Cora \
    --gb_generation adaptive \
    --matching hard \
    --positive_mode multi \
    --use_uncertainty 1 \
    --gb_drift_weight 0.05
```

### Without Weighted Multi-Positive Learning

Use one strict positive granular ball:

```bash
python run_pipeline.py \
    --only Cora \
    --gb_generation adaptive \
    --matching ot \
    --positive_mode single \
    --use_uncertainty 1 \
    --gb_drift_weight 0.05
```

### Without Uncertainty Weighting

```bash
python run_pipeline.py \
    --only Cora \
    --gb_generation adaptive \
    --matching ot \
    --positive_mode multi \
    --use_uncertainty 0 \
    --gb_drift_weight 0.05
```

### Without Drift Control

```bash
python run_pipeline.py \
    --only Cora \
    --gb_generation adaptive \
    --matching ot \
    --positive_mode multi \
    --use_uncertainty 1 \
    --gb_drift_weight 0
```

The automated script can be launched with:

```bash
python ablation.py
```

**Note:** in the current version of `ablation.py`, the `Full` experiment is commented out. The script automatically executes the five component-removal experiments on **Cora** and stores their result files in:

```text
ablation_results/
```

Run the full-model command separately if a complete Full-vs-Ablation comparison is required.

---

## Evaluation Protocol

The implementation reports four clustering metrics:

* **ACC** — clustering accuracy after Hungarian assignment;
* **NMI** — normalized mutual information;
* **ARI** — adjusted Rand index;
* **PUR** — clustering purity.

During evaluation, the model extracts four representation families:

```text
H
Z
LHZ
HHZ
```

For each representation family:

1. every view is evaluated independently;
2. all views are concatenated and evaluated as a multi-view representation;
3. K-Means is used for clustering;
4. MiniBatchKMeans is used automatically for datasets containing more than 10,000 samples.

The current implementation selects the multi-view representation with the highest **ACC**, with NMI and ARI used as tie breakers.

The result JSON additionally records:

```text
multi   -> selected multi-view clustering result
single  -> best single-view clustering result
mean    -> selected mean single-view result
```

### Model checkpoint

```text
models/<dataset>_best.pth
```

### Dataset result

```text
models/<dataset>_results.json
```

The JSON file stores:

```text
dataset information
training configuration
ablation configuration
granular-ball parameters
optimal-transport parameters
drift-control parameters
multi/single/mean clustering metrics
checkpoint metadata
elapsed training time
```

### Pipeline outputs

When using `run_pipeline.py`:

```text
models/pipeline_status.json
models/pipeline_summary.json
models/pipeline_failure.json
models/pipeline_logs/
```

`pipeline_failure.json` is created when a dataset fails, together with the corresponding console output and log location.

### Training logs

```text
logs/
```
