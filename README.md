# Stealthy Low-Rate Temporal Attacks on Dynamic Transaction Graphs

Research code for studying temporal graph models and controlled edge-injection perturbations on transaction-network benchmarks. This repository is an experimental framework, not a validated real-world fraud-evasion system.

## Status and limitations

- Static GCN baseline and GCN-GRU temporal model are implemented.
- Elliptic preprocessing preserves endpoint labels, masks unknown labels, and makes an edge visible only after both endpoint transactions have been observed.
- Dynamic snapshots are cumulative and use chronological train/validation/test splits.
- Ground-truth neighbor-label features have been disabled to avoid target leakage.
- Node activation and first-seen masks are tracked per snapshot; inactive nodes do not accumulate recurrent hidden state, and labeled nodes are used for supervision/evaluation only at their first appearance.
- Feature scaling is fitted on training snapshots only, then reused without refitting on validation/test data.
- Python regression tests cover preprocessing, temporal state, attack intervention persistence, and synthetic end-to-end training/attack smoke runs.
- The attack now rejects self-loops, duplicate directed edges, inactive endpoints, sources above a training-derived degree percentile, per-node degree increases beyond a configured cap, and interventions that exceed a configured Jensen–Shannon divergence bound on degree distributions.
- Attack results report the measured degree-distribution divergence, rejected candidates, per-snapshot metrics, and the validation-selected decision threshold.
- These are explicit **degree-structure constraints**, not proof of universal stealth: they do not constrain every graph statistic, transaction attribute, or detector-specific signal.
- The default attack selects its target using model scores and node activity only; ground-truth labels are used only for post-hoc reporting. An explicit `oracle_highest_score_illicit` mode remains available for oracle upper-bound comparisons. Source selection never uses source labels.
- Thresholds are selected on validation predictions only and frozen before test evaluation. A multi-seed runner reports mean and standard deviation.
- Historical metrics from previous runs should not be treated as results from the updated code. Rerun experiments after pulling this branch.

## Dataset

The pipeline expects the Elliptic Bitcoin transaction dataset files placed locally under either `data/raw/` or `data/raw/elliptic_bitcoin_dataset/`:

- `elliptic_txs_edgelist.csv`
- `elliptic_txs_classes.csv`
- `elliptic_txs_features.csv`

The dataset is not included in this repository. The Elliptic benchmark does not provide transaction amounts in the edge list; the current loader uses a documented placeholder amount of 1.0. Do not interpret amount-based features or amount-similarity claims as meaningful on this dataset until real amounts are available.

## Setup

Python 3.10–3.12 is supported by CI.

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install pytest
```

The default configuration uses CPU for portability. The graph convolution uses directed degree normalization with source out-degree and target in-degree. If CUDA is available, set `project.device: cuda` in `configs/default.yaml`.

## Run

Run commands from the repository root.

```bash
python scripts/run_week1.py
python scripts/train_baseline.py --config configs/default.yaml
python scripts/train_dynamic.py --config configs/default.yaml
python scripts/run_attack.py --config configs/default.yaml

# Optional: full multi-seed experiments (compute-intensive; default seeds 42,43,44)
python scripts/run_multiseed.py --config configs/default.yaml --seeds 42,43,44
```

The dataset must be present locally before the pipeline can run. Training and attack scripts expect graph snapshots under `data/dynamic_graphs/`; checkpoints, metrics, and plots are written under `outputs/`.

## Tests

```bash
python -m compileall -q src scripts tests
pytest -q
```

GitHub Actions runs compilation checks and the test suite on pushes and pull requests with Python 3.10, 3.11, and 3.12. Tests include metrics and split edge cases, Elliptic loader and causal timing checks, training-only scaling, inactive-node state, attack-edge persistence/recomputation, and synthetic end-to-end smoke runs for baseline, temporal training, and attack execution. They do not replace a full dataset-backed experiment.

## Evaluation protocol

- Split snapshots chronologically, never randomly across time.
- Select checkpoints by validation PR-AUC (falling back to validation F1 if PR-AUC is undefined). Calibrate the classification threshold by validation F1 and freeze it before test evaluation.
- Preserve hidden-state history when evaluating later time windows. Primary test metrics use newly observed labeled nodes only; all-active pooled metrics are recorded separately as a transductive secondary view.
- Report F1, precision, recall, ROC-AUC, and PR-AUC; ROC-AUC and PR-AUC are undefined for single-class evaluation sets and are recorded as `null`.
- Report seeds, class distributions, dataset provenance, and configuration with any published results.
- Run `scripts/run_multiseed.py` and report mean and standard deviation across seeds; compare clean and attacked cases before making robustness claims.

## Research limitations

The Elliptic dataset is historical and labels are benchmark ground truth, not necessarily labels available to a live detector at transaction time. Any feature using labels must be treated as supervision only, never as a model input. Simulated edge injection is not evidence of evasion against a deployed financial institution. The simulator enforces a per-window edge budget and configured local structural constraints, including a Jensen–Shannon divergence cap on degree distributions. This is a narrow operational definition of structural similarity, not proof of stealth against all graph statistics or a real detector. The default target selection is label-free; if the optional target-label oracle mode is used, it must be reported separately as an oracle upper bound. Elliptic has no transaction amounts in the provided edge list, so the loader uses a placeholder amount; amount-based claims are not meaningful on this dataset. Validate the updated dataset-backed pipeline and attacker-knowledge assumptions before making research claims.

## License

See the repository license file, if present, for reuse terms.
