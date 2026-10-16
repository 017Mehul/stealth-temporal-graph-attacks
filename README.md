# Stealthy Low-Rate Temporal Attacks on Dynamic Transaction Graphs

Research code for studying temporal graph models and controlled edge-injection perturbations on transaction-network benchmarks. This repository is an experimental framework, not a validated real-world fraud-evasion system.

## Status and limitations

- Static GCN baseline and GCN-GRU temporal model are implemented.
- Elliptic preprocessing preserves endpoint labels, masks unknown labels, and makes an edge visible only after both endpoint transactions have been observed.
- Dynamic snapshots are cumulative and use chronological train/validation/test splits.
- Ground-truth neighbor-label features have been disabled to avoid target leakage.
- Node activation is tracked per snapshot; inactive nodes do not accumulate recurrent hidden state.
- Feature scaling is fitted on training snapshots only, then reused without refitting on validation/test data.
- Python regression tests and CI are included.
- **Research caveat:** before treating any attack results as evidence, validate that interventions update all affected graph features and history consistently. The current attack simulator remains experimental and should not be described as validated or realistic.
- No benchmark scores are published here because results must be rerun after these fixes.

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

The default configuration uses CPU for portability. If CUDA is available, set `project.device: cuda` in `configs/default.yaml`.

## Run

Run commands from the repository root.

```bash
python scripts/run_week1.py
python scripts/train_baseline.py --config configs/default.yaml
python scripts/train_dynamic.py --config configs/default.yaml
python scripts/run_attack.py --config configs/default.yaml
```

The dataset must be present locally before the pipeline can run. Training and attack scripts expect graph snapshots under `data/dynamic_graphs/`; checkpoints, metrics, and plots are written under `outputs/`.

## Tests

```bash
python -m compileall -q src scripts tests
pytest -q
```

GitHub Actions runs the fast test suite on pushes and pull requests. These tests cover metrics edge cases and chronological split behavior; they do not replace a full dataset-backed training run.

## Evaluation protocol

- Split snapshots chronologically, never randomly across time.
- Select checkpoints and thresholds using validation data only.
- Preserve hidden-state history when evaluating later time windows.
- Report F1, precision, recall, ROC-AUC, and PR-AUC; ROC-AUC and PR-AUC are undefined for single-class evaluation sets and are recorded as `null`.
- Report seeds, class distributions, dataset provenance, and configuration with any published results.
- Run multiple seeds and compare clean and attacked cases before making robustness claims.

## Research limitations

The Elliptic dataset is historical and labels are benchmark ground truth, not necessarily labels available to a live detector at transaction time. Any feature using labels must be treated as supervision only, never as a model input. Simulated edge injection is not evidence of evasion against a deployed financial institution. The attack pipeline requires further validation of causal graph updates, feature recomputation, attacker knowledge assumptions, and stealth constraints.

## License

See the repository license file, if present, for reuse terms.
