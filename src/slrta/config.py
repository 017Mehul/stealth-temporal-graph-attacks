import random
from pathlib import Path

import numpy as np
import torch
import yaml


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def ensure_dirs(cfg: dict) -> None:
    for key in ["raw_dir", "dynamic_dir", "checkpoints_dir", "metrics_dir", "plots_dir"]:
        p = cfg["paths"].get(key)
        if p:
            Path(p).mkdir(parents=True, exist_ok=True)
    processed = Path(cfg["paths"]["processed_csv"]).parent
    processed.mkdir(parents=True, exist_ok=True)
