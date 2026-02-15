import argparse
from pathlib import Path


def main(dataset: str, out_dir: str):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if dataset.lower() == "elliptic":
        print("Network download is not executed in this environment.")
        print("Place these files manually in data/raw/:")
        print("  - elliptic_txs_edgelist.csv")
        print("  - elliptic_txs_classes.csv")
        print("  - elliptic_txs_features.csv")
    elif dataset.lower() == "paysim":
        print("Network download is not executed in this environment.")
        print("Place PaySim CSV in data/raw/transactions.csv and set dataset_type=generic.")
    else:
        raise ValueError("dataset must be one of: elliptic, paysim")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="elliptic", choices=["elliptic", "paysim"])
    parser.add_argument("--out_dir", default="data/raw")
    args = parser.parse_args()
    main(args.dataset, args.out_dir)
