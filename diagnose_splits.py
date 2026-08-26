"""
PERM - Predictive Edge Resource Management
Split Diagnostics

Quick diagnostic for investigating negative/poor R² on the RF models.
Prints target variance and distribution per split so you can see whether
poor test performance is a train/test distribution mismatch (common with
a temporal split when the test window happens to land in a low-variance
phase like cooldown) versus a genuine model/feature problem.

Run:
    python diagnose_splits.py
"""

import argparse

import pandas as pd


def describe_split(name: str, path: str):
    df = pd.read_csv(path)
    print(f"\n=== {name} ({path}) — n={len(df)} ===")

    print("Phase mix:")
    counts = df["phase"].value_counts(normalize=True).round(3) * 100
    for phase, pct in counts.items():
        print(f"    {phase:<10} {pct:5.1f}%")

    for col in ["target_cpu_percent", "target_latency_ms"]:
        s = df[col]
        print(f"{col}: mean={s.mean():7.2f}  std={s.std():7.2f}  "
              f"min={s.min():7.2f}  max={s.max():7.2f}  variance={s.var():9.2f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="features_train.csv")
    parser.add_argument("--val", default="features_val.csv")
    parser.add_argument("--test", default="features_test.csv")
    args = parser.parse_args()

    describe_split("TRAIN", args.train)
    describe_split("VAL", args.val)
    describe_split("TEST", args.test)

    print(
        "\nWhat to look for:\n"
        "  - If TEST's target variance is much smaller than TRAIN's, negative R²\n"
        "    on test is expected/less alarming - R² is measured relative to that\n"
        "    split's OWN variance, so a low-variance split makes R² look worse\n"
        "    even with the same absolute error (MAE/RMSE) as other splits.\n"
        "  - If TEST's mean is far from TRAIN's mean for the same phase, that's\n"
        "    a real distribution shift the model never saw during training.\n"
    )


if __name__ == "__main__":
    main()