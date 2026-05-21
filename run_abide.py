import argparse
from pathlib import Path

import yaml
import pandas as pd

from train import run_repeated_cv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/abide_smaf_v1.yaml"
    )
    args = parser.parse_args()

    config_path = Path(args.config)

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    all_results, summary = run_repeated_cv(config)

    exp_name = config_path.stem

    results_df = pd.DataFrame(all_results)
    results_path = f"{exp_name}_all_folds.csv"
    results_df.to_csv(results_path, index=False)

    summary_df = pd.DataFrame([summary])
    summary_path = f"{exp_name}_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    print("\nSaved:")
    print(results_path)
    print(summary_path)


if __name__ == "__main__":
    main()