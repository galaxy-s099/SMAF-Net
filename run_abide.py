import argparse
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

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    all_results, summary = run_repeated_cv(config)

    results_df = pd.DataFrame(all_results)
    results_df.to_csv("smaf_v1_all_folds.csv", index=False)

    summary_df = pd.DataFrame([summary])
    summary_df.to_csv("smaf_v1_summary.csv", index=False)

    print("\nSaved:")
    print("smaf_v1_all_folds.csv")
    print("smaf_v1_summary.csv")


if __name__ == "__main__":
    main()