import os
import sys
import pandas as pd

sys.path.append("src")

from modeling.data import load_dataset, spatial_split, prepare_task_data
from modeling.train import train_baseline_model, train_tree_model
from modeling.evaluate import evaluate_model, false_change_rate, add_feature_noise


DATASET_PATH = "data/labels/combined_format/nuremberg_features_labels.parquet"
TASK = "changing_areas"


def main():
    print("Loading dataset...")
    df = load_dataset(DATASET_PATH)

    print("Creating spatial split...")
    train_df, test_df = spatial_split(df)

    X_train, y_train = prepare_task_data(train_df, TASK)
    X_test, y_test = prepare_task_data(test_df, TASK)

    print("\nTraining baseline model...")
    baseline_model = train_baseline_model(X_train, y_train)
    baseline_results, baseline_preds = evaluate_model(baseline_model, X_test, y_test)
    baseline_fcr = false_change_rate(y_test.values, baseline_preds)

    print("Baseline results:")
    print(baseline_results)
    print({"false_change_rate": baseline_fcr})

    print("\nTraining tree model...")
    tree_model = train_tree_model(X_train, y_train)
    tree_results, tree_preds = evaluate_model(tree_model, X_test, y_test)
    tree_fcr = false_change_rate(y_test.values, tree_preds)

    print("Tree model results:")
    print(tree_results)
    print({"false_change_rate": tree_fcr})

    print("\nRunning robustness check with noisy features...")
    X_test_noisy = add_feature_noise(X_test)
    noisy_results, noisy_preds = evaluate_model(tree_model, X_test_noisy, y_test)
    noisy_fcr = false_change_rate(y_test.values, noisy_preds)

    print("Robustness results:")
    print(noisy_results)
    print({"false_change_rate": noisy_fcr})

    os.makedirs("output/modeling_results", exist_ok=True)

    results_df = pd.DataFrame([
        {"model": "baseline", **baseline_results, "false_change_rate": baseline_fcr},
        {"model": "tree_model", **tree_results, "false_change_rate": tree_fcr},
        {"model": "tree_model_noisy_test", **noisy_results, "false_change_rate": noisy_fcr},
    ])

    results_df.to_csv(f"output/modeling_results/{TASK}_results.csv", index=False)

    print(f"\nSaved results to output/modeling_results/{TASK}_results.csv")


if __name__ == "__main__":
    main()
