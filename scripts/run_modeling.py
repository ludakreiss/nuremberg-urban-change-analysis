import os
import sys
import pandas as pd

sys.path.append("../src")

from modeling.data import (
    load_dataset,
    spatial_split,
    prepare_task_data,
    align_and_impute,
    balance_training_data,
)
from modeling.train import train_baseline_model, train_tree_model
from modeling.evaluate import evaluate_model, false_change_rate, add_feature_noise


DATASET_PATH = "../data/labels/combined_format/nuremberg_features_labels.parquet"
TASKS = ["changing_areas", "built_up_increase", "vegetation_decline"]

TASK_THRESHOLDS = {
    "changing_areas": 0.35,
    "built_up_increase": 0.30,
    "vegetation_decline": 0.30,
}


def run_task(df, task):
    print(f"\n{'=' * 60}")
    print(f"RUNNING TASK: {task}")
    print(f"{'=' * 60}")

    train_df, test_df = spatial_split(df)

    X_train, y_train = prepare_task_data(train_df, task)
    X_test, y_test = prepare_task_data(test_df, task)

    X_train, X_test = align_and_impute(X_train, X_test)

    print("Original train shape:", X_train.shape)
    print("Test shape:", X_test.shape)

    print("Original train label distribution:")
    print(y_train.value_counts(normalize=True))

    print("Test label distribution:")
    print(y_test.value_counts(normalize=True))

    X_train_bal, y_train_bal = balance_training_data(X_train, y_train, majority_ratio=3)

    print("Balanced train shape:", X_train_bal.shape)
    print("Balanced train label distribution:")
    print(y_train_bal.value_counts(normalize=True))

    threshold = TASK_THRESHOLDS[task]
    print("Decision threshold:", threshold)

    print("\nTraining baseline model...")
    baseline_model = train_baseline_model(X_train_bal, y_train_bal)
    baseline_results, baseline_preds, baseline_probs = evaluate_model(
        baseline_model, X_test, y_test, threshold=threshold
    )
    baseline_fcr = false_change_rate(y_test.values, baseline_preds)

    print("Baseline results:")
    print(baseline_results)
    print({"false_change_rate": baseline_fcr})

    print("\nTraining tree model...")
    tree_model = train_tree_model(X_train_bal, y_train_bal)
    tree_results, tree_preds, tree_probs = evaluate_model(
        tree_model, X_test, y_test, threshold=threshold
    )
    tree_fcr = false_change_rate(y_test.values, tree_preds)

    print("Tree model results:")
    print(tree_results)
    print({"false_change_rate": tree_fcr})

    print("\nRunning robustness check with noisy features...")
    X_test_noisy = add_feature_noise(X_test)
    noisy_results, noisy_preds, noisy_probs = evaluate_model(
        tree_model, X_test_noisy, y_test, threshold=threshold
    )
    noisy_fcr = false_change_rate(y_test.values, noisy_preds)

    print("Robustness results:")
    print(noisy_results)
    print({"false_change_rate": noisy_fcr})

    results_df = pd.DataFrame([
        {"task": task, "model": "baseline", "threshold": threshold, **baseline_results, "false_change_rate": baseline_fcr},
        {"task": task, "model": "tree_model", "threshold": threshold, **tree_results, "false_change_rate": tree_fcr},
        {"task": task, "model": "tree_model_noisy_test", "threshold": threshold, **noisy_results, "false_change_rate": noisy_fcr},
    ])

    return results_df


def main():
    print("Loading dataset...")
    df = load_dataset(DATASET_PATH)

    os.makedirs("../output/modeling_results", exist_ok=True)

    all_results = []

    for task in TASKS:
        task_results = run_task(df, task)
        all_results.append(task_results)

        task_results.to_csv(f"../output/modeling_results/{task}_results.csv", index=False)
        print(f"Saved: ../output/modeling_results/{task}_results.csv")

    final_results = pd.concat(all_results, ignore_index=True)
    final_results.to_csv("../output/modeling_results/all_tasks_results.csv", index=False)

    print("\nSaved: ../output/modeling_results/all_tasks_results.csv")
    print("\nDone. All 3 tasks finished.")


if __name__ == "__main__":
    main()