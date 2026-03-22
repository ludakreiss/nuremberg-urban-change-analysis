import os
import sys
import pandas as pd

sys.path.append("src")

from modeling.data import load_dataset, spatial_split, prepare_task_data
from modeling.train import train_baseline_model, train_tree_model
from modeling.evaluate import evaluate_model, false_change_rate, add_feature_noise


DATASET_PATH = "data/labels/combined_format/nuremberg_features_labels.parquet"
TASKS = ["changing_areas", "built_up_increase", "vegetation_decline"]


def run_task(df, task):
    print(f"\n{'=' * 60}")
    print(f"RUNNING TASK: {task}")
    print(f"{'=' * 60}")

    train_df, test_df = spatial_split(df)

    X_train, y_train = prepare_task_data(train_df, task)
    X_test, y_test = prepare_task_data(test_df, task)

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

    results_df = pd.DataFrame([
        {"task": task, "model": "baseline", **baseline_results, "false_change_rate": baseline_fcr},
        {"task": task, "model": "tree_model", **tree_results, "false_change_rate": tree_fcr},
        {"task": task, "model": "tree_model_noisy_test", **noisy_results, "false_change_rate": noisy_fcr},
    ])

    return results_df


def main():
    print("Loading dataset...")
    df = load_dataset(DATASET_PATH)

    os.makedirs("output/modeling_results", exist_ok=True)

    all_results = []

    for task in TASKS:
        task_results = run_task(df, task)
        all_results.append(task_results)

        task_results.to_csv(f"output/modeling_results/{task}_results.csv", index=False)
        print(f"Saved: output/modeling_results/{task}_results.csv")

    final_results = pd.concat(all_results, ignore_index=True)
    final_results.to_csv("output/modeling_results/all_tasks_results.csv", index=False)

    print("\nSaved: output/modeling_results/all_tasks_results.csv")
    print("\nDone. All 3 tasks finished.")


if __name__ == "__main__":
    main()
