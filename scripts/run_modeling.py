import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    precision_recall_curve,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "data" / "labels" / "combined_format" / "nuremberg_features_labels.parquet"
OUTPUT_DIR = PROJECT_ROOT / "output" / "modeling_results"

TASKS = ["changing_areas", "built_up_increase", "vegetation_decline"]

LABEL_COLUMNS = {"changing_areas", "built_up_increase", "vegetation_decline"}

BASE_EXCLUDE = {
    "grid_id",
    "geometry",
}

DIRECT_LEAKAGE_BY_TASK = {
    "changing_areas": {"label_2020", "label_2021"},
    "built_up_increase": {"label_2020", "label_2021"},
    "vegetation_decline": {"label_2020", "label_2021"},
}


def load_dataset(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def spatial_split(df: pd.DataFrame):
    median_longitude = df["longitude"].median()
    train_df = df[df["longitude"] < median_longitude].copy()
    test_df = df[df["longitude"] >= median_longitude].copy()
    return train_df, test_df


def split_train_validation(train_df: pd.DataFrame):
    median_lat = train_df["latitude"].median()
    train_inner = train_df[train_df["latitude"] < median_lat].copy()
    val_df = train_df[train_df["latitude"] >= median_lat].copy()

    if len(train_inner) == 0 or len(val_df) == 0:
        split_idx = int(len(train_df) * 0.8)
        train_inner = train_df.iloc[:split_idx].copy()
        val_df = train_df.iloc[split_idx:].copy()

    return train_inner, val_df


def feature_columns_for_task(df: pd.DataFrame, task: str):
    candidate_cols = []
    drop_cols = BASE_EXCLUDE | LABEL_COLUMNS | DIRECT_LEAKAGE_BY_TASK.get(task, set())

    for col in df.columns:
        if col in drop_cols:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            candidate_cols.append(col)

    candidate_cols = [c for c in candidate_cols if c not in {"longitude", "latitude"}]

    forbidden_substrings = [
        "target",
        "future_label",
        "transition",
    ]
    candidate_cols = [
        c for c in candidate_cols
        if not any(s in c.lower() for s in forbidden_substrings)
    ]

    return candidate_cols


def prepare_task_data(df: pd.DataFrame, task: str):
    cols = feature_columns_for_task(df, task)
    X = df[cols].copy()
    y = df[task].astype(int).copy()

    print(f"[INFO] {task}: using {len(cols)} features")
    print(f"[INFO] {task}: excluded leakage cols {sorted(DIRECT_LEAKAGE_BY_TASK.get(task, set()))}")

    return X, y


def align_and_impute(X_train: pd.DataFrame, X_other: pd.DataFrame):
    X_other = X_other.reindex(columns=X_train.columns, fill_value=np.nan)

    imputer = SimpleImputer(strategy="median")
    X_train_imp = pd.DataFrame(
        imputer.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index,
    )
    X_other_imp = pd.DataFrame(
        imputer.transform(X_other),
        columns=X_train.columns,
        index=X_other.index,
    )

    return X_train_imp.astype(np.float32), X_other_imp.astype(np.float32)


def balance_training_data(X_train: pd.DataFrame, y_train: pd.Series, majority_ratio: int = 3):
    pos_idx = y_train[y_train == 1].index
    neg_idx = y_train[y_train == 0].index

    if len(pos_idx) == 0 or len(neg_idx) == 0:
        return X_train.copy(), y_train.copy()

    rng = np.random.RandomState(42)
    max_neg = min(len(neg_idx), len(pos_idx) * majority_ratio)
    sampled_neg = rng.choice(neg_idx, size=max_neg, replace=False)

    selected_idx = np.concatenate([pos_idx.values, sampled_neg])
    selected_idx = rng.permutation(selected_idx)

    return X_train.loc[selected_idx], y_train.loc[selected_idx]


def train_baseline_model(X_train, y_train):
    model = LogisticRegression(
        max_iter=3000,
        class_weight="balanced",
        solver="liblinear",
        random_state=42,
    )
    model.fit(X_train, y_train)
    return model


def train_tree_model(X_train, y_train):
    model = RandomForestClassifier(
        n_estimators=250,
        max_depth=18,
        min_samples_leaf=5,
        min_samples_split=10,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def train_boosted_model(X_train, y_train):
    model = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=250,
        max_depth=10,
        min_samples_leaf=30,
        l2_regularization=1.0,
        random_state=42,
    )
    model.fit(X_train, y_train)
    return model


def get_model_probabilities(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]

    if hasattr(model, "decision_function"):
        raw = model.decision_function(X)
        return 1.0 / (1.0 + np.exp(-raw))

    return model.predict(X).astype(float)


def evaluate_model(model, X_test, y_test, threshold=0.5):
    y_prob = get_model_probabilities(model, X_test)
    y_pred = (y_prob >= threshold).astype(int)

    results = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
    }
    return results, y_pred, y_prob


def false_change_rate(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    predicted_positive = y_pred == 1
    if predicted_positive.sum() == 0:
        return 0.0

    false_positive = ((y_pred == 1) & (y_true == 0)).sum()
    return float(false_positive / predicted_positive.sum())


def find_best_threshold(y_true, y_prob):
    if len(np.unique(y_true)) < 2:
        return 0.5, 0.0

    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)

    best_threshold = 0.5
    best_f1 = -1.0

    for i, thr in enumerate(thresholds):
        p = precisions[i]
        r = recalls[i]
        f1 = 0.0 if (p + r) == 0 else 2 * p * r / (p + r)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(thr)

    return float(best_threshold), float(best_f1)


def build_result_row(task, model_name, threshold, metrics, fcr, y_train, y_val, y_test, n_features):
    return {
        "task": task,
        "model": model_name,
        "threshold": float(threshold),
        "n_features": int(n_features),
        "accuracy": float(metrics["accuracy"]),
        "precision": float(metrics["precision"]),
        "recall": float(metrics["recall"]),
        "f1": float(metrics["f1"]),
        "false_change_rate": float(fcr),
        "train_positive_rate": float(np.mean(y_train)),
        "val_positive_rate": float(np.mean(y_val)),
        "test_positive_rate": float(np.mean(y_test)),
        "train_positive_count": int(np.sum(y_train)),
        "val_positive_count": int(np.sum(y_val)),
        "test_positive_count": int(np.sum(y_test)),
        "rank_score": float(
            0.55 * metrics["f1"] +
            0.25 * metrics["recall"] +
            0.15 * metrics["precision"] +
            0.05 * metrics["accuracy"]
        ),
    }


def run_task(df, task):
    print(f"\n{'=' * 70}")
    print(f"RUNNING TASK: {task}")
    print(f"{'=' * 70}")

    train_df, test_df = spatial_split(df)
    train_inner_df, val_df = split_train_validation(train_df)

    X_train, y_train = prepare_task_data(train_inner_df, task)
    X_val, y_val = prepare_task_data(val_df, task)
    X_test, y_test = prepare_task_data(test_df, task)

    n_features = X_train.shape[1]

    X_train, X_val = align_and_impute(X_train, X_val)
    X_train, X_test = align_and_impute(X_train, X_test)

    X_train_bal, y_train_bal = balance_training_data(X_train, y_train, majority_ratio=3)

    print("Train shape:", X_train_bal.shape)
    print("Validation shape:", X_val.shape)
    print("Test shape:", X_test.shape)
    print("Train positives:", int(y_train.sum()), "out of", len(y_train), f"({y_train.mean():.4%})")
    print("Val positives:", int(y_val.sum()), "out of", len(y_val), f"({y_val.mean():.4%})")
    print("Test positives:", int(y_test.sum()), "out of", len(y_test), f"({y_test.mean():.4%})")

    rows = []

    for model_name, trainer in [
        ("baseline", train_baseline_model),
        ("random_forest", train_tree_model),
        ("boosted_model", train_boosted_model),
    ]:
        print(f"\nTraining {model_name}...")
        model = trainer(X_train_bal, y_train_bal)

        val_prob = get_model_probabilities(model, X_val)
        threshold, _ = find_best_threshold(y_val.values, val_prob)

        test_metrics, test_preds, _ = evaluate_model(model, X_test, y_test, threshold=threshold)
        fcr = false_change_rate(y_test.values, test_preds)

        rows.append(build_result_row(
            task=task,
            model_name=model_name,
            threshold=threshold,
            metrics=test_metrics,
            fcr=fcr,
            y_train=y_train.values,
            y_val=y_val.values,
            y_test=y_test.values,
            n_features=n_features,
        ))

    results_df = pd.DataFrame(rows)
    print("\nTask results:")
    print(results_df[["model", "accuracy", "precision", "recall", "f1", "false_change_rate", "rank_score"]])

    return results_df


def main():
    print("Loading dataset...")
    df = load_dataset(DATASET_PATH)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_results = []

    for task in TASKS:
        task_results = run_task(df, task)
        all_results.append(task_results)

        task_file = OUTPUT_DIR / f"{task}_results.csv"
        task_results.to_csv(task_file, index=False)
        print(f"Saved: {task_file}")

    final_results = pd.concat(all_results, ignore_index=True)
    final_file = OUTPUT_DIR / "all_tasks_results.csv"
    final_results.to_csv(final_file, index=False)

    print(f"\nSaved: {final_file}")
    print("\nDone. All tasks finished.")


if __name__ == "__main__":
    main()