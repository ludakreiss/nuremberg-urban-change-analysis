import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def load_data(path):
    df = pd.read_parquet(path)

    # remove missing
    df = df.dropna()

    return df

def prepare_data(df, task="changing_areas"):

    # labels from your feature engineer
    label_map = {
        "changing_areas": "change",
        "built_up_increase": "built_up_increase",
        "vegetation_decline": "veg_decline"
    }

    label_col = label_map[task]

    # remove non-feature columns
    drop_cols = ["longitude", "latitude", "grid_id",
                 "change", "built_up_increase", "veg_decline"]

    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    y = df[label_col]

    return X, y

def spatial_split(df):
    # simple: split by longitude
    train = df[df["longitude"] < df["longitude"].median()]
    test = df[df["longitude"] >= df["longitude"].median()]

    return train, test

def train_baseline(X_train, y_train):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    return model

def evaluate(model, X_test, y_test):

    y_pred = model.predict(X_test)

    results = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
    }

    return results

def add_noise(X):
    noise = np.random.normal(0, 0.01, X.shape)
    return X + noise

def run_pipeline(path, task="changing_areas"):

    print("Loading data...")
    df = load_data(path)

    print("Splitting spatially...")
    train_df, test_df = spatial_split(df)

    X_train, y_train = prepare_data(train_df, task)
    X_test, y_test = prepare_data(test_df, task)

    print("\n--- BASELINE MODEL ---")
    baseline = train_baseline(X_train, y_train)
    res_baseline = evaluate(baseline, X_test, y_test)
    print(res_baseline)

    print("\n--- RANDOM FOREST ---")
    rf = train_random_forest(X_train, y_train)
    res_rf = evaluate(rf, X_test, y_test)
    print(res_rf)

    print("\n--- ROBUSTNESS (noise test) ---")
    X_test_noisy = add_noise(X_test)
    res_noise = evaluate(rf, X_test_noisy, y_test)
    print(res_noise)

if __name__ == "__main__":
    run_pipeline(
        "data/processed/combined_format/nuremberg_features_labels.parquet",
        task="changing_areas"
    )
