import pandas as pd


def load_dataset(path):
    return pd.read_parquet(path)


def spatial_split(df):
    median_longitude = df["longitude"].median()

    train_df = df[df["longitude"] < median_longitude].copy()
    test_df = df[df["longitude"] >= median_longitude].copy()

    return train_df, test_df


def prepare_task_data(df, task):
    label_map = {
        "changing_areas": "changing_areas",
        "built_up_increase": "built_up_increase",
        "vegetation_decline": "vegetation_decline",
    }

    target_col = label_map[task]

    drop_cols = [
        "grid_id",
        "longitude",
        "latitude",
        "changing_areas",
        "built_up_increase",
        "vegetation_decline",
    ]

    existing_drop_cols = [col for col in drop_cols if col in df.columns]

    X = df.drop(columns=existing_drop_cols).copy()
    y = df[target_col].astype(int).copy()

    return X, y


def align_and_impute(X_train, X_test):
    common_cols = X_train.columns.intersection(X_test.columns)

    X_train = X_train[common_cols].copy()
    X_test = X_test[common_cols].copy()

    train_medians = X_train.median(numeric_only=True)

    X_train = X_train.fillna(train_medians)
    X_test = X_test.fillna(train_medians)

    return X_train, X_test


def balance_training_data(X_train, y_train, majority_ratio=3):
    train_df = X_train.copy()
    train_df["target"] = y_train.values

    minority_df = train_df[train_df["target"] == 1]
    majority_df = train_df[train_df["target"] == 0]

    if len(minority_df) == 0:
        return X_train, y_train

    majority_sample_n = min(len(majority_df), len(minority_df) * majority_ratio)

    majority_sample = majority_df.sample(
        n=majority_sample_n,
        random_state=42
    )

    balanced_df = pd.concat([minority_df, majority_sample], axis=0)
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

    y_balanced = balanced_df["target"].astype(int)
    X_balanced = balanced_df.drop(columns=["target"])

    return X_balanced, y_balanced