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
