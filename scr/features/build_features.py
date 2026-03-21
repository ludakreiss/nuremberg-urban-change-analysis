def aggregate_features(df):
    return df.groupby("grid_id").mean().reset_index()
