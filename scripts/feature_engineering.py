import os
import numpy as np
import pandas as pd


BUILT_UP_CLASS = 50
VEGETATION_CLASSES = {10, 30}  # Tree cover, Grassland
NON_VEGETATION_CLASSES = {40, 50, 60, 80}  # Cropland, Built-up, Bare/sparse, Water


def ndvi(nir, red):
    denom = nir + red
    return np.where(denom == 0, 0, (nir - red) / denom)


def ndbi(nir, swir):
    denom = swir + nir
    return np.where(denom == 0, 0, (swir - nir) / denom)


def safe_read_parquet(path):
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return None
    try:
        df = pd.read_parquet(path)
        print(f"Loaded parquet: {path} | shape={df.shape}")
        return df
    except Exception as e:
        print(f"Could not read parquet {path}: {e}")
        return None


def try_merge_worldcover_stats(df, wc_2020, wc_2021):
    if wc_2020 is None or wc_2021 is None:
        print("WorldCover stats missing. Skipping merge.")
        return df

    possible_keys = ["grid_id", "id", "tile_id", "cell_id"]
    shared_key = None

    for key in possible_keys:
        if key in df.columns and key in wc_2020.columns and key in wc_2021.columns:
            shared_key = key
            break

    if shared_key is None:
        print("No shared merge key found between master dataset and WC stats.")
        print("Skipping WC stats merge.")
        return df

    print(f"Merging WC stats using key: {shared_key}")

    wc_2020 = wc_2020.copy().rename(
        columns={c: f"{c}_2020" for c in wc_2020.columns if c != shared_key}
    )
    wc_2021 = wc_2021.copy().rename(
        columns={c: f"{c}_2021" for c in wc_2021.columns if c != shared_key}
    )

    df = df.merge(wc_2020, on=shared_key, how="left")
    df = df.merge(wc_2021, on=shared_key, how="left")

    print(f"After WC merge: {df.shape}")
    return df


def build_spatial_context_features(df, base_cols, lat_bin_size=0.001, lon_bin_size=0.001):
    print("Building scalable spatial neighborhood features...")

    df = df.copy()

    df["lat_bin"] = np.floor(df["latitude"] / lat_bin_size).astype(np.int32)
    df["lon_bin"] = np.floor(df["longitude"] / lon_bin_size).astype(np.int32)

    agg_dict = {col: "mean" for col in base_cols}
    bin_stats = (
        df.groupby(["lat_bin", "lon_bin"], as_index=False)
        .agg(agg_dict)
        .rename(columns={col: f"{col}_cell_mean" for col in base_cols})
    )

    neighbor_frames = []
    for dlat in [-1, 0, 1]:
        for dlon in [-1, 0, 1]:
            tmp = bin_stats.copy()
            tmp["lat_bin"] = tmp["lat_bin"] - dlat
            tmp["lon_bin"] = tmp["lon_bin"] - dlon
            neighbor_frames.append(tmp)

    neighbors_all = pd.concat(neighbor_frames, ignore_index=True)
    neighbor_cols = [f"{col}_cell_mean" for col in base_cols]

    neighbor_stats = (
        neighbors_all.groupby(["lat_bin", "lon_bin"], as_index=False)[neighbor_cols]
        .mean()
        .rename(columns={f"{col}_cell_mean": f"{col}_lag" for col in base_cols})
    )

    df = df.merge(neighbor_stats, on=["lat_bin", "lon_bin"], how="left")
    df.drop(columns=["lat_bin", "lon_bin"], inplace=True)

    print("Spatial neighborhood features created.")
    return df


def main():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    input_csv = os.path.join(project_root, "output", "nuremberg_dataset_final.csv")
    wc_2020_path = os.path.join(project_root, "data", "hf_data", "wc_stats_2020.parquet")
    wc_2021_path = os.path.join(project_root, "data", "hf_data", "wc_stats_2021.parquet")

    output_combined_dir = os.path.join(project_root, "data", "labels", "combined_format")
    output_split_dir = os.path.join(project_root, "data", "labels", "split_format")

    os.makedirs(output_combined_dir, exist_ok=True)
    os.makedirs(output_split_dir, exist_ok=True)

    df = pd.read_csv(input_csv)
    print("Loaded dataset:", df.shape)

    df = df.reset_index(drop=True)

    if "grid_id" not in df.columns:
        df.insert(0, "grid_id", np.arange(len(df), dtype=np.int64))

    # basic type cleanup for memory
    float_cols = [
        "longitude", "latitude",
        "b3_2020", "b4_2020", "b8_2020", "b11_2020",
        "b3_2021", "b4_2021", "b8_2021", "b11_2021",
    ]
    for col in float_cols:
        if col in df.columns:
            df[col] = df[col].astype(np.float32)

    for col in ["label_2020", "label_2021"]:
        if col in df.columns:
            df[col] = df[col].astype(np.int16)

    # spectral indices
    df["ndvi_2020"] = ndvi(df["b8_2020"], df["b4_2020"]).astype(np.float32)
    df["ndvi_2021"] = ndvi(df["b8_2021"], df["b4_2021"]).astype(np.float32)
    df["ndbi_2020"] = ndbi(df["b8_2020"], df["b11_2020"]).astype(np.float32)
    df["ndbi_2021"] = ndbi(df["b8_2021"], df["b11_2021"]).astype(np.float32)

    # temporal descriptors
    df["delta_ndvi"] = (df["ndvi_2021"] - df["ndvi_2020"]).astype(np.float32)
    df["delta_ndbi"] = (df["ndbi_2021"] - df["ndbi_2020"]).astype(np.float32)

    # optional WC stats
    wc_stats_2020 = safe_read_parquet(wc_2020_path)
    wc_stats_2021 = safe_read_parquet(wc_2021_path)
    df = try_merge_worldcover_stats(df, wc_stats_2020, wc_stats_2021)

    # spatial context features
    lag_base_cols = [
        "b3_2020", "b4_2020", "b8_2020", "b11_2020",
        "b3_2021", "b4_2021", "b8_2021", "b11_2021",
        "ndvi_2020", "ndvi_2021",
        "ndbi_2020", "ndbi_2021",
        "delta_ndvi", "delta_ndbi",
    ]
    lag_base_cols = [c for c in lag_base_cols if c in df.columns]
    df = build_spatial_context_features(df, lag_base_cols)

    # realistic labels from ESA transitions
    y0 = df["label_2020"]
    y1 = df["label_2021"]

    df["changing_areas"] = (y0 != y1).astype(np.int8)

    df["built_up_increase"] = (
        (y0 != BUILT_UP_CLASS) & (y1 == BUILT_UP_CLASS)
    ).astype(np.int8)

    df["vegetation_decline"] = (
        y0.isin(VEGETATION_CLASSES) & y1.isin(NON_VEGETATION_CLASSES)
    ).astype(np.int8)

    print("\nLabel distribution:")
    for label_col in ["changing_areas", "built_up_increase", "vegetation_decline"]:
        positive_rate = float(df[label_col].mean())
        positive_count = int(df[label_col].sum())
        print(f"  {label_col}: {positive_count}/{len(df)} ({positive_rate:.4%})")

    combined_path = os.path.join(output_combined_dir, "nuremberg_features_labels.parquet")
    df.to_parquet(combined_path, index=False)
    print(f"[OK] Saved combined dataset: {combined_path}")

    split_feature_cols_2020 = [c for c in [
        "grid_id", "longitude", "latitude",
        "label_2020", "b3_2020", "b4_2020", "b8_2020", "b11_2020", "ndvi_2020", "ndbi_2020"
    ] if c in df.columns]

    split_feature_cols_2021 = [c for c in [
        "grid_id", "label_2021", "b3_2021", "b4_2021", "b8_2021", "b11_2021", "ndvi_2021", "ndbi_2021"
    ] if c in df.columns]

    split_label_cols = [c for c in [
        "grid_id",
        "delta_ndvi", "delta_ndbi",
        "changing_areas", "built_up_increase", "vegetation_decline"
    ] if c in df.columns]

    df[split_feature_cols_2020].to_parquet(
        os.path.join(output_split_dir, "features_2020.parquet"),
        index=False,
    )
    df[split_feature_cols_2021].to_parquet(
        os.path.join(output_split_dir, "features_2021.parquet"),
        index=False,
    )
    df[split_label_cols].to_parquet(
        os.path.join(output_split_dir, "labels.parquet"),
        index=False,
    )

    print("Feature engineering completed successfully.")


if __name__ == "__main__":
    main()