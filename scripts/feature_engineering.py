import pandas as pd
import numpy as np
import geopandas as gpd
import shapely
import rasterio
import re
import os

from rasterstats import zonal_stats
from libpysal.weights import Queen, lag_spatial


# ------------- Feature functions  ------------- 

def ndvi(nir, red):
    return (nir - red) / (nir + red)


def ndbi(nir, swir):
    return (swir - nir) / (swir + nir)

#  -------------  Geometry helpers  ------------- 

def get_square_around_point(point_geom, delta_size=0.0005):
    point_coords = np.array(point_geom.coords[0])

    c1 = point_coords + [-delta_size, -delta_size]
    c2 = point_coords + [-delta_size, +delta_size]
    c3 = point_coords + [+delta_size, +delta_size]
    c4 = point_coords + [+delta_size, -delta_size]

    return shapely.geometry.Polygon([c1, c2, c3, c4])


def get_gdf_with_squares(gdf, delta_size=0.0005):
    gdf = gdf.copy()
    gdf["geometry"] = gdf["geometry"].apply(
        lambda geom: get_square_around_point(geom, delta_size)
    )
    return gdf


#  ------------- ESA extraction  ------------- 

def extract_proportions(gdf_squares, classes, wc_tif):
    wc_stats = zonal_stats(
        gdf_squares,
        wc_tif,
        stats="count",
        categorical=True,
        category_map=classes,
    )

    df = pd.DataFrame(wc_stats).fillna(0)
    total_pixels = df["count"]

    for _, original_name in classes.items():
        if original_name in df.columns:
            clean_name = re.sub(r"[^\w\s]", "", original_name)
            clean_name = re.sub(r"\s+", "_", clean_name).lower()

            df[f"{clean_name}_prop"] = df[original_name] / total_pixels

    return df


#  ------------- Main pipeline  ------------- 


def main():

    # Load data
    df = pd.read_csv("../output/nuremberg_dataset_final.csv")

    print("loaded dataset:", df.shape)

    # ----------------------------------------------------

    # Cleaning
    df = df.dropna()
    df.insert(0, "grid_id", range(len(df)))

    # ----------------------------------------------------

    # Spectral indices
    df["ndvi_2020"] = ndvi(df["b8_2020"], df["b4_2020"])
    df["ndvi_2021"] = ndvi(df["b8_2021"], df["b4_2021"])

    df["ndbi_2020"] = ndbi(df["b8_2020"], df["b11_2020"])
    df["ndbi_2021"] = ndbi(df["b8_2021"], df["b11_2021"])

    # ----------------------------------------------------

    # Temporal features
    df["delta_ndvi"] = df["ndvi_2021"] - df["ndvi_2020"]
    df["delta_ndbi"] = df["ndbi_2021"] - df["ndbi_2020"]

    # ----------------------------------------------------

    # GeoDataFrame
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["longitude"], df["latitude"]),
        crs="EPSG:4326",
    )

    gdf_squares = get_gdf_with_squares(gdf) # used to create wc_stats

    # ----------------------------------------------------

    # Load ESA stats (precomputed from notebook)
    wc_stats_2020 = pd.read_parquet("../data/hf_data/wc_stats_2020.parquet")
    wc_stats_2021 = pd.read_parquet("../data/hf_data/wc_stats_2021.parquet")
    # ----------------------------------------------------

    # More temporal features
    df["delta_built_up"] = (
        wc_stats_2021["built_up_prop"] - wc_stats_2020["built_up_prop"]
    )

    df["delta_veg"] = (
        wc_stats_2021["bare_sparse_vegetation_prop"]
        - wc_stats_2020["bare_sparse_vegetation_prop"]
    )

    # ----------------------------------------------------

    # Spatial autocorrelation
    w = Queen.from_dataframe(gdf)
    w.transform = "R"

    for col in ["ndvi_2020", "ndvi_2021", "ndbi_2020", "ndbi_2021"]:
        gdf[f"{col}_lag"] = lag_spatial(w, gdf[col])

    # ----------------------------------------------------

    # Labels
    threshold = 0.05

    df["changing_areas"] = np.abs(df["delta_built_up"]) > threshold
    df["built_up_increase"] = df["delta_built_up"] > threshold
    df["vegetation_decline"] = df["delta_veg"] < 0.0

    # ----------------------------------------------------

    # Export

    #Full Dataset
    os.makedirs("../data/labels/combined_format", exist_ok=True)

    df.to_parquet("../data/labels/combined_format/nuremberg_features_labels.parquet", index=False)

    # Split Dataset
    os.makedirs("../data/labels/split_format", exist_ok=True)

    features_2020 = ['b3_2020', 'b4_2020', 'b8_2020', 'b11_2020', 'ndvi_2020', 'ndbi_2020']
    features_2021 = ['b3_2021', 'b4_2021', 'b8_2021', 'b11_2021', 'ndvi_2021', 'ndbi_2021']
    features_delta = ['delta_ndvi', 'delta_ndbi', 'delta_built_up','delta_veg']
    labels = ['changing_areas', 'built_up_increase', 'vegetation_decline']

    df[['grid_id'] + features_2020].to_parquet("../data/labels/split_format/features_2020.parquet", index=False)
    df[['grid_id'] + features_2021].to_parquet("../data/labels/split_format/features_2021.parquet", index=False)
    df[['grid_id'] + features_delta + labels].to_parquet("../data/labels/split_format/labels.parquet", index=False)




if __name__ == "__main__":
    main()
