from huggingface_hub import hf_hub_download
import os

REPO_ID = "Daksanna/nuremberg-urban-dynamics-data"
LOCAL_DIR = "../data/hf_data"

FILES = [
    "ESA_WorldCover_10m_2020_v100_N48E009_Map.tif",
    "ESA_WorldCover_10m_2021_v200_N48E009_Map.tif",
    "Sentinel2_B11_20210730.tiff",
    "Sentinel2_B11_20210812.tiff",
    "Sentinel2_B3_4_8_20210730.tiff",
    "Sentinel2_B3_4_8_20210812.tiff",
    "wc_stats_2020.parquet",
    "wc_stats_2021.parquet",
]

os.makedirs(LOCAL_DIR, exist_ok=True)

for file_name in FILES:
    path = hf_hub_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        filename=file_name,
        local_dir=LOCAL_DIR,
    )
    print("Downloaded:", path)