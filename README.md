# Nuremberg Urban Change Analysis

This repository contains the code and analysis for a project focused on detecting and understanding urban land cover changes in Nuremberg, Germany, between 2020 and 2021. The analysis leverages satellite imagery from ESA WorldCover and Sentinel-2 to train machine learning models that can predict urban development, vegetation decline, and other significant changes. The project includes an interactive dashboard built with Streamlit for visual exploration of the data and model results.

## Features
- **Satellite Data Integration**: Combines ESA WorldCover land cover maps with Sentinel-2 multispectral imagery.
- **Feature Engineering**: Calculates spectral indices (NDVI, NDBI), temporal differences, and spatial autocorrelation features to capture land cover dynamics.
- **Machine Learning Pipeline**: Implements a full modeling pipeline, including data preparation, spatial cross-validation, and training of Logistic Regression, Random Forest, and HistGradientBoostingClassifier models.
- **Change Detection Tasks**: Trains models to perform three distinct tasks:
    1. Identify any significant land cover change.
    2. Detect increases in built-up areas.
    3. Pinpoint areas of vegetation decline.
- **Robust Evaluation**: Evaluates models using standard classification metrics and a custom False Change Rate to measure precision in change detection.
- **Interactive Dashboard**: Features a Streamlit application for visualizing map data, pixel information, and detailed model performance metrics.

## Project Structure
```
.
├── LICENSE
├── main.py
├── orchestrator.py
├── pipeline_worker.py
├── requirements.txt
├── notebooks/
│   ├── 00_ml_final_project.ipynb
│   └── feature_engineering.ipynb
├── output/
│   └── .getkeep
├── scripts/
│   ├── build_master_dataset.py
│   ├── download_hf_data.py
│   ├── feature_engineering.py
│   └── run_modeling.py
└── src/
    ├── geospatial/
    │   └── raster_utils.py
    ├── modeling/
    │   ├── data.py
    │   ├── evaluate.py
    │   └── train.py
    └── ui/
        └── dashboard.py
```

## Setup and Installation

**1. Clone the Repository**
```bash
git clone https://github.com/ludakreiss/nuremberg-urban-change-analysis.git
cd nuremberg-urban-change-analysis
```

**2. Create a Virtual Environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

**3. Install Dependencies**
```bash
pip install -r requirements.txt
```

## Usage
The entire analysis can be run from a single command, which executes the download, data processing, and modeling steps in order.

```bash
python main.py
```
This will automatically:
1. Download data from Hugging Face.
2. Build the master dataset from raw satellite imagery.
3. Engineer features and labels.
4. Train and evaluate all models.
5. Launch the Streamlit dashboard.

Alternatively, you can run each step of the pipeline individually using the scripts in the `scripts/` directory.

### 1. Download Data
The required datasets are hosted on Hugging Face Hub. This script downloads them into a `data/hf_data/` directory.
```bash
python scripts/download_hf_data.py
```

### 2. Build the Master Dataset
This script combines the raw Sentinel-2 and ESA WorldCover satellite rasters for the Nuremberg area into a unified CSV file. This will generate `output/nuremberg_dataset_final.csv`.
```bash
python scripts/build_master_dataset.py
```

### 3. Perform Feature Engineering
This script processes the master dataset to calculate spectral indices, temporal differences, spatial lag features, and labels for the machine learning tasks. The final feature-engineered dataset is saved to `data/labels/combined_format/nuremberg_features_labels.parquet`.
```bash
python scripts/feature_engineering.py
```

### 4. Run the Modeling Pipeline
This script trains and evaluates the baseline (Logistic Regression) and tree-based (Random Forest, Gradient Boosting) models for all three prediction tasks. Results, including performance metrics and the false change rate, will be saved as CSV files in the `output/modeling_results/` directory.
```bash
python scripts/run_modeling.py
```

### 5. Launch the Interactive Dashboard
The project includes a Streamlit dashboard for exploring the data and results. This dashboard allows for visualizing map data, pixel information, and detailed model performance metrics.
```bash
streamlit run src/ui/dashboard.py
