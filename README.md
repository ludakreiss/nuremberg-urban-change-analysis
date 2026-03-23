# Nuremberg Urban Change Analysis
This repository contains the code and analysis for a project focused on detecting and understanding urban land cover changes in Nuremberg, Germany, between 2020 and 2021. The analysis leverages satellite imagery from ESA WorldCover and Sentinel-2 to train machine learning models that can predict urban development, vegetation decline, and other significant changes. The project includes interactive dashboards built with Streamlit for visual exploration of the data and model results.

## Features
* Satellite Data Integration: Combines ESA WorldCover land cover maps with Sentinel-2 multispectral imagery.

* Feature Engineering: Calculates spectral indices (NDVI, NDBI), temporal differences, and spatial autocorrelation features to capture land cover dynamics.
* Machine Learning Pipeline: Implements a full modeling pipeline, including data preparation, spatial cross-validation, and training of 
   * Logistic Regression and 
   * Random Forest models.

* Change Detection Tasks: Trains models to perform three distinct tasks:
   * Identify any significant land cover change.
   * Detect increases in built-up areas.
   * Pinpoint areas of vegetation decline.

* Robust Evaluation: Evaluates models using standard classification metrics and a custom False Change Rate to measure precision in change detection.

* Interactive Dashboards: Features two Streamlit applications for visualizing data, model predictions, and evaluation results.

## Project Structure
```
├── data/                    # Holds raw and processed data (populated by scripts)
├── notebooks/               # Jupyter notebooks for exploration and prototyping
├── output/                  # Stores generated files like datasets and model results
├── scripts/                 # Main Python scripts for the data and modeling pipeline
│   ├── download_hf_data.py  # Downloads data from Hugging Face Hub
│   ├── build_master_dataset.py # Creates the initial raster dataset
│   ├── feature_engineering.py # Generates features and labels
│   └── run_modeling.py      # Trains and evaluates models
├── src/                     # Source code for the project
│   ├── modeling/            # Modules for data handling, training, and evaluation
│   ├── nud/                 # Geospatial utility functions
│   └── ui/                  # Streamlit dashboard applications
├── requirements.txt         # Project dependencies
└── LICENSE                  # Project license
```

## Setup and Installation
Clone the Repository

```
git clone https://github.com/ludakreiss/nuremberg-urban-change-analysis.git 
cd nuremberg-urban-change-analysis
```

Create a Virtual Environment

```
python -m venv venv
source venv/bin/activate # On Windows use venv\Scripts\activate ```

Install Dependencies

```pip install -r requirements.txt```

## Usage
Follow these steps to run the complete pipeline from data download to model training and visualization.

### 1. Download Data
The required datasets are hosted on Hugging Face Hub. Run the following script to download them into the `data/hf_data/` directory.

```python scripts/download_hf_data.py```

### 2. Build the Master Dataset
This script combines the raw Sentinel-2 and ESA WorldCover satellite rasters for the Nuremberg area into a unified CSV file.

```python scripts/build_master_dataset.py```

This will generate `output/nuremberg_dataset_final.csv`.

### 3. Perform Feature Engineering
This script processes the master dataset to calculate spectral indices, temporal differences, and other features. It also defines the labels for the machine learning tasks.

```python scripts/feature_engineering.py```

The final feature-engineered dataset is saved to `data/labels/combined_format/nuremberg_features_labels.parquet`.

### 4. Run the Modeling Pipeline
This script trains and evaluates the baseline (Logistic Regression) and tree-based (Random Forest) models for all three prediction tasks.

```python scripts/run_modeling.py```

Results, including performance metrics and the false change rate, will be saved as CSV files in the `output/modeling_results/` directory.

### 5. Launch the Interactive Dashboards
The project includes two Streamlit dashboards for exploring the data and results.

Model and Data Explorer (`dashboard.py`): An in-depth dashboard for visualizing map data, pixel information, and detailed model performance metrics.

```streamlit run src/ui/dashboard.py```

Location Search App (`app.py`): A user-friendly application to search for locations within Nuremberg and view mock land cover predictions.

```streamlit run src/ui/app.py```
