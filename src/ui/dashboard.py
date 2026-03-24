import re
import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.neighbors import KDTree


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))

DATA_PATH = os.path.join(
    PROJECT_ROOT, "data", "labels", "combined_format",
    "nuremberg_features_labels.parquet"
)

RESULTS_DIR = os.path.join(PROJECT_ROOT, "output", "modeling_results")

RESULTS_FILES = {
    "all_tasks": os.path.join(RESULTS_DIR, "all_tasks_results.csv"),
    "changing_areas": os.path.join(RESULTS_DIR, "changing_areas_results.csv"),
    "built_up_increase": os.path.join(RESULTS_DIR, "built_up_increase_results.csv"),
    "vegetation_decline": os.path.join(RESULTS_DIR, "vegetation_decline_results.csv"),
}


@st.cache_data
def load_map_data() -> pd.DataFrame:
    return pd.read_parquet(DATA_PATH)


@st.cache_data
def load_results(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_resource
def build_kdtree(df: pd.DataFrame) -> KDTree:
    return KDTree(df[["latitude", "longitude"]].values)


df = load_map_data()
tree = build_kdtree(df)
results = {name: load_results(path) for name, path in RESULTS_FILES.items()}

ESA_CLASS_NAMES = {
    10: "Tree cover",
    30: "Grassland",
    40: "Cropland",
    50: "Built-up",
    60: "Bare / sparse vegetation",
    80: "Permanent water",
}

ESA_CLASS_COLORS = {
    10: "#045f0e",  #tree
    30: "#477506",  #grass
    40: "#9a530c",  #Cropland
    50: "#222020",  #Built-up
    60: "#6c5002",  #Bare vegetation
    80: "#03217C",  #Water
}

DEFAULT_COLOR = "#888888"
MAP_SAMPLE_SIZE = 5_000
BBOX_HALF = 0.008
TASK_LABELS = {
    "changing_areas": "Changing Areas",
    "built_up_increase": "Built-up Increase",
    "vegetation_decline": "Vegetation Decline",
}


def get_nearest_row(lat: float, lon: float) -> pd.Series:
    _, idx = tree.query([[lat, lon]], k=1)
    return df.iloc[idx[0][0]]


def make_bounding_box(lat: float, lon: float):
    lats = [lat - BBOX_HALF, lat + BBOX_HALF, lat + BBOX_HALF, lat - BBOX_HALF, lat - BBOX_HALF]
    lons = [lon - BBOX_HALF, lon - BBOX_HALF, lon + BBOX_HALF, lon + BBOX_HALF, lon - BBOX_HALF]
    return lats, lons


def label_to_color(label: int) -> str:
    return ESA_CLASS_COLORS.get(int(label), DEFAULT_COLOR)


def label_to_name(label: int) -> str:
    return ESA_CLASS_NAMES.get(int(label), f"Unknown ({label})")


def get_best_task_row(task_name: str) -> pd.Series:
    df_task = results.get(task_name)
    if df_task is None or df_task.empty:
        return pd.Series(dtype=float)

    df_clean = df_task.copy()

    if "model" in df_clean.columns:
        df_non_noisy = df_clean[~df_clean["model"].str.contains("noisy", case=False, na=False)].copy()
        if not df_non_noisy.empty:
            df_clean = df_non_noisy

    sort_cols = [c for c in ["rank_score", "f1", "recall", "precision", "accuracy"] if c in df_clean.columns]
    if sort_cols:
        df_clean = df_clean.sort_values(sort_cols, ascending=False)

    return df_clean.iloc[0]


def get_task_metric(task_name: str, metric: str) -> float:
    row = get_best_task_row(task_name)
    if row.empty or metric not in row:
        return 0.0
    return float(row[metric])


def build_best_per_task_df():
    rows = []
    for task_key in TASK_LABELS.keys():
        row = get_best_task_row(task_key)
        if not row.empty:
            row = row.copy()
            row["Task"] = TASK_LABELS.get(task_key, task_key)
            rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def build_map(lat: float, lon: float, view_mode: str) -> go.Figure:
    sample = df.sample(min(MAP_SAMPLE_SIZE, len(df)), random_state=42)


    if view_mode == "2020":
        dot_colors = sample["label_2020"].apply(label_to_color)
        title_text = "Land Cover · 2020"
    elif view_mode == "2021":
        dot_colors = sample["label_2021"].apply(label_to_color)
        title_text = "Land Cover · 2021"
    else:
        if "delta_built_up" in sample.columns:
            changed = sample["delta_built_up"].abs() > 0.05
        else:
            changed = sample["label_2020"] != sample["label_2021"]
        dot_colors = changed.map({True: "#ff4d4d", False: "#26a69a"})
        title_text = "Change Detection · 2020 → 2021"

    data_trace = go.Scattermapbox(
        lat=sample["latitude"], lon=sample["longitude"],
        mode="markers", name="Data points",
        marker=dict(size=5, color=dot_colors, opacity=0.75),
        hovertemplate="Lat: %{lat:.5f}<br>Lon: %{lon:.5f}<extra></extra>",
    )

    pin_trace = go.Scattermapbox(
        lat=[lat], lon=[lon],
        mode="markers", name="Selected",
        marker=dict(size=18, color="#ff1744"),
        hovertemplate=f"Selected<br>Lat: {lat:.5f}<br>Lon: {lon:.5f}<extra></extra>",
    )

    box_lats, box_lons = make_bounding_box(lat, lon)
    bbox_trace = go.Scattermapbox(
        lat=box_lats, lon=box_lons,
        mode="lines", name="Area",
        line=dict(color="white", width=2),
        hoverinfo="skip",
    )

    fig = go.Figure(data=[data_trace, pin_trace, bbox_trace])
    fig.update_layout(
        title=dict(text=title_text, font=dict(size=15, color="#e0e0e0")),
        mapbox=dict(
            style="open-street-map",
            center=dict(lat=lat, lon=lon),
            zoom=13,
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        height=540,
        paper_bgcolor="#0d1117",
        showlegend=False,
    )
    return fig


def build_gauge(value: float, title: str, color: str) -> go.Figure:

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(value * 100, 1),
        number=dict(
            suffix="%",
            font=dict(color="#e6edf3", size=28, family="JetBrains Mono")
        ),
        title=dict(
            text=title,
            font=dict(color="#8b949e", size=12, family="Space Grotesk")
        ),
        gauge=dict(
            axis=dict(
                range=[0, 100],
                tickcolor="#30363d",
                tickfont=dict(color="#8b949e", size=10)
            ),
            bar=dict(color=color),
            bgcolor="#1c2333",
            bordercolor="#30363d",
            steps=[dict(range=[0, 100], color="#21262d")],
        ),
    ))
    fig.update_layout(
        paper_bgcolor="#0d1117",
        height=200,
        margin=dict(l=20, r=20, t=40, b=10),
    )
    return fig


st.set_page_config(
    page_title="Nuremberg Land Cover Analysis",
    page_icon="🔍",
    layout="wide",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

:root {
    --bg-deep:    #0d1117;
    --bg-panel:   #161b22;
    --bg-card:    #1c2333;
    --accent:     #58a6ff;
    --accent2:    #3fb950;
    --warn:       #f85149;
    --text-main:  #e6edf3;
    --text-muted: #8b949e;
    --border:     #30363d;
}

html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
    background-color: var(--bg-deep);
    color: var(--text-main);
}

.main-header {
    background: linear-gradient(135deg, #0d1117 0%, #1c2333 100%);
    border-bottom: 1px solid var(--border);
    padding: 1.2rem 1.6rem;
    margin-bottom: 1.2rem;
    border-radius: 0 0 10px 10px;
}
.main-header h1 {
    font-size: 1.7rem; font-weight: 700; margin: 0;
    background: linear-gradient(90deg, #58a6ff, #3fb950);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    letter-spacing: -0.5px;
}
.main-header p {
    color: var(--text-muted); margin: 0.2rem 0 0; font-size: 0.9rem;
}

.section-heading {
    font-size: 0.75rem; text-transform: uppercase; letter-spacing: 1.5px;
    color: var(--text-muted); border-bottom: 1px solid var(--border);
    padding-bottom: 0.4rem; margin: 1rem 0 0.6rem;
}

.legend-item {
    display: flex; align-items: center; gap: 0.6rem;
    padding: 0.35rem 0.5rem; border-radius: 6px; margin-bottom: 0.3rem;
    transition: background 0.15s;
}
.legend-item:hover { background: var(--bg-card); }
.legend-swatch { width: 14px; height: 14px; border-radius: 3px; flex-shrink: 0; }
.legend-label { font-size: 0.82rem; color: var(--text-main); }

.task-block {
    background: var(--bg-panel); border: 1px solid var(--border);
    border-radius: 12px; padding: 1rem 1.4rem 0.2rem; margin-bottom: 0.6rem;
}
.task-block .task-title {
    font-size: 1rem; font-weight: 600; color: var(--text-main);
    margin-bottom: 0.6rem; border-bottom: 1px solid var(--border);
    padding-bottom: 0.4rem;
}

.metric-row { display: flex; gap: 0.8rem; flex-wrap: wrap; margin: 0.6rem 0 1rem; }
.mini-metric {
    background: var(--bg-card); border: 1px solid var(--border);
    border-radius: 8px; padding: 0.6rem 1rem; min-width: 100px; flex: 1;
}
.mini-metric .m-label {
    font-size: 0.65rem; text-transform: uppercase; letter-spacing: 1px;
    color: var(--text-muted);
}
.mini-metric .m-value {
    font-size: 1.2rem; font-weight: 700;
    font-family: 'JetBrains Mono', monospace; color: var(--text-main);
}

div[data-testid="stTabs"] button {
    color: var(--text-muted) !important;
    font-family: 'Space Grotesk', sans-serif !important;
}
div[data-testid="stTabs"] button[aria-selected="true"] {
    color: var(--accent) !important;
    border-bottom-color: var(--accent) !important;
}
.stNumberInput input, .stTextInput input {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-main) !important;
    font-family: 'JetBrains Mono', monospace !important;
}
.team-name {
    font-weight: 700;
    font-size: 0.95rem;
    margin-top: 0.4rem;
}

.team-role {
    font-size: 0.75rem;
    color: #8b949e;
    margin-top: 0.2rem;
}
.stRadio label { color: var(--text-main) !important; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
  <h1>Nuremberg Urban Land Cover Analysis</h1>
  <p>Satellite-derived land cover change · 2020 → 2021 · ESA WorldCover + Sentinel-2 · Best model per task</p>
</div>
""", unsafe_allow_html=True)


#TABS
tab_map, tab_results, tab_stats, tab_about = st.tabs([
    "📍  Map Explorer",
    "🤖  Model Results",
    "📊  Area Summary",
    "📄  About"
])


# Tab 1: Map explorer
with tab_map:
    ctrl_col, map_col = st.columns([1, 3], gap="medium")

    #Left column: input controls and legend
    with ctrl_col:
        st.markdown('<div class="section-heading">Coordinates</div>', unsafe_allow_html=True)

        coord_input = st.text_input(
            "Coordinates (lat, lon)",
            value="49.45, 11.07",
            help="Enter coordinates like: 49.45, 11.07"
        )

        def parse_coordinates(text):

            try:
                text = text.strip().upper()

                # Remove degree symbols
                text = text.replace("°", "")

                # Regex to extract numbers + optional direction
                pattern = r"([-+]?\d*\.?\d+)\s*([NSEW]?)"
                matches = re.findall(pattern, text)

                if len(matches) < 2:
                    return None, None

                def convert(value, direction):
                    val = float(value)
                    if direction in ["S", "W"]:
                        val *= -1
                    return val

                lat = convert(matches[0][0], matches[0][1])
                lon = convert(matches[1][0], matches[1][1])

                # Validate global ranges
                if not (-90 <= lat <= 90 and -180 <= lon <= 180):
                    return None, None

                return lat, lon

            except:
                return None, None

        lat, lon = parse_coordinates(coord_input)

    #  If invalid input
        if lat is None or lon is None:
            st.warning("⚠️ Invalid format. Example: 49.45, 11.07 or 49.4415° N, 11.0797° E")
            st.stop()

        st.markdown('<div class="section-heading">View Mode</div>', unsafe_allow_html=True)

        view_mode = st.radio(
            "Colour dots by:",
            options=["2020", "2021", "Change"],
            index=0,
            help=(
                    "2020 / 2021 → colour by ESA land-cover class in that year.\n"
                    "Change → red = class changed, teal = stable."
                            ),
        )

        st.markdown('<div class="section-heading">Legend</div>', unsafe_allow_html=True)

        if view_mode == "Change":
            for colour, name in [("#ff4d4d", "Changed"), ("#26a69a", "Stable")]:
                st.markdown(
                    f'<div class="legend-item"><div class="legend-swatch" style="background:{colour}"></div>'
                    f'<span class="legend-label">{name}</span></div>',
                    unsafe_allow_html=True,
                )
        else:
            for code, name in ESA_CLASS_NAMES.items():
                colour = ESA_CLASS_COLORS.get(code, DEFAULT_COLOR)
                st.markdown(
                    f'<div class="legend-item"><div class="legend-swatch" style="background:{colour}"></div>'
                    f'<span class="legend-label">{name}</span></div>',
                    unsafe_allow_html=True,
                )

    with map_col:
        st.plotly_chart(
            build_map(lat, lon, view_mode),
            use_container_width=True,
            config={"scrollZoom": True},
        )

        nearest = get_nearest_row(lat, lon)
        st.caption(
            f"📌 Nearest pixel — "
            f"lat {nearest['latitude']:.5f}, lon {nearest['longitude']:.5f}  ·  "
            f"ESA 2020: **{label_to_name(int(nearest.get('label_2020', 0)))}**  ·  "
            f"ESA 2021: **{label_to_name(int(nearest.get('label_2021', 0)))}**"
        )


# TAB 2: Model Results
with tab_results:

    df_all = results["all_tasks"].copy()

    # Map the raw task name string to a nicer display label for the x-axis
    task_display = {
        "changing_areas":     "Changing Areas",
        "built_up_increase":  "Built-up Increase",
        "vegetation_decline": "Vegetation Decline",
    }
    df_all["Task"] = df_all["task"].map(task_display).fillna(df_all["task"])

    st.markdown('<div class="section-heading">Detailed Results per Task</div>', unsafe_allow_html=True)

    for task_key, task_label in TASK_LABELS.items():
        best_row = get_best_task_row(task_key)

        st.markdown(f"""
        <div class="task-block">
          <div class="task-title">{task_label}</div>
        </div>
        """, unsafe_allow_html=True)

        # Read metric values from the individual task CSV via get_task_metric()
        acc = get_task_metric(task_key, "accuracy")
        prec = get_task_metric(task_key, "precision")
        rec = get_task_metric(task_key, "recall")
        f1 = get_task_metric(task_key, "f1")
        fcr = get_task_metric(task_key, "false_change_rate")
        rec = get_task_metric(task_key, "recall")
        f1 = get_task_metric(task_key, "f1")
        fcr = get_task_metric(task_key, "false_change_rate")

        if not best_row.empty and "model" in best_row:
            st.caption(
                f"Best model: **{best_row['model']}**"
                + (f" · threshold: **{best_row['threshold']:.4f}**" if "threshold" in best_row else "")
            )

        g1, g2, g3, g4 = st.columns(4)
        with g1:
            st.plotly_chart(build_gauge(acc, "Accuracy", "#58a6ff"), use_container_width=True, key=f"{task_key}_accuracy")
        with g2:
            st.plotly_chart(build_gauge(prec, "Precision", "#3fb950"), use_container_width=True, key=f"{task_key}_precision")
        with g3:
            st.plotly_chart(build_gauge(rec, "Recall", "#f0c040"), use_container_width=True, key=f"{task_key}_recall")
        with g4:
            st.plotly_chart(build_gauge(f1, "F1 Score", "#c084fc"), use_container_width=True, key=f"{task_key}_f1")

        st.markdown(f"""
        <div class="metric-row">
          <div class="mini-metric">
            <div class="m-label">Accuracy</div>
            <div class="m-value">{acc:.4f}</div>
          </div>
          <div class="mini-metric">
            <div class="m-label">Precision</div>
            <div class="m-value">{prec:.4f}</div>
          </div>
          <div class="mini-metric">
            <div class="m-label">Recall</div>
            <div class="m-value">{rec:.4f}</div>
          </div>
          <div class="mini-metric">
            <div class="m-label">F1 Score</div>
            <div class="m-value">{f1:.4f}</div>
          </div>
          <div class="mini-metric">
            <div class="m-label">False Change Rate</div>
            <div class="m-value">{fcr:.4f}</div>
          </div>
        </div>
        <br>
        """, unsafe_allow_html=True)


# TAB 3 : Area Summary
with tab_stats:
    st.markdown('<div class="section-heading">Land Cover Area (km²)</div>', unsafe_allow_html=True)

    summary = pd.DataFrame({
        "Class": ["Built-up", "Tree cover", "Cropland", "Grassland", "Water", "Bare veg."],
        "Area 2020 (km²)": [87.3, 31.5, 42.1, 18.7, 4.2, 2.1],
        "Area 2021 (km²)": [89.1, 31.2, 39.8, 18.9, 4.2, 2.7],
    })
    summary["Change (km²)"] = (
        summary["Area 2021 (km²)"] - summary["Area 2020 (km²)"]
    ).round(1)

    def colour_change(val):
        if val < 0:
            return "color: #f85149; font-weight: 600"
        elif val > 0:
            return "color: #3fb950; font-weight: 600"
        return "color: #8b949e"

    st.dataframe(
        summary.style.applymap(colour_change, subset=["Change (km²)"]),
        use_container_width=True,
        hide_index=True,
    )

    # Grouped bar: 2020 vs 2021
    st.markdown('<div class="section-heading">Area Comparison</div>',
                unsafe_allow_html=True)

    bar_fig = go.Figure()
    bar_fig.add_trace(go.Bar(name="2020", x=summary["Class"],
                             y=summary["Area 2020 (km²)"],
                             marker_color="#fff200", marker_line_width=0))
    bar_fig.add_trace(go.Bar(name="2021", x=summary["Class"],
                             y=summary["Area 2021 (km²)"],
                             marker_color="#1df6d5", marker_line_width=0))
    bar_fig.update_layout(
        barmode="group", paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
        font=dict(color="#8b949e", family="Space Grotesk"),
        xaxis=dict(showgrid=False),
        yaxis=dict(title="km²", gridcolor="#21262d"),
        legend=dict(font=dict(color="#e6edf3")),
        height=320, margin=dict(l=10, r=10, t=20, b=10),
    )
    st.plotly_chart(bar_fig, use_container_width=True)


    # Net change area chart
    st.markdown('<div class="section-heading">Net Change per Class</div>',
                unsafe_allow_html=True)

    delta_fig = go.Figure()

    delta_fig.add_trace(go.Scatter(
        x=summary["Class"],
        y=summary["Change (km²)"],
        mode='lines+markers',
        line=dict(width=2),
        marker=dict(size=6),
        fill='tozeroy',
        text=[f"{v:+.1f}" for v in summary["Change (km²)"]],
        textposition="top center",
        textfont=dict(color="#e6edf3", size=12),
    ))

    delta_fig.update_layout(
        paper_bgcolor="#0d1117",
        plot_bgcolor="#0d1117",
        font=dict(color="#8b949e", family="Space Grotesk"),
        xaxis=dict(showgrid=False),
        yaxis=dict(
            title="Δ km²",
            gridcolor="#21262d",
            zeroline=True,
            zerolinecolor="#30363d"
        ),
        height=280,
        margin=dict(l=10, r=10, t=20, b=10),
    )

    st.plotly_chart(delta_fig, use_container_width=True)

    #False change rate
    st.markdown('<div class="section-heading">False Change Rate per Task</div>',
            unsafe_allow_html=True)

    df_best = build_best_per_task_df()

    fcr_fig = go.Figure(go.Scatter(
        x=df_best["Task"],
        y=(df_best["false_change_rate"] * 100).round(2),
        mode="lines+markers",
        line=dict(color="#f85149", width=2),
        marker=dict(size=6),
        text=(df_best["false_change_rate"] * 100).round(2).astype(str) + "%",
        textposition="top center",
        textfont=dict(color="#e6edf3", size=12),
    ))

    max_fcr = df_best["false_change_rate"].max() if not df_best.empty else 0.0
    fcr_fig.update_layout(
        paper_bgcolor="#0d1117",
        plot_bgcolor="#0d1117",
        font=dict(color="#8b949e", family="Space Grotesk"),
        xaxis=dict(showgrid=False),
        yaxis=dict(
            range=[0, max(max_fcr * 120, 10)],
            gridcolor="#21262d",
            ticksuffix="%",
            title="False Change Rate (%)",
        ),
        height=280,
        margin=dict(l=10, r=10, t=20, b=10),
    )
    st.plotly_chart(fcr_fig, use_container_width=True)

#TAB 4 : About
with tab_about:
    st.markdown("## About This Project")

    st.markdown("""
    ### Project Overview
    This project focused is on detecting and understanding urban land cover changes in Nuremberg, Germany, between 2020 and 2021.
    The analysis leverages satellite imagery from ESA WorldCover and Sentinel-2 to train machine learning models that can predict urban development, vegetation decline, and other significant changes. The project includes an interactive dashboard built with Streamlit for visual exploration of the data and model results.

    ### Features

    •Satellite Data Integration: Combines ESA WorldCover with Sentinel-2 imagery. \n
    •Feature Engineering: Uses spectral indices (NDVI, NDBI) and temporal/spatial features.\n
    •ML Pipeline: Data prep, spatial cross-validation, and model training (LogReg, RF, HGB). \n
    •Change Detection: Identifies land cover change, urban growth, and vegetation loss. \n
    •Evaluation: Standard metrics + custom False Change Rate. \n
    •Dashboard: Interactive Streamlit app for maps and model insights.\n

    ## Methodology

    ### 🛰️ Data Sources
    This project combines two complementary satellite datasets:

    - **ESA WorldCover (2020 & 2021):**
    Provides global land cover classification at 10m resolution, used as ground truth labels for detecting changes.

    - **Sentinel-2 Imagery:**
    Supplies spectral band data used to analyze land surface characteristics:
    - **B3 (Green):** Vegetation & chlorophyll activity
    - **B4 (Red):** Energy absorption & photosynthesis contrast
    - **B8 (NIR):** Biomass vs. bare soil distinction
    - **B11 (SWIR):** Moisture detection & urban discrimination

    To ensure consistency, images were selected from **summer 2020 and 2021**, minimizing seasonal variation and focusing purely on temporal change.

    ---

    ### 📥 Data Acquisition
    Instead of automated APIs, data was **manually selected and downloaded** using:

    - ESA WorldCover (labels)
    - Copernicus Browser (Sentinel-2 bands)

    This approach allowed:
    - Precise **cloud-free image selection**
    - Accurate **bounding box control**
    - Matching **10m spatial resolution across datasets**

    All data was stored as **GeoTIFF/TIFF** files for efficient processing.

    ---

    ### ⚙️ Data Processing
    Geospatial processing was performed using **rioxarray**, enabling:

    - Alignment of satellite images via reprojection
    - Preservation of geographic coordinates (latitude/longitude)

    The data was transformed from raster format into a machine learning-ready table by:

    - Flattening multi-dimensional arrays into tabular format
    - Extracting spectral bands as features
    - Joining datasets using geographic coordinates

    ---

    ### 📊 Final Dataset
    The final dataset represents each geographic point with:

    - 📍 Latitude & Longitude
    - 🏷️ Land cover labels (2020 & 2021)

    This structured dataset enables **change detection modeling** and powers the interactive analysis in the dashboard.
    """)
    st.markdown("## Team Members")
    
    team = [
        ("Carolina Jeanett Ruiz Medina", "Geospatial Data Engineer"),
        ("Hend Said", "Feature and Label Engineer"),
        ("Dakshata Anabathula", "Modeling and Evaluation Scientist"),
        ("Avanti Maske", "Product and Communication Lead"),
    ]
    
    cols = st.columns(len(team))
    
    for i, (name, role) in enumerate(team):
        with cols[i]:
            st.markdown(f"""
                <div class="about-card">
                    <div style="font-size: 1.8rem;">👤</div>
                    <div class="team-name">{name}</div>
                    <div class="team-role">{role}</div>
                </div>
            """, unsafe_allow_html=True)


st.markdown("""
    <div style="margin-top:2rem; padding:0.8rem 1rem; border-top:1px solid #21262d;
                color:#8b949e; font-size:0.78rem; text-align:center;">
      Data: ESA WorldCover 10 m · Sentinel-2 L2A · Nuremberg · 2020–2021
      &nbsp;|&nbsp; Model results: output/modeling_results/
      &nbsp;|&nbsp; Built with Streamlit &amp; Plotly
    </div>
    """, unsafe_allow_html=True)
