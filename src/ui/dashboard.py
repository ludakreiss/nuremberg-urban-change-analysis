import re
import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.neighbors import KDTree


BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))

# Pixel-level dataset used to draw the map
DATA_PATH = os.path.join(
    PROJECT_ROOT, "data", "labels", "combined_format",
    "nuremberg_features_labels.parquet"
)

# Folder that holds all four model-result CSV files
RESULTS_DIR = os.path.join(PROJECT_ROOT, "output", "modeling_results")

# Map each logical name → full file path
RESULTS_FILES = {
    "all_tasks":          os.path.join(RESULTS_DIR, "all_tasks_results.csv"),
    "changing_areas":     os.path.join(RESULTS_DIR, "changing_areas_results.csv"),
    "built_up_increase":  os.path.join(RESULTS_DIR, "built_up_increase_results.csv"),
    "vegetation_decline": os.path.join(RESULTS_DIR, "vegetation_decline_results.csv"),
}



@st.cache_data
def load_map_data() -> pd.DataFrame:
    # parquet dataset used to draw map
    return pd.read_parquet(DATA_PATH)


@st.cache_data
def load_results(path: str) -> pd.DataFrame:
    # model results csv
    return pd.read_csv(path)


@st.cache_resource
def build_kdtree(df: pd.DataFrame) -> KDTree:

    return KDTree(df[["latitude", "longitude"]].values)


# Load map data and spatial index at startup
df   = load_map_data()
tree = build_kdtree(df)

# Load all four result CSVs into a dictionary  { name: DataFrame }
results = {name: load_results(path) for name, path in RESULTS_FILES.items()}


#CONSTANTS

# ESA WorldCover numeric codes → human-readable class names
ESA_CLASS_NAMES: dict[int, str] = {
    10: "Tree cover",
    30: "Grassland",
    40: "Cropland",
    50: "Built-up",
    60: "Bare / sparse vegetation",
    80: "Permanent water",
}

# Hex colour for each ESA class (loosely follows the official WorldCover palette)
ESA_CLASS_COLORS: dict[int, str] = {
    10: "#1a7d26",   #trees
    30: "#a4d65e",   #grass
    40: "#c8a951",   #crops
    50: "#d73027",   #built-up / urban
    60: "#d9c99e",   #bare land
    80: "#2196f3",   #water

}

DEFAULT_COLOR   = "#888888"
MAP_SAMPLE_SIZE = 5_000
BBOX_HALF       = 0.008
TASK_LABELS = {
    "changing_areas":     "Changing Areas",
    "built_up_increase":  "Built-up Increase",
    "vegetation_decline": "Vegetation Decline",
}


#HELPER FUNCTIONS

def get_nearest_row(lat: float, lon: float) -> pd.Series:
    """Return the DataFrame row closest to (lat, lon) using the KDTree."""
    _, idx = tree.query([[lat, lon]], k=1)
    return df.iloc[idx[0][0]]


def make_bounding_box(lat: float, lon: float):
    """
    Return closed-polygon corner coordinates for a small square centred on
    (lat, lon). Used to draw a highlight rectangle on the map.
    Returns: (list_of_lats, list_of_lons)
    """
    lats = [lat - BBOX_HALF, lat + BBOX_HALF,
            lat + BBOX_HALF, lat - BBOX_HALF, lat - BBOX_HALF]
    lons = [lon - BBOX_HALF, lon - BBOX_HALF,
            lon + BBOX_HALF, lon + BBOX_HALF, lon - BBOX_HALF]
    return lats, lons


def label_to_color(label: int) -> str:
    """Return the hex colour for an ESA class code."""
    return ESA_CLASS_COLORS.get(int(label), DEFAULT_COLOR)


def label_to_name(label: int) -> str:
    """Return the human-readable name for an ESA class code."""
    return ESA_CLASS_NAMES.get(int(label), f"Unknown ({label})")


def get_task_metric(task_name: str, metric: str) -> float:

    df_task = results.get(task_name)
    if df_task is None or metric not in df_task.columns:
        return 0.0
    return float(df_task[metric].iloc[0])


#MAP BUILDER

def build_map(lat: float, lon: float, view_mode: str) -> go.Figure:

    sample = df.sample(min(MAP_SAMPLE_SIZE, len(df)), random_state=42)


    if view_mode == "2020":
        dot_colors = sample["label_2020"].apply(label_to_color)
        title_text = "Land Cover · 2020"

    elif view_mode == "2021":
        dot_colors = sample["label_2021"].apply(label_to_color)
        title_text = "Land Cover · 2021"

    else:   # "Change" view
        # Use delta_built_up column if available, otherwise compare labels
        if "delta_built_up" in sample.columns:
            changed = sample["delta_built_up"].abs() > 0.05
        else:
            changed = sample["label_2020"] != sample["label_2021"]
        dot_colors = changed.map({True: "#ff4d4d", False: "#26a69a"})
        title_text = "Change Detection · 2020 → 2021"

    # Layer 1 – data dots
    data_trace = go.Scattermapbox(
        lat=sample["latitude"], lon=sample["longitude"],
        mode="markers", name="Data points",
        marker=dict(size=5, color=dot_colors, opacity=0.75),
        hovertemplate="Lat: %{lat:.5f}<br>Lon: %{lon:.5f}<extra></extra>",
    )

    # Layer 2 – pin at selected point
    pin_trace = go.Scattermapbox(
        lat=[lat], lon=[lon],
        mode="markers", name="Selected",
        marker=dict(size=18, color="#ff1744"),
        hovertemplate=f"Selected<br>Lat: {lat:.5f}<br>Lon: {lon:.5f}<extra></extra>",
    )

    # Layer 3 – bounding box
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


# Speedometer style chart
def build_gauge(value: float, title: str, color: str) -> go.Figure:

    fig = go.Figure(go.Indicator(
        mode  = "gauge+number",
        value = round(value * 100, 1),     # convert to percentage
        number= dict(suffix="%",
                     font=dict(color="#e6edf3", size=28,
                               family="JetBrains Mono")),
        title = dict(text=title,
                     font=dict(color="#8b949e", size=12,
                               family="Space Grotesk")),
        gauge = dict(
            axis      = dict(range=[0, 100],
                             tickcolor="#30363d",
                             tickfont=dict(color="#8b949e", size=10)),
            bar       = dict(color=color),
            bgcolor   = "#1c2333",
            bordercolor="#30363d",
            steps     = [dict(range=[0, 100], color="#21262d")],
        ),
    ))
    fig.update_layout(
        paper_bgcolor="#0d1117",
        height=200,
        margin=dict(l=20, r=20, t=40, b=10),
    )
    return fig


# ─── PAGE CONFIG & CSS ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Nuremberg Land Cover Analysis",
    page_icon="🔍",
    layout="wide",
)

# Custom CSS injected as HTML — gives the app a dark GitHub-style theme.
# All colours use CSS variables defined in :root so they're easy to change.
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

/* ── Top header bar ── */
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

/* ── Section headings ── */
.section-heading {
    font-size: 0.75rem; text-transform: uppercase; letter-spacing: 1.5px;
    color: var(--text-muted); border-bottom: 1px solid var(--border);
    padding-bottom: 0.4rem; margin: 1rem 0 0.6rem;
}

/* ── Legend items ── */
.legend-item {
    display: flex; align-items: center; gap: 0.6rem;
    padding: 0.35rem 0.5rem; border-radius: 6px; margin-bottom: 0.3rem;
    transition: background 0.15s;
}
.legend-item:hover { background: var(--bg-card); }
.legend-swatch { width: 14px; height: 14px; border-radius: 3px; flex-shrink: 0; }
.legend-label { font-size: 0.82rem; color: var(--text-main); }

/* ── Task result block (wraps gauges + metric row) ── */
.task-block {
    background: var(--bg-panel); border: 1px solid var(--border);
    border-radius: 12px; padding: 1rem 1.4rem 0.2rem; margin-bottom: 0.6rem;
}
.task-block .task-title {
    font-size: 1rem; font-weight: 600; color: var(--text-main);
    margin-bottom: 0.6rem; border-bottom: 1px solid var(--border);
    padding-bottom: 0.4rem;
}

/* ── Inline mini-metric row (exact numbers below gauges) ── */
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

/* ── Streamlit widget overrides ── */
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
.stRadio label { color: var(--text-main) !important; }
</style>
""", unsafe_allow_html=True)


# ─── PAGE HEADER ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
  <h1>Nuremberg Urban Land Cover Analysis</h1>
  <p>Satellite-derived land cover change · 2020 → 2021 · ESA WorldCover + Sentinel-2 · Real model results</p>
</div>
""", unsafe_allow_html=True)


#TABS
tab_map, tab_results, tab_stats = st.tabs([
    "📍  Map Explorer",
    "🤖  Model Results",
    "📊  Area Summary",
])


# Tab 1: Map explorer
with tab_map:

    ctrl_col, map_col = st.columns([1, 3], gap="medium")

    #Left column: input controls and legend
    with ctrl_col:
        st.markdown('<div class="section-heading">Coordinates</div>',
                    unsafe_allow_html=True)

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

        st.markdown('<div class="section-heading">View Mode</div>',
                    unsafe_allow_html=True)

        view_mode = st.radio(
            "Colour dots by:",
            options=["2020", "2021", "Change"],
            index=0,
            help=(
                    "2020 / 2021 → colour by ESA land-cover class in that year.\n"
                    "Change → red = class changed, teal = stable."
                            ),
        )

        st.markdown('<div class="section-heading">Legend</div>',
                    unsafe_allow_html=True)

        # Show a two-item legend for Change mode, full ESA palette otherwise
        if view_mode == "Change":
            for colour, name in [("#ff4d4d", "Changed"), ("#26a69a", "Stable")]:
                st.markdown(
                    f'<div class="legend-item">'
                    f'<div class="legend-swatch" style="background:{colour}"></div>'
                    f'<span class="legend-label">{name}</span></div>',
                    unsafe_allow_html=True,
                )
        else:
            for code, name in ESA_CLASS_NAMES.items():
                colour = ESA_CLASS_COLORS.get(code, DEFAULT_COLOR)
                st.markdown(
                    f'<div class="legend-item">'
                    f'<div class="legend-swatch" style="background:{colour}"></div>'
                    f'<span class="legend-label">{name}</span></div>',
                    unsafe_allow_html=True,
                )

    # map
    with map_col:
        st.plotly_chart(
            build_map(lat, lon, view_mode),
            use_container_width=True,
            config={"scrollZoom": True},
        )

        # Info caption below the map showing the nearest pixel's ESA labels
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

    st.markdown('<div class="section-heading">Detailed Results per Task</div>',
                unsafe_allow_html=True)

    for task_key, task_label in TASK_LABELS.items():

        # Opening HTML of the task block (dark panel)
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

        # Four columns, one gauge chart each
        g1, g2, g3, g4 = st.columns(4)
        with g1:
            st.plotly_chart(
                build_gauge(acc, "Accuracy", "#58a6ff"),
                use_container_width=True,
                key=f"{task_key}_accuracy"
            )

        with g2:
            st.plotly_chart(
                build_gauge(prec, "Precision", "#3fb950"),
                use_container_width=True,
                key=f"{task_key}_precision"
            )
        with g3:
            st.plotly_chart(
                build_gauge(rec, "Recall", "#f0c040"),
                use_container_width=True,
                key=f"{task_key}_recall"
            )
        with g4:
            st.plotly_chart(
                build_gauge(f1, "F1 Score", "#c084fc"),
                use_container_width=True,
                key=f"{task_key}_f1"
            )

        # Exact numeric values as a horizontal row of mini cards below gauges
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

#False change rate
    st.markdown('<div class="section-heading">False Change Rate per Task</div>',
                unsafe_allow_html=True)


    fcr_fig = go.Figure(go.Bar(
        x            = df_all["Task"],
        y            = (df_all["false_change_rate"] * 100).round(2),
        marker_color = "#f85149",
        marker_line_width = 0,
        text         = (df_all["false_change_rate"] * 100).round(2).astype(str) + "%",
        textposition = "outside",
        textfont     = dict(color="#e6edf3", size=12),
    ))

    max_fcr = df_all["false_change_rate"].max()
    fcr_fig.update_layout(
        paper_bgcolor = "#0d1117",
        plot_bgcolor  = "#0d1117",
        font          = dict(color="#8b949e", family="Space Grotesk"),
        xaxis         = dict(showgrid=False),
        yaxis         = dict(
            range=[0, max(max_fcr * 120, 10)],   # always show at least 10% range
            gridcolor="#21262d",
            ticksuffix="%",
            title="False Change Rate (%)",
        ),
        height  = 280,
        margin  = dict(l=10, r=10, t=20, b=10),
    )

    st.plotly_chart(fcr_fig, use_container_width=True)

# TAB 3 : Area Summary
with tab_stats:

    st.markdown('<div class="section-heading">Land Cover Area (km²)</div>',
                unsafe_allow_html=True)

    summary = pd.DataFrame({
        "Class":           ["Built-up", "Tree cover", "Cropland",
                            "Grassland", "Water", "Bare veg."],
        "Area 2020 (km²)": [87.3, 31.5, 42.1, 18.7, 4.2, 2.1],
        "Area 2021 (km²)": [89.1, 31.2, 39.8, 18.9, 4.2, 2.7],
    })
    summary["Change (km²)"] = (
        summary["Area 2021 (km²)"] - summary["Area 2020 (km²)"]
    ).round(1)

    # Colour-code the Change column: red = shrinkage, green = growth
    def colour_change(val):
        if val < 0:   return "color: #f85149; font-weight: 600"
        elif val > 0: return "color: #3fb950; font-weight: 600"
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
                             marker_color="#58a6ff", marker_line_width=0))
    bar_fig.add_trace(go.Bar(name="2021", x=summary["Class"],
                             y=summary["Area 2021 (km²)"],
                             marker_color="#3fb950", marker_line_width=0))
    bar_fig.update_layout(
        barmode="group", paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
        font=dict(color="#8b949e", family="Space Grotesk"),
        xaxis=dict(showgrid=False),
        yaxis=dict(title="km²", gridcolor="#21262d"),
        legend=dict(font=dict(color="#e6edf3")),
        height=320, margin=dict(l=10, r=10, t=20, b=10),
    )
    st.plotly_chart(bar_fig, use_container_width=True)

    # Net change bar
    st.markdown('<div class="section-heading">Net Change per Class</div>',
                unsafe_allow_html=True)

    delta_fig = go.Figure(go.Bar(
        x=summary["Class"], y=summary["Change (km²)"],
        text=[f"{v:+.1f}" for v in summary["Change (km²)"]],
        textposition="outside",
        textfont=dict(color="#e6edf3", size=12),
        marker_color=["#f85149" if v < 0 else "#3fb950"
                      for v in summary["Change (km²)"]],
        marker_line_width=0,
    ))
    delta_fig.update_layout(
        paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
        font=dict(color="#8b949e", family="Space Grotesk"),
        xaxis=dict(showgrid=False),
        yaxis=dict(title="Δ km²", gridcolor="#21262d",
                   zeroline=True, zerolinecolor="#30363d"),
        height=280, margin=dict(l=10, r=10, t=20, b=10),
    )
    st.plotly_chart(delta_fig, use_container_width=True)


#FOOTER
st.markdown("""
<div style="margin-top:2rem; padding:0.8rem 1rem; border-top:1px solid #21262d;
            color:#8b949e; font-size:0.78rem; text-align:center;">
  Data: ESA WorldCover 10 m · Sentinel-2 L2A · Nuremberg · 2020–2021
  &nbsp;|&nbsp; Model results: output/modeling_results/
  &nbsp;|&nbsp; Built with Streamlit &amp; Plotly
</div>
""", unsafe_allow_html=True)
