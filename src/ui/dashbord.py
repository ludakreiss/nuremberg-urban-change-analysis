import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.neighbors import KDTree


#PATH SETUP

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))

DATA_PATH = os.path.join(
    PROJECT_ROOT,
    "data", "labels", "combined_format",
    "nuremberg_features_labels.parquet"
)


#DATA LOADING
@st.cache_data
def load_data() -> pd.DataFrame:
    """Read the parquet data file and return a Pandas DataFrame."""
    return pd.read_parquet(DATA_PATH)



@st.cache_resource
def build_kdtree(df: pd.DataFrame) -> KDTree:
    """
    Build a KDTree from the latitude / longitude columns.
    A KDTree is a data structure for very fast nearest-neighbour searches:
    given a clicked lat/lon we instantly find the closest row in the dataset.
    """
    return KDTree(df[["latitude", "longitude"]].values)


# Load everything at startup
df   = load_data()
tree = build_kdtree(df)


# ─── 4. CONSTANTS ──────────────────────────────────────────────────────────────
# ESA WorldCover uses numeric codes to represent land-cover types.
# These dictionaries map code → human name  and  code → hex colour.
ESA_CLASS_NAMES: dict[int, str] = {
    10: "Tree cover",
    30: "Grassland",
    40: "Cropland",
    50: "Built-up",
    60: "Bare / sparse vegetation",
    80: "Permanent water"
}

# Colours loosely follow the official ESA WorldCover colour scheme
ESA_CLASS_COLORS: dict[int, str] = {
    10: "#1a7d26",   # trees
    30: "#a4d65e",   # grass
    40: "#c8a951",   # crops
    50: "#d73027",   # built-up / urban
    60: "#d9c99e",   # bare land
    80: "#2196f3",   # water
}

# Default class colour when a code isn't in the dictionary above
DEFAULT_COLOR = "#888888"

# When the user clicks the map we draw a small bounding-box rectangle
# around the selected point. BBOX_HALF is half the side length in degrees.
BBOX_HALF = 0.008   # ≈ 0.8 km at Nuremberg's latitude

# We show a random sample of the data on the map (too many dots = slow)
MAP_SAMPLE_SIZE = 5_000


# ─── 5. HELPER FUNCTIONS ───────────────────────────────────────────────────────

def get_nearest_row(lat: float, lon: float) -> pd.Series:
    """
    Given a latitude and longitude, return the DataFrame row whose
    coordinates are closest to that point.
    Uses the pre-built KDTree for speed (O(log n) instead of O(n)).
    """
    _, idx = tree.query([[lat, lon]], k=1)
    return df.iloc[idx[0][0]]


def make_bounding_box(lat: float, lon: float):
    """
    Return the five corner coordinates (closed polygon) of a small square
    centred on (lat, lon).  Used to draw a highlight rectangle on the map.
    Returns: (list_of_lats, list_of_lons)
    """
    lats = [lat - BBOX_HALF, lat + BBOX_HALF,
            lat + BBOX_HALF, lat - BBOX_HALF,
            lat - BBOX_HALF]
    lons = [lon - BBOX_HALF, lon - BBOX_HALF,
            lon + BBOX_HALF, lon + BBOX_HALF,
            lon - BBOX_HALF]
    return lats, lons


def label_to_color(label: int) -> str:
    """Return the hex colour string for a given ESA class code."""
    return ESA_CLASS_COLORS.get(label, DEFAULT_COLOR)


def label_to_name(label: int) -> str:
    """Return the human-readable name for a given ESA class code."""
    return ESA_CLASS_NAMES.get(label, f"Unknown ({label})")


def classify_point(row: pd.Series) -> dict:
    """
    A simple rule-based 'model' that assigns land-cover labels to a single
    data point based on its Sentinel-2 spectral features.

    Rules (very simplified):
    ─────────────────────────
    • High SWIR (B11 > 0.25)  →  Built-up  (urban surfaces reflect SWIR)
    • High NDVI (> 0.4)       →  Tree cover (dense vegetation)
    • Medium NDVI (0.15–0.4)  →  Grassland / cropland
    • Otherwise               →  Bare / sparse vegetation

    Returns a dict with predicted labels and whether change occurred.
    """
    def _classify(b11: float, ndvi: float) -> int:
        if b11 > 0.25:
            return 50   # Built-up
        elif ndvi > 0.40:
            return 10   # Tree cover
        elif ndvi > 0.15:
            return 30   # Grassland
        else:
            return 60   # Bare land

    # Read feature columns – fall back to 0 if column doesn't exist
    b11_2020  = row.get("b11_2020",  0.0)
    b11_2021  = row.get("b11_2021",  0.0)
    ndvi_2020 = row.get("ndvi_2020", 0.0)
    ndvi_2021 = row.get("ndvi_2021", 0.0)

    label_2020 = _classify(b11_2020, ndvi_2020)
    label_2021 = _classify(b11_2021, ndvi_2021)

    return {
        "label_2020": label_2020,
        "label_2021": label_2021,
        "changed":    label_2020 != label_2021,
        # Confidence proxy: how far is NDVI from the decision boundary 0.4?
        "conf_2020":  float(np.clip(abs(ndvi_2020) / 0.6, 0, 1)),
        "conf_2021":  float(np.clip(abs(ndvi_2021) / 0.6, 0, 1)),
    }


# ─── 6. MAP BUILDER ────────────────────────────────────────────────────────────

def build_map(lat: float, lon: float, view_mode: str) -> go.Figure:
    """
    Build and return a Plotly Mapbox scatter map.

    Parameters
    ──────────
    lat, lon  : coordinates of the selected / search point
    view_mode : one of  '2020' | '2021' | 'Change'

    What gets drawn
    ───────────────
    Layer 1 – coloured dots (random sample of all data points)
    Layer 2 – big red marker pin at (lat, lon)
    Layer 3 – white bounding-box rectangle around the pin
    """
    # Take a random sample so the map stays responsive
    sample = df.sample(min(MAP_SAMPLE_SIZE, len(df)), random_state=42)

    # ── Determine dot colours based on the chosen view mode ──────────────────
    if view_mode == "2020":
        # Colour each dot by its 2020 land-cover label
        dot_colors = sample["label_2020"].apply(label_to_color)
        title_text = "Land Cover 2020"

    elif view_mode == "2021":
        # Colour each dot by its 2021 land-cover label
        dot_colors = sample["label_2021"].apply(label_to_color)
        title_text = "Land Cover 2021"

    else:   # "Change"
        # Red = changed,  teal = stable
        # We use delta_built_up if available; otherwise compare labels directly
        if "delta_built_up" in sample.columns:
            changed = sample["delta_built_up"].abs() > 0.05
        else:
            changed = sample["label_2020"] != sample["label_2021"]
        dot_colors = changed.map({True: "#ff4d4d", False: "#26a69a"})
        title_text = "Change Detection (2020 → 2021)"

    # ── Layer 1: data dots ────────────────────────────────────────────────────
    data_trace = go.Scattermapbox(
        lat    = sample["latitude"],
        lon    = sample["longitude"],
        mode   = "markers",
        name   = "Data points",
        marker = dict(
            size    = 5,
            color   = dot_colors,
            opacity = 0.75,
        ),
        hovertemplate = (
            "Lat: %{lat:.5f}<br>"
            "Lon: %{lon:.5f}<extra></extra>"
        ),
    )

    # ── Layer 2: selected-point pin ───────────────────────────────────────────
    pin_trace = go.Scattermapbox(
        lat    = [lat],
        lon    = [lon],
        mode   = "markers+text",
        name   = "Selected",
        text   = ["▼"],
        textposition = "top center",
        marker = dict(size=18, color="#ff1744", symbol="circle"),
        hovertemplate = f"Selected<br>Lat: {lat:.5f}<br>Lon: {lon:.5f}<extra></extra>",
    )

    # ── Layer 3: bounding-box rectangle ──────────────────────────────────────
    box_lats, box_lons = make_bounding_box(lat, lon)
    bbox_trace = go.Scattermapbox(
        lat  = box_lats,
        lon  = box_lons,
        mode = "lines",
        name = "Area",
        line = dict(color="white", width=2),
        hoverinfo = "skip",
    )

    # ── Assemble the figure ───────────────────────────────────────────────────
    fig = go.Figure(data=[data_trace, pin_trace, bbox_trace])

    fig.update_layout(
        title = dict(text=title_text, font=dict(size=15, color="#e0e0e0")),
        mapbox = dict(
            style  = "open-street-map",      # map style
            center = dict(lat=lat, lon=lon),
            zoom   = 13,
        ),
        margin       = dict(l=0, r=0, t=40, b=0),
        height       = 540,
        paper_bgcolor = "#0d1117",
        showlegend   = False,
    )

    return fig


#PAGE CONFIG & GLOBAL CSS ---------------------------------------------------------------------------------------------------
st.set_page_config(
    page_title = "Nuremberg Land Cover Analysis",
    page_icon  = "🔍",
    layout     = "wide",
)


st.markdown("""
<style>
/* ── Google font import ── */
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

/* ── Root colours & fonts ── */
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

/* ── Header strip ── */
.main-header {
    background: linear-gradient(135deg, #0d1117 0%, #1c2333 100%);
    border-bottom: 1px solid var(--border);
    padding: 1.2rem 1.6rem;
    margin-bottom: 1.2rem;
    border-radius: 0 0 10px 10px;
}
.main-header h1 {
    font-size: 1.7rem;
    font-weight: 700;
    margin: 0;
    background: linear-gradient(90deg, #58a6ff, #3fb950);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -0.5px;
}
.main-header p {
    color: var(--text-muted);
    margin: 0.2rem 0 0;
    font-size: 0.9rem;
}

/* ── Metric cards ── */
.metric-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.75rem;
    transition: border-color 0.2s;
}
.metric-card:hover { border-color: var(--accent); }
.metric-card .label {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: var(--text-muted);
    margin-bottom: 0.25rem;
}
.metric-card .value {
    font-size: 1.5rem;
    font-weight: 700;
    color: var(--text-main);
    font-family: 'JetBrains Mono', monospace;
}
.metric-card .sub {
    font-size: 0.75rem;
    color: var(--text-muted);
    margin-top: 0.15rem;
}

/* ── Status banners ── */
.banner-change {
    background: rgba(248,81,73,0.12);
    border: 1px solid #f85149;
    border-left: 4px solid #f85149;
    border-radius: 8px;
    padding: 1rem 1.2rem;
    margin: 0.5rem 0;
}
.banner-stable {
    background: rgba(63,185,80,0.10);
    border: 1px solid #3fb950;
    border-left: 4px solid #3fb950;
    border-radius: 8px;
    padding: 1rem 1.2rem;
    margin: 0.5rem 0;
}
.banner-change .title  { color: #f85149; font-weight: 700; font-size: 0.95rem; margin-bottom: 0.4rem; }
.banner-stable .title  { color: #3fb950; font-weight: 700; font-size: 0.95rem; margin-bottom: 0.4rem; }
.banner-change .detail { color: var(--text-main); font-size: 0.85rem; }
.banner-stable .detail { color: var(--text-main); font-size: 0.85rem; }

/* ── Legend item ── */
.legend-item {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    padding: 0.35rem 0.5rem;
    border-radius: 6px;
    margin-bottom: 0.3rem;
    transition: background 0.15s;
    cursor: default;
}
.legend-item:hover { background: var(--bg-card); }
.legend-swatch {
    width: 14px;
    height: 14px;
    border-radius: 3px;
    flex-shrink: 0;
}
.legend-label {
    font-size: 0.82rem;
    color: var(--text-main);
}

/* ── Section headings ── */
.section-heading {
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: var(--text-muted);
    border-bottom: 1px solid var(--border);
    padding-bottom: 0.4rem;
    margin: 1rem 0 0.6rem;
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


# PAGE HEADER ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
  <h1> Nuremberg Urban Land Cover Analysis </h1>
  <p>Explore satellite-derived land cover change between 2020 and 2021 · ESA WorldCover + Sentinel-2</p>
</div>
""", unsafe_allow_html=True)


# TABS ────────────────────────────────────────────────────────────────────

tab_map, tab_analysis, tab_stats = st.tabs([
    "📍  Search Location",
    "🔬  Model results",
    "📊  Data Summary",
])


# TAB 1 ------

with tab_map:

    # ── Split into left sidebar controls and right map area ──────────────────
    ctrl_col, map_col = st.columns([1, 3], gap="medium")

    # ── Left column: controls ─────────────────────────────────────────────────
    with ctrl_col:

        st.markdown('<div class="section-heading">Coordinates</div>', unsafe_allow_html=True)

        # Latitude / longitude number inputs.
        # value= sets the default.  step= controls how much +/– changes it.
        lat = st.number_input(
            "Latitude",
            value  = 49.45,
            min_value = 49.30,
            max_value = 49.60,
            step   = 0.001,
            format = "%.5f",
            help   = "WGS-84 decimal degrees latitude (Nuremberg range: 49.30 – 49.60)"
        )
        lon = st.number_input(
            "Longitude",
            value  = 11.07,
            min_value = 10.90,
            max_value = 11.25,
            step   = 0.001,
            format = "%.5f",
            help   = "WGS-84 decimal degrees longitude (Nuremberg range: 10.90 – 11.25)"
        )

        st.markdown('<div class="section-heading">View Mode</div>', unsafe_allow_html=True)

        # Radio buttons to choose what the map colouring shows
        view_mode = st.radio(
            label  = "Colour dots by:",
            options= ["2020", "2021", "Change"],
            index  = 0,
            help   = (
                "2020 / 2021 → colour each point by its ESA land-cover class in that year.\n"
                "Change      → red = changed class, teal = stable."
            ),
        )

        # ── Legend ─────────────────────────────────────────────────────────────
        st.markdown('<div class="section-heading">Legend</div>', unsafe_allow_html=True)

        if view_mode == "Change":
            # Special 2-item legend for the change view
            for colour, name in [("#ff4d4d", "Changed"), ("#1b11d0", "Stable")]:
                st.markdown(
                    f'<div class="legend-item">'
                    f'<div class="legend-swatch" style="background:{colour}"></div>'
                    f'<span class="legend-label">{name}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
        else:
            # Full ESA class legend
            for code, name in ESA_CLASS_NAMES.items():
                colour = ESA_CLASS_COLORS.get(code, DEFAULT_COLOR)
                st.markdown(
                    f'<div class="legend-item">'
                    f'<div class="legend-swatch" style="background:{colour}"></div>'
                    f'<span class="legend-label">{name}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    #  Right column: map
    with map_col:

        st.plotly_chart(
            build_map(lat, lon, view_mode),
            use_container_width=True,
            config={"scrollZoom": True},   # allow mouse-wheel zoom
        )

        # Tiny info row below the map
        nearest = get_nearest_row(lat, lon)
        st.caption(
            f"📌 Nearest data point: "
            f"lat {nearest['latitude']:.5f}, lon {nearest['longitude']:.5f}  ·  "
            f"ESA 2020: **{label_to_name(int(nearest.get('label_2020', 0)))}**  ·  "
            f"ESA 2021: **{label_to_name(int(nearest.get('label_2021', 0)))}**"
        )


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 2 – POINT ANALYSIS
#  Shows a detailed spectral and classification analysis for the point that
#  was selected on the map (or the nearest point to the entered coordinates).
# ══════════════════════════════════════════════════════════════════════════════
with tab_analysis:

    # Get the data row nearest to the user's coordinates
    row  = get_nearest_row(lat, lon)
    pred = classify_point(row)

    left_col, right_col = st.columns([1, 1], gap="large")

    # ── Left: classification result & metadata ────────────────────────────────
    with left_col:

        st.markdown('<div class="section-heading">Classification Result</div>',
                    unsafe_allow_html=True)

        # Show a red "CHANGE" banner or a green "STABLE" banner
        if pred["changed"]:
            st.markdown(f"""
            <div class="banner-change">
              <div class="title">⚠️  LAND COVER CHANGE DETECTED</div>
              <div class="detail">
                <strong>2020:</strong> {label_to_name(pred['label_2020'])}<br>
                <strong>2021:</strong> {label_to_name(pred['label_2021'])}
              </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="banner-stable">
              <div class="title">✅  NO CHANGE — STABLE LAND COVER</div>
              <div class="detail">
                <strong>Both years:</strong> {label_to_name(pred['label_2020'])}
              </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<div class="section-heading">Spectral Features</div>',
                    unsafe_allow_html=True)

        # Display key feature values as metric cards.
        # .get("column", default) avoids crashes if the column doesn't exist.
        features = [
            ("NDVI 2020",  f"{row.get('ndvi_2020', 0):.4f}",  "Vegetation index (higher = greener)"),
            ("NDVI 2021",  f"{row.get('ndvi_2021', 0):.4f}",  "Vegetation index (higher = greener)"),
            ("SWIR B11 2020", f"{row.get('b11_2020', 0):.4f}", "Short-wave infrared (built-up proxy)"),
            ("SWIR B11 2021", f"{row.get('b11_2021', 0):.4f}", "Short-wave infrared (built-up proxy)"),
        ]

        for label, value, sub in features:
            st.markdown(f"""
            <div class="metric-card">
              <div class="label">{label}</div>
              <div class="value">{value}</div>
              <div class="sub">{sub}</div>
            </div>
            """, unsafe_allow_html=True)

    # ── Right: confidence bar chart ───────────────────────────────────────────
    with right_col:

        st.markdown('<div class="section-heading">Model Confidence</div>',
                    unsafe_allow_html=True)

        # Two bars showing how confident the rule-based classifier is for each year.
        # Confidence is a proxy derived from how far the NDVI is from the threshold.
        conf_fig = go.Figure()

        conf_fig.add_trace(go.Bar(
            x     = ["2020", "2021"],
            y     = [pred["conf_2020"] * 100, pred["conf_2021"] * 100],
            text  = [f"{pred['conf_2020']*100:.0f}%", f"{pred['conf_2021']*100:.0f}%"],
            textposition = "outside",
            textfont     = dict(color="#e6edf3", size=14, family="JetBrains Mono"),
            marker_color = ["#58a6ff", "#3fb950"],
            marker_line_width = 0,
        ))

        conf_fig.update_layout(
            paper_bgcolor = "#0d1117",
            plot_bgcolor  = "#0d1117",
            font          = dict(color="#8b949e", family="Space Grotesk"),
            xaxis         = dict(showgrid=False),
            yaxis         = dict(range=[0, 115], showgrid=True,
                                 gridcolor="#21262d", ticksuffix="%"),
            height        = 260,
            margin        = dict(l=10, r=10, t=20, b=10),
            showlegend    = False,
        )

        st.plotly_chart(conf_fig, use_container_width=True)

        # ── Spectral band comparison (radar / spider chart) ──────────────────
        st.markdown('<div class="section-heading">Band Profile Comparison</div>',
                    unsafe_allow_html=True)

        # Collect available band columns (b3, b4, b8, b11) for 2020 and 2021
        band_labels = []
        vals_2020   = []
        vals_2021   = []

        for band in ["b3", "b4", "b8", "b11"]:
            col_2020 = f"{band}_2020"
            col_2021 = f"{band}_2021"
            if col_2020 in row.index and col_2021 in row.index:
                band_labels.append(band.upper())
                vals_2020.append(float(row[col_2020]))
                vals_2021.append(float(row[col_2021]))

        if band_labels:
            radar_fig = go.Figure()

            # Close the polygon by repeating the first value at the end
            theta = band_labels + [band_labels[0]]

            radar_fig.add_trace(go.Scatterpolar(
                r     = vals_2020 + [vals_2020[0]],
                theta = theta,
                name  = "2020",
                line  = dict(color="#58a6ff", width=2),
                fill  = "toself",
                fillcolor = "rgba(88,166,255,0.15)",
            ))

            radar_fig.add_trace(go.Scatterpolar(
                r     = vals_2021 + [vals_2021[0]],
                theta = theta,
                name  = "2021",
                line  = dict(color="#3fb950", width=2),
                fill  = "toself",
                fillcolor = "rgba(63,185,80,0.15)",
            ))

            radar_fig.update_layout(
                polar = dict(
                    bgcolor   = "#161b22",
                    angularaxis = dict(color="#8b949e", gridcolor="#30363d"),
                    radialaxis  = dict(color="#8b949e", gridcolor="#30363d",
                                       showticklabels=True),
                ),
                paper_bgcolor = "#0d1117",
                font      = dict(color="#8b949e", family="Space Grotesk"),
                legend    = dict(font=dict(color="#e6edf3")),
                height    = 320,
                margin    = dict(l=40, r=40, t=30, b=10),
            )

            st.plotly_chart(radar_fig, use_container_width=True)
        else:
            st.info("Band columns (b3, b4, b8, b11) not found in the dataset.")


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 3 – AREA SUMMARY
#  High-level statistics about the whole Nuremberg study area.
#  Numbers here are placeholders that you can replace with real aggregations
#  from the loaded DataFrame once you know the column structure.
# ══════════════════════════════════════════════════════════════════════════════
with tab_stats:

    st.markdown('<div class="section-heading">Land Cover Area (km²)</div>',
                unsafe_allow_html=True)

    # ── Summary table ─────────────────────────────────────────────────────────
    # Replace these hard-coded values with  df.groupby("label_2020").size()
    # calculations once the data schema is confirmed.
    summary = pd.DataFrame({
        "Class":    ["Built-up", "Tree cover", "Cropland", "Grassland",
                     "Water", "Bare veg."],
        "Code":     [50, 10, 40, 30, 80, 60],
        "Area 2020 (km²)": [87.3, 31.5, 42.1, 18.7, 4.2, 2.1],
        "Area 2021 (km²)": [89.1, 31.2, 39.8, 18.9, 4.2, 2.7],
    })
    summary["Change (km²)"] = (
        summary["Area 2021 (km²)"] - summary["Area 2020 (km²)"]
    ).round(1)

    # Colour the Change column: red for shrinkage, green for growth
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

    # ── Grouped bar chart: 2020 vs 2021 per class ─────────────────────────────
    st.markdown('<div class="section-heading">Area Comparison by Class</div>',
                unsafe_allow_html=True)

    bar_fig = go.Figure()

    bar_fig.add_trace(go.Bar(
        name         = "2020",
        x            = summary["Class"],
        y            = summary["Area 2020 (km²)"],
        marker_color = "#58a6ff",
        marker_line_width = 0,
    ))

    bar_fig.add_trace(go.Bar(
        name         = "2021",
        x            = summary["Class"],
        y            = summary["Area 2021 (km²)"],
        marker_color = "#3fb950",
        marker_line_width = 0,
    ))

    bar_fig.update_layout(
        barmode       = "group",
        paper_bgcolor = "#0d1117",
        plot_bgcolor  = "#0d1117",
        font          = dict(color="#8b949e", family="Space Grotesk"),
        xaxis         = dict(showgrid=False),
        yaxis         = dict(title="km²", gridcolor="#21262d"),
        legend        = dict(font=dict(color="#e6edf3")),
        height        = 340,
        margin        = dict(l=10, r=10, t=20, b=10),
    )

    st.plotly_chart(bar_fig, use_container_width=True)

    # ── Change bar (delta) ────────────────────────────────────────────────────
    st.markdown('<div class="section-heading">Net Change per Class</div>',
                unsafe_allow_html=True)

    delta_fig = go.Figure(go.Bar(
        x            = summary["Class"],
        y            = summary["Change (km²)"],
        text         = [f"{v:+.1f}" for v in summary["Change (km²)"]],
        textposition = "outside",
        textfont     = dict(color="#e6edf3", size=12),
        marker_color = [
            "#f85149" if v < 0 else "#3fb950"
            for v in summary["Change (km²)"]
        ],
        marker_line_width = 0,
    ))

    delta_fig.update_layout(
        paper_bgcolor = "#0d1117",
        plot_bgcolor  = "#0d1117",
        font          = dict(color="#8b949e", family="Space Grotesk"),
        xaxis         = dict(showgrid=False),
        yaxis         = dict(title="Δ km²", gridcolor="#21262d",
                             zeroline=True, zerolinecolor="#30363d"),
        height        = 300,
        margin        = dict(l=10, r=10, t=20, b=10),
    )

    st.plotly_chart(delta_fig, use_container_width=True)


# ─── 10. FOOTER ──────────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-top:2rem; padding:0.8rem 1rem; border-top:1px solid #21262d;
            color:#8b949e; font-size:0.78rem; text-align:center;">
  Data: ESA WorldCover 10 m · Sentinel-2 L2A · Nuremberg area · 2020–2021<br>
  Built with Streamlit &amp; Plotly
</div>
""", unsafe_allow_html=True)
