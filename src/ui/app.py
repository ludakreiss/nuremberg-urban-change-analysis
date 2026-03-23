import re
import requests
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np


# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Nuremberg Land Cover Explorer",
    page_icon="🌍",
    layout="wide",
)


# =========================================================
# CONSTANTS
# =========================================================
NUREMBERG_LAT = 49.452
NUREMBERG_LON = 11.077

NUREMBERG_BOUNDS = {
    "min_lat": 49.38,
    "max_lat": 49.52,
    "min_lon": 10.95,
    "max_lon": 11.15,
}

ESA_CLASSES = {
    10: {"name": "Tree cover", "color": "#006400"},
    30: {"name": "Grassland", "color": "#50E80A"},
    40: {"name": "Cropland", "color": "#BC721D"},
    50: {"name": "Built-up", "color": "#111111"},
    60: {"name": "Bare / sparse vegetation", "color": "#C0EC70"},
    80: {"name": "Permanent water", "color": "#358FE9"},
}

QUICK_LOCATIONS = {
    "Old Town": (49.4543, 11.0775),
    "Industrial North": (49.4900, 11.0600),
    "Reichswald Forest": (49.4100, 11.1300),
    "Airport": (49.4970, 11.0780),
}

BBOX_HALF = 0.008


# =========================================================
# OPTIONAL DATA LOADER (KEEP FOR LATER)
# =========================================================
# @st.cache_data
# def load_data():
#     url = "https://huggingface.co/datasets/YOUR_USERNAME/YOUR_DATASET/resolve/main/nuremberg_dataset_final.csv"
#     return pd.read_csv(url)


# =========================================================
# STYLING
# =========================================================
st.markdown(
    """
    <style>
        .stApp {
            background-color: #0b1220;
        }

        .main-title-box {
            background: linear-gradient(135deg, #111827 0%, #172554 55%, #1e3a8a 100%);
            border: 1px solid #334155;
            border-radius: 18px;
            padding: 1.4rem 1.6rem;
            margin-bottom: 1rem;
        }

        .main-title {
            color: #f8fafc;
            font-size: 2rem;
            font-weight: 800;
            margin-bottom: 0.2rem;
        }

        .main-subtitle {
            color: #cbd5e1;
            font-size: 0.96rem;
            margin: 0;
        }

        .section-card {
            background: #111827;
            border: 1px solid #334155;
            border-radius: 14px;
            padding: 1rem 1.1rem;
            margin-bottom: 1rem;
        }

        .legend-row {
            display: flex;
            align-items: center;
            gap: 8px;
            margin-bottom: 8px;
            color: #e2e8f0;
            font-size: 0.9rem;
        }

        .legend-dot {
            width: 14px;
            height: 14px;
            border-radius: 50%;
            display: inline-block;
            border: 1px solid rgba(255,255,255,0.15);
        }

        .metric-box {
            background: #111827;
            border: 1px solid #334155;
            border-radius: 12px;
            padding: 0.9rem 1rem;
            margin-bottom: 0.8rem;
        }

        .metric-title {
            color: #93c5fd;
            font-size: 0.85rem;
            margin-bottom: 0.35rem;
        }

        .metric-value {
            color: #f8fafc;
            font-size: 1.15rem;
            font-weight: 700;
        }

        .small-note {
            color: #94a3b8;
            font-size: 0.82rem;
        }

        [data-testid="metric-container"] {
            background: #111827 !important;
            border: 1px solid #334155 !important;
            border-radius: 12px !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# HELPERS
# =========================================================
def in_bounds(lat: float, lon: float) -> bool:
    b = NUREMBERG_BOUNDS
    return b["min_lat"] <= lat <= b["max_lat"] and b["min_lon"] <= lon <= b["max_lon"]


def try_parse_coordinates(text: str):
    cleaned = re.sub(r"[°NSEWnsew]", " ", text)
    cleaned = cleaned.replace(",", " ")
    numbers = re.findall(r"-?\d+\.?\d*", cleaned)

    if len(numbers) == 2:
        try:
            lat = float(numbers[0])
            lon = float(numbers[1])
            if -90 <= lat <= 90 and -180 <= lon <= 180:
                return lat, lon
        except ValueError:
            pass

    return None


def geocode_place(name: str) -> dict:
    full_query = f"{name}, Nuremberg, Germany"

    try:
        r = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={
                "q": full_query,
                "format": "json",
                "limit": 1,
                "viewbox": "10.95,49.52,11.15,49.38",
                "bounded": 1,
                "accept-language": "en",
            },
            headers={"User-Agent": "NurembergLandCoverApp/1.0"},
            timeout=6,
        )
        results = r.json()

        if not results:
            r2 = requests.get(
                "https://nominatim.openstreetmap.org/search",
                params={
                    "q": full_query,
                    "format": "json",
                    "limit": 1,
                    "accept-language": "en",
                },
                headers={"User-Agent": "NurembergLandCoverApp/1.0"},
                timeout=6,
            )
            results = r2.json()

        if not results:
            return {
                "lat": None,
                "lon": None,
                "display": None,
                "error": f"Could not find '{name}' near Nuremberg.",
            }

        best = results[0]
        return {
            "lat": float(best["lat"]),
            "lon": float(best["lon"]),
            "display": best["display_name"],
            "error": None,
        }

    except requests.exceptions.Timeout:
        return {
            "lat": None,
            "lon": None,
            "display": None,
            "error": "Search timed out. Please try again.",
        }
    except Exception as e:
        return {
            "lat": None,
            "lon": None,
            "display": None,
            "error": f"Search failed: {str(e)}",
        }


def smart_search(query: str) -> dict:
    query = query.strip()

    if not query:
        return {
            "lat": None,
            "lon": None,
            "display": None,
            "error": "Please type a place or coordinates.",
            "search_type": None,
        }

    parsed = try_parse_coordinates(query)
    if parsed:
        lat, lon = parsed
        return {
            "lat": lat,
            "lon": lon,
            "display": f"{lat:.4f}° N, {lon:.4f}° E",
            "error": None,
            "search_type": "coords",
        }

    result = geocode_place(query)
    result["search_type"] = "place"
    return result


def esa_color(class_id: int) -> str:
    return ESA_CLASSES.get(class_id, {}).get("color", "#64748b")


def esa_name(class_id: int) -> str:
    return ESA_CLASSES.get(class_id, {}).get("name", "Unknown")


def make_location_bbox(lat: float, lon: float, half: float = BBOX_HALF):
    box_lats = [lat - half, lat + half, lat + half, lat - half, lat - half]
    box_lons = [lon - half, lon - half, lon + half, lon + half, lon - half]
    return box_lats, box_lons


# =========================================================
# PLACEHOLDER MODEL
# =========================================================
def mock_predict(lat: float, lon: float) -> dict:
    dist = ((lat - NUREMBERG_LAT) ** 2 + (lon - NUREMBERG_LON) ** 2) ** 0.5

    if dist < 0.05:
        l2020, l2021 = 50, 50
    elif dist < 0.07:
        l2020, l2021 = 40, 50
    else:
        l2020, l2021 = 40, 40

    rng = np.random.default_rng(seed=int(abs(lat * 1000 + lon * 100)))

    return {
        "label_2020": l2020,
        "label_2021": l2021,
        "conf_2020": round(float(rng.uniform(0.68, 0.92)), 2),
        "conf_2021": round(float(rng.uniform(0.70, 0.94)), 2),
        "changed": l2020 != l2021,
    }


# =========================================================
# MAP
# =========================================================
def make_scatter_map(
    center_lat: float,
    center_lon: float,
    year_view: str,
    marker_lat: float = None,
    marker_lon: float = None,
    pred: dict = None,
) -> go.Figure:
    np.random.seed(42)
    n = 1200

    lats = np.random.uniform(NUREMBERG_BOUNDS["min_lat"], NUREMBERG_BOUNDS["max_lat"], n)
    lons = np.random.uniform(NUREMBERG_BOUNDS["min_lon"], NUREMBERG_BOUNDS["max_lon"], n)

    labels_2020, labels_2021 = [], []

    for la, lo in zip(lats, lons):
        d = ((la - NUREMBERG_LAT) ** 2 + (lo - NUREMBERG_LON) ** 2) ** 0.5
        if d < 0.05:
            l20, l21 = 50, 50
        elif d < 0.07:
            prob = np.random.rand()
            l20 = 40 if prob > 0.3 else 50
            l21 = 50
        else:
            l20 = l21 = np.random.choice([10, 30, 40])

        labels_2020.append(l20)
        labels_2021.append(l21)

    labels_2020 = np.array(labels_2020)
    labels_2021 = np.array(labels_2021)

    if year_view == "2020":
        title_tag = "2020 Land Cover"
        dot_colors = [esa_color(l) for l in labels_2020]
        dot_text = [esa_name(l) for l in labels_2020]
    elif year_view == "2021":
        title_tag = "2021 Land Cover"
        dot_colors = [esa_color(l) for l in labels_2021]
        dot_text = [esa_name(l) for l in labels_2021]
    else:
        title_tag = "Land Cover Change 2020 → 2021"
        changed_mask = labels_2020 != labels_2021
        dot_colors = ["#ef4444" if c else "#1d4ed8" for c in changed_mask]
        dot_text = [
            f"CHANGED: {esa_name(l20)} → {esa_name(l21)}" if c else f"Stable: {esa_name(l20)}"
            for c, l20, l21 in zip(changed_mask, labels_2020, labels_2021)
        ]

    fig = go.Figure()

    fig.add_trace(
        go.Scattermapbox(
            lat=lats,
            lon=lons,
            mode="markers",
            marker=dict(size=6, color=dot_colors, opacity=0.75),
            text=dot_text,
            hovertemplate="<b>%{text}</b><br>Lat: %{lat:.4f}<br>Lon: %{lon:.4f}<extra></extra>",
            name=title_tag,
        )
    )

    nb = NUREMBERG_BOUNDS
    fig.add_trace(
        go.Scattermapbox(
            lat=[nb["min_lat"], nb["max_lat"], nb["max_lat"], nb["min_lat"], nb["min_lat"]],
            lon=[nb["min_lon"], nb["min_lon"], nb["max_lon"], nb["max_lon"], nb["min_lon"]],
            mode="lines",
            line=dict(color="#f43f5e", width=2),
            hoverinfo="skip",
            name="Study area boundary",
        )
    )

    if marker_lat is not None and marker_lon is not None:
        bbox_lats, bbox_lons = make_location_bbox(marker_lat, marker_lon)
        fig.add_trace(
            go.Scattermapbox(
                lat=bbox_lats,
                lon=bbox_lons,
                mode="lines",
                line=dict(color="#ffffff", width=2.5),
                hoverinfo="skip",
                name="Search area",
            )
        )

    if marker_lat is not None and marker_lon is not None and pred is not None:
        if year_view == "Change":
            hover = (
                f"<b>{'⚠️ Changed' if pred['changed'] else 'Stable'}</b><br>"
                f"{esa_name(pred['label_2020'])} → {esa_name(pred['label_2021'])}"
            )
        elif year_view == "2020":
            hover = (
                f"<b>Location – 2020</b><br>"
                f"{esa_name(pred['label_2020'])}<br>"
                f"Confidence: {pred['conf_2020']*100:.0f}%"
            )
        else:
            hover = (
                f"<b>Location – 2021</b><br>"
                f"{esa_name(pred['label_2021'])}<br>"
                f"Confidence: {pred['conf_2021']*100:.0f}%"
            )

        fig.add_trace(
            go.Scattermapbox(
                lat=[marker_lat],
                lon=[marker_lon],
                mode="markers",
                marker=dict(size=15, color="#f8fafc", symbol="star"),
                text=[hover],
                hovertemplate="%{text}<extra></extra>",
                name="Selected location",
            )
        )

    fig.update_layout(
        mapbox=dict(
            style="open-street-map",
            center=dict(lat=center_lat, lon=center_lon),
            zoom=11 if marker_lat is None else 14,
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        height=560,
        paper_bgcolor="#0b1220",
        font_color="#e2e8f0",
        title=dict(
            text=f"<b>{title_tag}</b> · Nuremberg",
            x=0.01,
            font=dict(size=16, color="#93c5fd"),
        ),
        legend=dict(
            bgcolor="#111827",
            bordercolor="#334155",
            borderwidth=1,
            font=dict(color="#e2e8f0"),
        ),
    )

    return fig


def make_bar_chart(pred: dict) -> go.Figure:
    fig = go.Figure(
        go.Bar(
            x=["2020", "2021"],
            y=[pred["conf_2020"] * 100, pred["conf_2021"] * 100],
            marker_color=["#2563eb", "#06b6d4"],
            text=[f"{pred['conf_2020']*100:.0f}%", f"{pred['conf_2021']*100:.0f}%"],
            textposition="outside",
            textfont=dict(color="#e2e8f0"),
        )
    )

    fig.update_layout(
        title=dict(text="Model confidence", font=dict(color="#93c5fd", size=14)),
        yaxis=dict(range=[0, 110], ticksuffix="%", gridcolor="#334155", color="#94a3b8"),
        xaxis=dict(color="#94a3b8"),
        paper_bgcolor="#111827",
        plot_bgcolor="#111827",
        margin=dict(l=10, r=10, t=40, b=10),
        height=240,
        font=dict(color="#e2e8f0"),
    )

    return fig


# =========================================================
# SESSION STATE
# =========================================================
default_state = {
    "sel_lat": NUREMBERG_LAT,
    "sel_lon": NUREMBERG_LON,
    "has_marker": False,
    "pred": None,
    "found_name": None,
}

for key, value in default_state.items():
    if key not in st.session_state:
        st.session_state[key] = value


# =========================================================
# HEADER
# =========================================================
st.markdown(
    """
    <div class="main-title-box">
        <div class="main-title">🌍 Nuremberg Land Cover Explorer</div>
        <p class="main-subtitle">
            Interactive urban change exploration using ESA WorldCover, Sentinel-2, and machine learning predictions.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# TABS
# =========================================================
tab_search, tab_model, tab_summary = st.tabs(
    ["📍 Search", "🤖 Model Output", "📊 Summary"]
)


# =========================================================
# TAB 1
# =========================================================
with tab_search:
    st.subheader("Search a location in Nuremberg")

    col_search, col_button = st.columns([5, 1])

    with col_search:
        query = st.text_input(
            "Search",
            placeholder='Enter a place or coordinates, e.g. "Maxfeld" or "49.4415, 11.0797"',
            label_visibility="collapsed",
        )
        st.markdown(
            '<div class="small-note">Accepted input: place names, decimal coordinates, or coordinates with ° N / ° E.</div>',
            unsafe_allow_html=True,
        )

    with col_button:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        do_search = st.button("Search", use_container_width=True)

    st.markdown("**Quick locations**")
    quick_cols = st.columns(len(QUICK_LOCATIONS))

    for col, (label, (qlat, qlon)) in zip(quick_cols, QUICK_LOCATIONS.items()):
        if col.button(label, use_container_width=True):
            st.session_state.sel_lat = qlat
            st.session_state.sel_lon = qlon
            st.session_state.has_marker = True
            st.session_state.pred = mock_predict(qlat, qlon)
            st.session_state.found_name = label
            st.rerun()

    year_view = st.radio(
        "Map view",
        options=["2020", "2021", "Change"],
        horizontal=True,
    )

    if do_search and query.strip():
        with st.spinner("Searching location..."):
            result = smart_search(query)

        if result["error"]:
            st.error(result["error"])
        elif not in_bounds(result["lat"], result["lon"]):
            st.warning(
                f"Found '{result['display'][:80]}' but it is outside the Nuremberg study area."
            )
        else:
            st.session_state.sel_lat = result["lat"]
            st.session_state.sel_lon = result["lon"]
            st.session_state.has_marker = True
            st.session_state.pred = mock_predict(result["lat"], result["lon"])
            st.session_state.found_name = result["display"]

            if result["search_type"] == "coords":
                st.success(f"Coordinates selected: {result['lat']:.4f}, {result['lon']:.4f}")
            else:
                st.success(f"Found: {result['display'][:100]}")

    col_map, col_side = st.columns([3, 1], gap="large")

    with col_map:
        c_lat = st.session_state.sel_lat if st.session_state.has_marker else NUREMBERG_LAT
        c_lon = st.session_state.sel_lon if st.session_state.has_marker else NUREMBERG_LON

        fig_map = make_scatter_map(
            center_lat=c_lat,
            center_lon=c_lon,
            year_view=year_view,
            marker_lat=st.session_state.sel_lat if st.session_state.has_marker else None,
            marker_lon=st.session_state.sel_lon if st.session_state.has_marker else None,
            pred=st.session_state.pred,
        )
        st.plotly_chart(fig_map, use_container_width=True)

    with col_side:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("#### Land cover legend")

        for _, info in ESA_CLASSES.items():
            st.markdown(
                f"""
                <div class="legend-row">
                    <span class="legend-dot" style="background:{info['color']}"></span>
                    <span>{info['name']}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("</div>", unsafe_allow_html=True)

        if st.session_state.has_marker and st.session_state.found_name:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown("#### Selected location")
            st.write(st.session_state.found_name[:120])
            st.caption(
                f"{st.session_state.sel_lat:.4f}° N, {st.session_state.sel_lon:.4f}° E"
            )
            st.markdown("</div>", unsafe_allow_html=True)


# =========================================================
# TAB 2
# =========================================================
with tab_model:
    st.subheader("Model output")

    if st.session_state.pred:
        pred = st.session_state.pred

        col_left, col_right = st.columns([1.2, 1], gap="large")

        with col_left:
            st.markdown(
                f"""
                <div class="metric-box">
                    <div class="metric-title">2020 prediction</div>
                    <div class="metric-value" style="color:{esa_color(pred['label_2020'])};">
                        {esa_name(pred['label_2020'])}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown(
                f"""
                <div class="metric-box">
                    <div class="metric-title">2021 prediction</div>
                    <div class="metric-value" style="color:{esa_color(pred['label_2021'])};">
                        {esa_name(pred['label_2021'])}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if pred["changed"]:
                st.error(
                    f"Change detected: {esa_name(pred['label_2020'])} → {esa_name(pred['label_2021'])}"
                )
            else:
                st.success(f"Stable land cover: {esa_name(pred['label_2020'])}")

        with col_right:
            st.plotly_chart(make_bar_chart(pred), use_container_width=True)

    else:
        st.info("Search for a location first to view model predictions.")

    st.subheader("Limitations")
    st.markdown(
        """
        <div class="section-card">
            <div class="small-note">
                • Features: Sentinel-2 bands B3, B4, B8, B11<br>
                • Labels: ESA WorldCover at 10 m resolution<br>
                • Predictions may be affected by cloud cover and seasonal variation<br>
                • Short temporal coverage limits long-term forecasting
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# =========================================================
# TAB 3
# =========================================================
with tab_summary:
    st.subheader("Data summary")

    summary_df = pd.DataFrame({
        "Land Cover": ["Built-up", "Cropland", "Tree cover", "Grassland", "Water"],
        "Area 2020 km²": [87.3, 42.1, 31.5, 18.7, 4.2],
        "Area 2021 km²": [89.1, 39.8, 31.2, 18.9, 4.2],
        "Change km²": [1.8, -2.3, -0.3, 0.2, 0.0],
    })

    def color_change(val):
        if val > 0:
            return "color: #22c55e; font-weight: bold"
        if val < 0:
            return "color: #ef4444; font-weight: bold"
        return "color: #94a3b8"

    st.dataframe(
        summary_df.style.map(color_change, subset=["Change km²"]),
        use_container_width=True,
        hide_index=True,
    )

    fig_summary = px.bar(
        summary_df,
        x="Land Cover",
        y=["Area 2020 km²", "Area 2021 km²"],
        barmode="group",
        color_discrete_map={
            "Area 2020 km²": "#2563eb",
            "Area 2021 km²": "#06b6d4",
        },
        title="Area by land cover class: 2020 vs 2021",
        labels={"value": "Area (km²)", "variable": "Year"},
    )

    fig_summary.update_layout(
        paper_bgcolor="#0b1220",
        plot_bgcolor="#111827",
        font_color="#e2e8f0",
        title_font_color="#93c5fd",
        legend_bgcolor="#111827",
        xaxis=dict(gridcolor="#334155"),
        yaxis=dict(gridcolor="#334155"),
        height=420,
    )

    st.plotly_chart(fig_summary, use_container_width=True)


# =========================================================
# FOOTER
# =========================================================
st.divider()
st.caption(
    "Data sources: ESA WorldCover 2020/2021 · Sentinel-2 MSI · "
    "Model backend: Random Forest / Logistic Regression · "
    "Storage: Hugging Face · App: Streamlit"
)