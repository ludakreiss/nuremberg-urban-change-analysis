import re
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import requests

# @st.cache_data   #Will need this to download data
# def load_data():
#     url = "https://huggingface.co/datasets/YOUR_USERNAME/YOUR_DATASET/resolve/main/nuremberg_dataset_final.csv"
#     df = pd.read_csv(url)
#     return df

# df = load_data()



st.set_page_config(
    page_title="Nuremberg Land Cover Explorer",
    page_icon="🔍",
    layout="wide",
)


# Bounding for Nuremberg coordinates only

NUREMBERG_LAT = 49.452
NUREMBERG_LON = 11.077

NUREMBERG_BOUNDS = {
    "min_lat": 49.38, "max_lat": 49.52,
    "min_lon": 10.95, "max_lon": 11.15,
}

ESA_CLASSES = {
    10:  {"name": "Tree cover",         "color": "#006400"},
    30:  {"name": "Grassland",          "color": "#50E80A"},
    40:  {"name": "Cropland",           "color": "#BC721D"},
    50:  {"name": "Built-up",           "color": "#000000"},
    60:  {"name": "Bare / sparse veg",  "color": "#C0EC70"},
    80:  {"name": "Permanent water",    "color": "#358FE9"},
}

QUICK_LOCATIONS = {
    "Old Town":          (49.4543, 11.0775),
    "Industrial North":  (49.490,  11.060),
    "Reichswald Forest": (49.410,  11.130),
    "Airport":           (49.497,  11.078),
}

# Half-size of the bounding box drawn around a searched location.
# 0.008 degrees ≈ roughly 600 m on each side.
BBOX_HALF = 0.008

# CSS

st.markdown("""
<style>
    .stApp { background-color: #0d1117; }

    .header-banner {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 60%, #0f3460 100%);
        border-radius: 14px;
        padding: 1.4rem 2rem;
        margin-bottom: 1.2rem;
        border: 1px solid #2d3f55;
    }
    .header-title {
        color: #e94560;
        font-size: 2rem;
        font-weight: 800;
        margin: 0 0 4px 0;
        letter-spacing: -0.5px;
    }
    .header-sub { color: #a8b2d8; font-size: 0.95rem; margin: 0; }

    .card {
        background: #161b22;
        border: 1px solid #2d3f55;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin: 0.5rem 0;
        color: #cdd6f4;
    }
    .card h4 { color: #89b4fa; margin: 0 0 6px 0; }
    .card small { color: #8b949e; }

    .legend-row {
        display: flex; align-items: center;
        gap: 8px; margin: 5px 0;
        font-size: 0.82rem; color: #cdd6f4;
    }
    .dot { width: 13px; height: 13px; border-radius: 50%; flex-shrink: 0; }

    [data-testid="metric-container"] {
        background: #161b22 !important;
        border: 1px solid #2d3f55 !important;
        border-radius: 8px;
    }
    .stRadio > label { color: #a8b2d8 !important; }

    .search-hint { font-size: 0.8rem; color: #8b949e; margin-top: 4px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────

def in_bounds(lat: float, lon: float) -> bool:
    """Returns True if the point is inside the Nuremberg study area."""
    b = NUREMBERG_BOUNDS
    return b["min_lat"] <= lat <= b["max_lat"] and b["min_lon"] <= lon <= b["max_lon"]


def try_parse_coordinates(text: str):

    # Step 1: remove degree symbols and compass directions
    cleaned = re.sub(r"[°NSEWnsew]", " ", text)
    # Step 2: replace commas with spaces
    cleaned = cleaned.replace(",", " ")
    # Step 3: extract all numbers (allowing decimals and minus for S/W)
    numbers = re.findall(r"-?\d+\.?\d*", cleaned)

    if len(numbers) == 2:
        try:
            lat = float(numbers[0])
            lon = float(numbers[1])
            # Sanity check: valid geographic ranges
            if -90 <= lat <= 90 and -180 <= lon <= 180:
                return lat, lon
        except ValueError:
            pass
    return None


def geocode_place(name: str) -> dict:
    
    full_query = f"{name}, Nuremberg, Germany"
    try:
        # First try: stay inside Nuremberg bounding box
        r = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={
                "q":               full_query,
                "format":          "json",
                "limit":           1,
                "viewbox":         "10.95,49.52,11.15,49.38",
                "bounded":         1,
                "accept-language": "en",
            },
            headers={"User-Agent": "NurembergLandCoverApp/1.0"},
            timeout=5,
        )
        results = r.json()

        # Fallback: search without bounding box
        if not results:
            r2 = requests.get(
                "https://nominatim.openstreetmap.org/search",
                params={
                    "q":               full_query,
                    "format":          "json",
                    "limit":           1,
                    "accept-language": "en",
                },
                headers={"User-Agent": "NurembergLandCoverApp/1.0"},
                timeout=5,
            )
            results = r2.json()

        if not results:
            return {
                "lat": None, "lon": None, "display": None,
                "error": f"Could not find '{name}' near Nuremberg. Try a different name.",
            }

        best = results[0]
        return {
            "lat":     float(best["lat"]),
            "lon":     float(best["lon"]),
            "display": best["display_name"],
            "error":   None,
        }

    except requests.exceptions.Timeout:
        return {"lat": None, "lon": None, "display": None,
                "error": "Search timed out – please try again."}
    except Exception as e:
        return {"lat": None, "lon": None, "display": None,
                "error": f"Search failed: {str(e)}"}


def smart_search(query: str) -> dict:
    
    query = query.strip()
    if not query:
        return {"lat": None, "lon": None, "display": None,
                "error": "Please type something.", "search_type": None}

    # Try coordinates first
    parsed = try_parse_coordinates(query)
    if parsed:
        lat, lon = parsed
        return {
            "lat":         lat,
            "lon":         lon,
            "display":     f"{lat:.4f}° N, {lon:.4f}° E",
            "error":       None,
            "search_type": "coords",
        }

    # Otherwise geocode it as a place name
    result = geocode_place(query)
    result["search_type"] = "place"
    return result


def esa_color(class_id: int) -> str:
    return ESA_CLASSES.get(class_id, {}).get("color", "#808080")


def esa_name(class_id: int) -> str:
    return ESA_CLASSES.get(class_id, {}).get("name", "Unknown")


def mock_predict(lat: float, lon: float) -> dict:
    """
    PLACEHOLDER – replace with your real model.
    Returns fake predictions based on distance from city centre.
    """
    dist = ((lat - NUREMBERG_LAT) ** 2 + (lon - NUREMBERG_LON) ** 2) ** 0.5
    if dist < 0.05:
        l2020, l2021 = 50, 50        # Built-up → Built-up
    elif dist < 0.07:
        l2020, l2021 = 40, 50        # Cropland → Built-up (CHANGE!)
    else:
        l2020, l2021 = 40, 40        # Cropland → Cropland

    rng = np.random.default_rng(seed=int(abs(lat * 1000 + lon * 100)))
    return {
        "label_2020": l2020,
        "label_2021": l2021,
        "conf_2020":  round(float(rng.uniform(0.68, 0.92)), 2),
        "conf_2021":  round(float(rng.uniform(0.70, 0.94)), 2),
        "changed":    l2020 != l2021,
    }


def make_location_bbox(lat: float, lon: float, half: float = BBOX_HALF):
    
    box_lats = [lat - half, lat + half, lat + half, lat - half, lat - half]
    box_lons = [lon - half, lon - half, lon + half, lon + half, lon - half]
    return box_lats, box_lons


def make_scatter_map(
    center_lat: float,
    center_lon: float,
    year_view: str,
    marker_lat: float = None,
    marker_lon: float = None,
    pred: dict = None,
) -> go.Figure:

    # ── Fake pixel grid (replace with real HuggingFace data) ─────────
    np.random.seed(42)
    n = 1200
    lats = np.random.uniform(NUREMBERG_BOUNDS["min_lat"],
                             NUREMBERG_BOUNDS["max_lat"], n)
    lons = np.random.uniform(NUREMBERG_BOUNDS["min_lon"],
                             NUREMBERG_BOUNDS["max_lon"], n)

    labels_2020, labels_2021 = [], []
    for la, lo in zip(lats, lons):
        d = ((la - NUREMBERG_LAT) ** 2 + (lo - NUREMBERG_LON) ** 2) ** 0.5
        if d < 0.05:
            l20, l21 = 50, 50
        elif d < 0.07:
            prob = np.random.rand()
            l20 = 40 if prob > 0.3 else 50
            l21 = 50 if prob > 0.3 else 50
        else:
            l20 = l21 = np.random.choice([10, 30, 40])
        labels_2020.append(l20)
        labels_2021.append(l21)

    labels_2020 = np.array(labels_2020)
    labels_2021 = np.array(labels_2021)

    # ── Colours depend on selected year/view ─────────────────────────
    if year_view == "2020":
        title_tag  = "2020 Land Cover"
        dot_colors = [esa_color(l) for l in labels_2020]
        dot_text   = [esa_name(l)  for l in labels_2020]
    elif year_view == "2021":
        title_tag  = "2021 Land Cover"
        dot_colors = [esa_color(l) for l in labels_2021]
        dot_text   = [esa_name(l)  for l in labels_2021]
    else:
        title_tag    = "Land Cover Change 2020 → 2021"
        changed_mask = labels_2020 != labels_2021
        dot_colors   = ["#e94560" if c else "#0b25cf" for c in changed_mask]
        dot_text     = [
            f"CHANGED: {esa_name(l20)} → {esa_name(l21)}" if c
            else f"Stable: {esa_name(l20)}"
            for c, l20, l21 in zip(changed_mask, labels_2020, labels_2021)
        ]

    fig = go.Figure()

    # Layer 1 – land cover dots
    fig.add_trace(go.Scattermapbox(
        lat=lats, lon=lons,
        mode="markers",
        marker=dict(size=6, color=dot_colors, opacity=0.75),
        text=dot_text,
        hovertemplate="<b>%{text}</b><br>Lat: %{lat:.4f}<br>Lon: %{lon:.4f}<extra></extra>",
        name=title_tag,
    ))

    # Layer 2 – red box = full Nuremberg study area boundary
    nb = NUREMBERG_BOUNDS
    fig.add_trace(go.Scattermapbox(
        lat=[nb["min_lat"], nb["max_lat"], nb["max_lat"], nb["min_lat"], nb["min_lat"]],
        lon=[nb["min_lon"], nb["min_lon"], nb["max_lon"], nb["max_lon"], nb["min_lon"]],
        mode="lines",
        line=dict(color="#e94560", width=2),
        hoverinfo="skip",
        name="Study area boundary",
    ))

    # Layer 3 – white box = bounding box around the searched location
    if marker_lat is not None:
        bbox_lats, bbox_lons = make_location_bbox(marker_lat, marker_lon)
        fig.add_trace(go.Scattermapbox(
            lat=bbox_lats,
            lon=bbox_lons,
            mode="lines",
            line=dict(color="#470808", width=2.5),
            hoverinfo="skip",
            name="Search area",
        ))

    # Layer 4 – star at the exact searched point
    if marker_lat is not None and pred is not None:
        if year_view == "Change":
            status = "⚠️ CHANGED" if pred["changed"] else " Stable"
            hover  = (f"<b>{status}</b><br>"
                      f"{esa_name(pred['label_2020'])} → {esa_name(pred['label_2021'])}")
        elif year_view == "2020":
            hover = (f"<b>📍 Your location – 2020</b><br>"
                     f"{esa_name(pred['label_2020'])}<br>"
                     f"Confidence: {pred['conf_2020']*100:.0f}%")
        else:
            hover = (f"<b>📍 Your location – 2021</b><br>"
                     f"{esa_name(pred['label_2021'])}<br>"
                     f"Confidence: {pred['conf_2021']*100:.0f}%")

        fig.add_trace(go.Scattermapbox(
            lat=[marker_lat], lon=[marker_lon],
            mode="markers",
            marker=dict(size=100, color="#000000", symbol="star"),
            text=[hover],
            hovertemplate="%{text}<extra></extra>",
            name="Your location",
        ))

    # ── Map layout ────────────────────────────────────────────────────
    fig.update_layout(
        mapbox=dict(
            style="open-street-map",
            center=dict(lat=center_lat, lon=center_lon),
            zoom=11 if marker_lat is None else 14,
        ),
        margin=dict(l=0, r=0, t=35, b=0),
        height=560,
        paper_bgcolor="#0d1117",
        font_color="#cdd6f4",
        title=dict(
            text=f"<b>{title_tag}</b>  –  Nuremberg",
            font=dict(size=15, color="#89b4fa"),
            x=0.01,
        ),
        legend=dict(
            bgcolor="#161b22", bordercolor="#2d3f55",
            borderwidth=1, font=dict(color="#cdd6f4"),
        ),
        showlegend=True,
    )
    return fig


def make_bar_chart(pred: dict) -> go.Figure:
    """Confidence bar chart for 2020 vs 2021."""
    fig = go.Figure(go.Bar(
        x=["2020", "2021"],
        y=[pred["conf_2020"] * 100, pred["conf_2021"] * 100],
        marker_color=["#4361ee", "#4cc9f0"],
        text=[f"{pred['conf_2020']*100:.0f}%", f"{pred['conf_2021']*100:.0f}%"],
        textposition="outside",
        textfont=dict(color="#cdd6f4"),
    ))
    fig.update_layout(
        title=dict(text="Model Confidence", font=dict(color="#89b4fa", size=13)),
        yaxis=dict(range=[0, 110], ticksuffix="%", gridcolor="#2d3f55", color="#8b949e"),
        xaxis=dict(color="#8b949e"),
        paper_bgcolor="#161b22", plot_bgcolor="#161b22",
        margin=dict(l=10, r=10, t=40, b=10),
        height=200, font=dict(color="#cdd6f4"),
    )
    return fig



# Store session data

for key, val in {
    "sel_lat":    NUREMBERG_LAT,
    "sel_lon":    NUREMBERG_LON,
    "has_marker": False,
    "pred":       None,
    "found_name": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = val



#header

st.markdown("""
<div class="header-banner">
    <p class="header-title">🌍 Nuremberg Land Cover Explorer</p>
    <p class="header-sub">
        Visualise urban change 2020 → 2021 &nbsp;·&nbsp;
        ESA WorldCover &nbsp;·&nbsp; Sentinel-2 &nbsp;·&nbsp; ML predictions
    </p>
</div>
""", unsafe_allow_html=True)


#Tabs

tab_search, tab_model, tab_summary = st.tabs(
    ["📍 Search for a Location", "🤖 Model Info", "📊 Data Summary"]
)

# TAB 1: Search area
with tab_search:
    st.markdown("### 📍 Search for a Location")

    search_col, btn_col = st.columns([5, 1])

    with search_col:
        query = st.text_input(
            label="search_bar",
            placeholder='Type a place  OR  coordinates — e.g. "Maxfeld"  or  "49.4415° N, 11.0797° E"',
            label_visibility="collapsed",
        )
        st.markdown(
            '<p class="search-hint">'
            "Works with: place names · coordinates with ° N/E symbols · "
            "plain numbers like <i>49.44, 11.08</i>"
            "</p>",
            unsafe_allow_html=True,
        )

    with btn_col:
        st.markdown("<br>", unsafe_allow_html=True)
        do_search = st.button("🔍 Search", use_container_width=True)

    # Quick location buttons
    st.markdown("**Quick locations →**")
    q_cols = st.columns(len(QUICK_LOCATIONS))
    for col, (label, (qlat, qlon)) in zip(q_cols, QUICK_LOCATIONS.items()):
        if col.button(label, use_container_width=True):
            st.session_state.sel_lat    = qlat
            st.session_state.sel_lon    = qlon
            st.session_state.has_marker = True
            st.session_state.pred       = mock_predict(qlat, qlon)
            st.session_state.found_name = label
            st.rerun()

    # year selecter
    st.markdown("### 🗺️ Map View")
    year_view = st.radio(
        label="Select what to display on the map:",
        options=["2020", "2021", "Change"],
        horizontal=True,
        help=(
            "2020 → land cover in 2020\n"
            "2021 → land cover in 2021\n"
            "Change → highlights pixels that changed between years"
        ),
    )

    # Process the search when button is pressed
    if do_search and query.strip():
        with st.spinner("Searching…"):
            result = smart_search(query)

        if result["error"]:
            st.error(f"❌ {result['error']}")

        elif not in_bounds(result["lat"], result["lon"]):
            st.warning(
                f"⚠️ Found **{result['display'][:70]}** "
                f"but it is outside the Nuremberg study area."
            )
        else:
            st.session_state.sel_lat    = result["lat"]
            st.session_state.sel_lon    = result["lon"]
            st.session_state.has_marker = True
            st.session_state.pred       = mock_predict(result["lat"], result["lon"])
            st.session_state.found_name = result["display"]

            if result["search_type"] == "coords":
                st.success(
                    f"Coordinates: **{result['lat']:.4f}° N, {result['lon']:.4f}° E**"
                )
            else:
                st.success(
                    f"Found: **{result['display'][:80]}**  "
                    f"({result['lat']:.4f}° N, {result['lon']:.4f}° E)"
                )

    map_col, side_col = st.columns([3, 1], gap="medium")



    with map_col:
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

    with side_col:
        # ESA colour legend
        st.markdown("#### 🎨 Land Cover Legend")
        for cls_id, info in ESA_CLASSES.items():
            st.markdown(
                f'<div class="legend-row">'
                f'<span class="dot" style="background:{info["color"]}"></span>'
                f'<span>{info["name"]}</span></div>',
                unsafe_allow_html=True,
            )



# TAB 2: MODEL INFO
with tab_model:
    if st.session_state.pred:
        pred = st.session_state.pred

        col1, col2 = st.columns(2)

        with col1:

            st.markdown("#### Model Predictions")
            if st.session_state.found_name:
                st.caption(f"📍 {st.session_state.found_name[:70]}")
            st.caption(
                f"{st.session_state.sel_lat:.4f}° N, "
                f"{st.session_state.sel_lon:.4f}° E"
            )

            # 2020 card
            st.markdown(
                f'<div class="card"><h4>2020</h4>'
                f'<span style="color:{esa_color(pred["label_2020"])};font-size:1.05rem">'
                f'■ {esa_name(pred["label_2020"])}</span><br>'
                f'Confidence: <b>{pred["conf_2020"]*100:.0f}%</b></div>',
                unsafe_allow_html=True,
            )

            # 2021 card
            st.markdown(
                f'<div class="card"><h4>2021</h4>'
                f'<span style="color:{esa_color(pred["label_2021"])};font-size:1.05rem">'
                f'■ {esa_name(pred["label_2021"])}</span><br>'
                f'Confidence: <b>{pred["conf_2021"]*100:.0f}%</b></div>',
                unsafe_allow_html=True,
            )

            # Change alert
            if pred["changed"]:
                st.error(
                    f"⚠️ **Change detected!**\n\n"
                    f"{esa_name(pred['label_2020'])} → {esa_name(pred['label_2021'])}"
                )
            else:
                st.success(f"**No change**\nStable: {esa_name(pred['label_2020'])}")

        with col2:

            st.markdown("#### Model Confidence")

            # Confidence chart
            st.plotly_chart(make_bar_chart(pred), use_container_width=True, height = 450 )

    else:
        st.info("Search for a location first to see ML predictions here.")

    st.divider()
    st.markdown("#### ⚠️ Limitations")
    st.markdown(
        '<div class="card"><small>'
        "• Features: Sentinel-2 bands B3, B4, B8, B11<br>"
        "• Labels: ESA WorldCover (10 m resolution)<br>"
        "• Cloud cover may reduce accuracy<br>"
        "• Seasonal effects not corrected<br>"
        "</small></div>",
        unsafe_allow_html=True,
    )

# TAB 3: DATA SUMMARY
with tab_summary:
    st.markdown("#### 📊 Nuremberg Land Cover Summary")

    summary_df = pd.DataFrame({
        "Land Cover":    ["Built-up", "Cropland", "Tree cover", "Grassland", "Water"],
        "Area 2020 km²": [87.3, 42.1, 31.5, 18.7, 4.2],
        "Area 2021 km²": [89.1, 39.8, 31.2, 18.9, 4.2],
        "Change km²":    [+1.8, -2.3, -0.3, +0.2, 0.0],
    })

    def colour_change_tab(val):
        if val > 0:   return "color: #4caf50; font-weight: bold"
        elif val < 0: return "color: #e94560; font-weight: bold"
        return "color: #8b949e"

    st.dataframe(
        summary_df.style.applymap(colour_change_tab, subset=["Change km²"]),
        use_container_width=True, hide_index=True,
    )

    fig_bar_tab = px.bar(
        summary_df,
        x="Land Cover",
        y=["Area 2020 km²", "Area 2021 km²"],
        barmode="group",
        color_discrete_map={"Area 2020 km²": "#4361ee", "Area 2021 km²": "#4cc9f0"},
        title="Area by Land Cover Class – 2020 vs 2021",
        labels={"value": "Area (km²)", "variable": "Year"},
    )
    fig_bar_tab.update_layout(
        paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
        font_color="#cdd6f4", title_font_color="#89b4fa",
        legend_bgcolor="#161b22",
        xaxis=dict(gridcolor="#2d3f55"),
        yaxis=dict(gridcolor="#2d3f55"),
        height=320,
    )
    st.plotly_chart(fig_bar_tab, use_container_width=True, height =500)




st.divider()



st.caption(
    "📌 Data: ESA WorldCover 2020 & 2021 · Sentinel-2 MSI · "
    "Model: Random Forest (GitHub) · Dataset: HuggingFace"
)
