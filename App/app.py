"""
Deal Advisor — UK Used Car Price Predictor & Deal Rating
Loads a trained RandomForestRegressor + preprocessing artifacts and predicts
a fair market price, then rates a seller's asking price against it.

Expects these files in the same directory as this script:
    model.pkl           (RandomForestRegressor, joblib-dumped)
    scaler.pkl           (StandardScaler, fit on numerical_features)
    encoder.pkl           (OneHotEncoder, fit on categorical_features)
    app_metadata.json     (dropdown options + feature ordering)
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Deal Advisor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_DIR = Path(__file__).parent

# ─────────────────────────────────────────────────────────────────────────
# Design tokens — "Instrument Cluster"
# ─────────────────────────────────────────────────────────────────────────
BG = "#17181A"
PANEL = "#232527"
ACCENT = "#F0A93E"
BORDER = "#3A3D42"
TEXT = "#F2EFE6"
TEXT_MUTED = "#8A8F98"

RATING_COLORS = {
    "great": "#4F9D6E",
    "good": "#7FB37A",
    "fair": "#F0A93E",
    "slight": "#D97A3D",
    "over": "#B84C3E",
}

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=JetBrains+Mono:wght@400;500;700&family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif;
    color: {TEXT};
}}

.stApp {{
    background-color: {BG};
    background-image:
        radial-gradient(circle at 15% 10%, rgba(240,169,62,0.05), transparent 40%),
        radial-gradient(circle at 85% 90%, rgba(240,169,62,0.04), transparent 40%);
}}

section[data-testid="stSidebar"] {{
    background-color: {PANEL};
    border-right: 1px solid {BORDER};
}}

h1, h2, h3 {{
    font-family: 'Space Grotesk', sans-serif !important;
    color: {TEXT} !important;
    letter-spacing: -0.01em;
}}

.dash-eyebrow {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: {ACCENT};
    margin-bottom: 0.25rem;
}}

.dash-panel {{
    background-color: {PANEL};
    border: 1px solid {BORDER};
    border-radius: 10px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
}}

.odometer {{
    font-family: 'JetBrains Mono', monospace;
    font-weight: 700;
    font-variant-numeric: tabular-nums;
}}

.dash-label {{
    font-family: 'Inter', sans-serif;
    font-size: 0.82rem;
    color: {TEXT_MUTED};
    text-transform: uppercase;
    letter-spacing: 0.06em;
}}

.rating-pill {{
    display: inline-block;
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 700;
    font-size: 1.15rem;
    padding: 0.45rem 1rem;
    border-radius: 999px;
    border: 1px solid currentColor;
}}

div.stButton > button {{
    background-color: {ACCENT};
    color: {BG};
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 700;
    border: none;
    border-radius: 8px;
    padding: 0.6rem 1.2rem;
    width: 100%;
    transition: filter 0.15s ease;
}}
div.stButton > button:hover {{
    filter: brightness(1.1);
    color: {BG};
}}

/* Inputs */
.stSelectbox div[data-baseweb="select"] > div,
.stNumberInput input {{
    background-color: {BG} !important;
    border: 1px solid {BORDER} !important;
    color: {TEXT} !important;
    font-family: 'JetBrains Mono', monospace;
}}

hr {{
    border-color: {BORDER};
}}

::-webkit-scrollbar {{ width: 8px; }}
::-webkit-scrollbar-track {{ background: {BG}; }}
::-webkit-scrollbar-thumb {{ background: {BORDER}; border-radius: 4px; }}
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────
# Load artifacts (cached)
# ─────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    missing = [
        f for f in ["model.pkl", "scaler.pkl", "encoder.pkl", "app_metadata.json"]
        if not (APP_DIR / f).exists()
    ]
    if missing:
        return None, None, None, None, missing

    model = joblib.load(APP_DIR / "model.pkl")
    scaler = joblib.load(APP_DIR / "scaler.pkl")
    encoder = joblib.load(APP_DIR / "encoder.pkl")
    with open(APP_DIR / "app_metadata.json") as f:
        metadata = json.load(f)
    return model, scaler, encoder, metadata, []


model, scaler, encoder, metadata, missing_files = load_artifacts()

if missing_files:
    st.error(
        "Missing required file(s) in the app folder: "
        + ", ".join(f"`{m}`" for m in missing_files)
        + ".\n\nPlace `model.pkl`, `scaler.pkl`, `encoder.pkl`, and "
        "`app_metadata.json` in the same folder as `app.py`, then rerun."
    )
    st.stop()

NUM_FEATURES = metadata["numerical_features"]     # ['year','mileage','tax','mpg','engineSize','car_age','mileage_per_year']
CAT_FEATURES = metadata["categorical_features"]    # ['model','Make','transmission','fuelType']
REFERENCE_YEAR = metadata["reference_year"]
PREMIUM_MAKES = set(metadata["premium_makes"])


# ─────────────────────────────────────────────────────────────────────────
# Prediction pipeline — mirrors training exactly
# ─────────────────────────────────────────────────────────────────────────
def predict_price(make, model_name, transmission, fuel_type, year, mileage, tax, mpg, engine_size):
    car_age = REFERENCE_YEAR - year
    car_age_safe = car_age if car_age != 0 else 1
    mileage_per_year = mileage / car_age_safe
    is_premium = 1 if make in PREMIUM_MAKES else 0

    row_num = pd.DataFrame(
        [[year, mileage, tax, mpg, engine_size, car_age, mileage_per_year]],
        columns=NUM_FEATURES,
    )
    row_cat = pd.DataFrame(
        [[model_name, make, transmission, fuel_type]],
        columns=CAT_FEATURES,
    )

    X_num = scaler.transform(row_num)
    X_cat = encoder.transform(row_cat)
    X_final = np.hstack([X_num, X_cat, np.array([[is_premium]])])

    return float(model.predict(X_final)[0])


def deal_advisor(predicted_price, seller_price):
    diff = predicted_price - seller_price
    pct_diff = (diff / predicted_price) * 100

    if pct_diff > 10:
        key, label = "great", "Great Deal"
    elif pct_diff >= 5:
        key, label = "good", "Good Deal"
    elif pct_diff >= -5:
        key, label = "fair", "Fair Price"
    elif pct_diff >= -10:
        key, label = "slight", "Slightly Overpriced"
    else:
        key, label = "over", "Overpriced"

    return {
        "predicted_price": round(predicted_price, 2),
        "seller_price": seller_price,
        "difference": round(diff, 2),
        "pct_difference": round(pct_diff, 2),
        "rating_key": key,
        "rating_label": label,
    }


# ─────────────────────────────────────────────────────────────────────────
# Gauge — semicircular needle sweep across the 5 rating zones
# ─────────────────────────────────────────────────────────────────────────
def render_gauge(pct_diff, clamp=15):
    """SVG speedometer-style gauge. Needle sweeps left (overpriced) to right (great deal)."""
    pct_clamped = max(-clamp, min(clamp, pct_diff))
    angle = 180 * (pct_clamped + clamp) / (2 * clamp)  # 0 = far left, 180 = far right
    angle_rad = np.radians(180 - angle)  # svg: 0deg = pointing right along +x

    cx, cy, r = 200, 190, 160
    needle_len = r - 25
    nx = cx + needle_len * np.cos(angle_rad)
    ny = cy - needle_len * np.sin(angle_rad)

    zones = [
        ("over", 0, 36),
        ("slight", 36, 72),
        ("fair", 72, 108),
        ("good", 108, 144),
        ("great", 144, 180),
    ]

    def arc_path(a0, a1, radius):
        a0r, a1r = np.radians(180 - a0), np.radians(180 - a1)
        x0, y0 = cx + radius * np.cos(a0r), cy - radius * np.sin(a0r)
        x1, y1 = cx + radius * np.cos(a1r), cy - radius * np.sin(a1r)
        return f"M {x0:.1f} {y0:.1f} A {radius} {radius} 0 0 1 {x1:.1f} {y1:.1f}"

    segments = "".join(
        f'<path d="{arc_path(a0, a1, r)}" stroke="{RATING_COLORS[key]}" '
        f'stroke-width="22" fill="none" stroke-linecap="butt" opacity="0.9"/>'
        for key, a0, a1 in zones
    )

    svg = f"""
    <svg viewBox="0 0 400 230" xmlns="http://www.w3.org/2000/svg" style="width:100%; max-width:420px;">
        <defs>
            <filter id="glow" x="-50%" y="-50%" width="200%" height="200%">
                <feGaussianBlur stdDeviation="3" result="blur"/>
                <feMerge>
                    <feMergeNode in="blur"/>
                    <feMergeNode in="SourceGraphic"/>
                </feMerge>
            </filter>
        </defs>
        {segments}
        <line x1="{cx}" y1="{cy}" x2="{nx:.1f}" y2="{ny:.1f}"
              stroke="{TEXT}" stroke-width="3.5" stroke-linecap="round" filter="url(#glow)"/>
        <circle cx="{cx}" cy="{cy}" r="9" fill="{ACCENT}" stroke="{BG}" stroke-width="2"/>
        <text x="{cx}" y="{cy+38}" text-anchor="middle"
              font-family="JetBrains Mono, monospace" font-size="26" font-weight="700"
              fill="{TEXT}">{pct_diff:+.1f}%</text>
        <text x="{cx}" y="{cy+58}" text-anchor="middle"
              font-family="Inter, sans-serif" font-size="11" letter-spacing="1"
              fill="{TEXT_MUTED}">VS. PREDICTED MARKET PRICE</text>
    </svg>
    """
    return svg


# ─────────────────────────────────────────────────────────────────────────
# Sidebar — input form
# ─────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="dash-eyebrow">Vehicle Specification</div>', unsafe_allow_html=True)
    st.markdown("### Enter Car Details")

    make = st.selectbox("Make", metadata["makes"])
    models_for_make = metadata["models_by_make"].get(make, [])
    model_name = st.selectbox("Model", models_for_make)
    transmission = st.selectbox("Transmission", metadata["transmissions"])
    fuel_type = st.selectbox("Fuel Type", metadata["fuel_types"])

    st.markdown("---")

    year = st.number_input(
        "Registration Year",
        min_value=int(metadata["year_min"]),
        max_value=int(metadata["year_max"]) + 5,
        value=min(2019, int(metadata["year_max"])),
        step=1,
    )
    mileage = st.number_input(
        "Mileage",
        min_value=0,
        max_value=int(metadata["mileage_max"]) * 2,
        value=25000,
        step=1000,
    )
    tax = st.number_input("Annual Tax (£)", min_value=0, max_value=1000, value=145, step=5)
    mpg = st.number_input("MPG", min_value=0.0, max_value=470.0, value=55.0, step=0.5)
    engine_size = st.number_input(
        "Engine Size (L)",
        min_value=0.0,
        max_value=float(metadata["engine_size_max"]) + 1,
        value=1.6,
        step=0.1,
    )

    st.markdown("---")
    seller_price = st.number_input(
        "Seller's Asking Price (£)", min_value=0, max_value=200000, value=15000, step=250
    )

    run = st.button("Run Valuation")


# ─────────────────────────────────────────────────────────────────────────
# Main panel
# ─────────────────────────────────────────────────────────────────────────
st.markdown('<div class="dash-eyebrow">UK Used Car Valuation</div>', unsafe_allow_html=True)
st.markdown("# Deal Advisor")
st.markdown(
    f'<p style="color:{TEXT_MUTED}; margin-top:-0.5rem;">'
    "Random Forest market price prediction, trained on ~99k UK used car listings."
    "</p>",
    unsafe_allow_html=True,
)
st.markdown("---")

if run:
    try:
        predicted = predict_price(
            make, model_name, transmission, fuel_type, year, mileage, tax, mpg, engine_size
        )
        result = deal_advisor(predicted, seller_price)

        col1, col2 = st.columns([1, 1.1])

        with col1:
            st.markdown('<div class="dash-panel">', unsafe_allow_html=True)
            st.markdown('<div class="dash-label">Predicted Market Price</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="odometer" style="font-size:2.6rem; color:{ACCENT};">'
                f'£{result["predicted_price"]:,.0f}</div>',
                unsafe_allow_html=True,
            )
            st.markdown("<br>", unsafe_allow_html=True)

            st.markdown('<div class="dash-label">Seller Asking Price</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="odometer" style="font-size:1.6rem;">£{result["seller_price"]:,.0f}</div>',
                unsafe_allow_html=True,
            )

            st.markdown("<br>", unsafe_allow_html=True)
            color = RATING_COLORS[result["rating_key"]]
            st.markdown(
                f'<span class="rating-pill" style="color:{color};">{result["rating_label"]}</span>',
                unsafe_allow_html=True,
            )
            diff = result["difference"]
            diff_word = "below" if diff > 0 else "above"
            st.markdown(
                f'<p style="color:{TEXT_MUTED}; margin-top:0.8rem;">'
                f'Asking price is <span class="odometer" style="color:{TEXT};">£{abs(diff):,.0f}</span> '
                f'{diff_word} predicted market value '
                f'(<span class="odometer" style="color:{TEXT};">{result["pct_difference"]:+.1f}%</span>).</p>',
                unsafe_allow_html=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="dash-panel" style="text-align:center;">', unsafe_allow_html=True)
            st.markdown('<div class="dash-label">Deal Gauge</div>', unsafe_allow_html=True)
            st.markdown(render_gauge(result["pct_difference"]), unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with st.expander("Vehicle summary"):
            summary_df = pd.DataFrame(
                {
                    "Field": ["Make", "Model", "Transmission", "Fuel Type", "Year", "Mileage", "Tax", "MPG", "Engine Size"],
                    "Value": [make, model_name, transmission, fuel_type, year, f"{mileage:,}", f"£{tax}", mpg, f"{engine_size}L"],
                }
            )
            st.dataframe(summary_df, hide_index=True, use_container_width=True)

    except Exception as e:
        st.error(f"Something went wrong running the valuation: {e}")
else:
    st.markdown(
        f'<div class="dash-panel" style="text-align:center; padding:3rem;">'
        f'<div style="color:{TEXT_MUTED};">Fill in the vehicle details on the left and hit '
        f'<span style="color:{ACCENT};">Run Valuation</span> to see the predicted price and deal rating.</div>'
        f"</div>",
        unsafe_allow_html=True,
    )
