# ============================================================
# app.py
# AI Dynamic Ticket Pricing — Interactive Dashboard
# Author: Ogbebor Osaheni
#
# PURPOSE: Production-grade Streamlit demo that lets anyone
# interact with the pricing model without touching Python.
#
# WHAT THIS DEMO DOES:
# 1. Takes event inputs from the user
# 2. Runs the trained XGBoost model
# 3. Returns a price recommendation with SHAP explanation
# 4. Shows ethics compliance status
# 5. Displays market overview charts
#
# This is the difference between a data science project
# and a shipped AI product.
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import xgboost as xgb

# ============================================================
# PAGE CONFIG — must be first Streamlit command
# ============================================================

st.set_page_config(
    page_title="AI Ticket Pricing — UK Live Events",
    page_icon="🎟️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# CUSTOM CSS — professional dark theme
# ============================================================

st.markdown("""
<style>
    /* Dark professional theme */
    .stApp {
        background-color: #0a0e1a;
        color: #e8eaf0;
    }

    /* Main header */
    .main-header {
        background: linear-gradient(135deg, #1a1f35 0%, #0d1225 100%);
        border: 1px solid #2a3050;
        border-radius: 12px;
        padding: 32px 36px;
        margin-bottom: 28px;
    }

    .main-title {
        font-family: 'Georgia', serif;
        font-size: 2.2rem;
        font-weight: 700;
        color: #ffffff;
        margin: 0 0 8px 0;
        letter-spacing: -0.5px;
    }

    .main-subtitle {
        font-size: 1.0rem;
        color: #8892b0;
        margin: 0;
        font-family: 'Courier New', monospace;
    }

    /* Metric cards */
    .metric-card {
        background: #111627;
        border: 1px solid #1e2640;
        border-radius: 10px;
        padding: 20px 24px;
        text-align: center;
        transition: border-color 0.2s;
    }

    .metric-card:hover {
        border-color: #3d5af1;
    }

    .metric-value {
        font-size: 2.0rem;
        font-weight: 700;
        color: #3d5af1;
        font-family: 'Courier New', monospace;
        display: block;
    }

    .metric-label {
        font-size: 0.78rem;
        color: #8892b0;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 4px;
        display: block;
    }

    /* Recommendation box */
    .rec-box-positive {
        background: linear-gradient(135deg, #0d2137 0%, #0a1a2e 100%);
        border: 2px solid #3d5af1;
        border-radius: 12px;
        padding: 28px 32px;
        text-align: center;
    }

    .rec-box-negative {
        background: linear-gradient(135deg, #1a0d2e 0%, #130a22 100%);
        border: 2px solid #8b5cf6;
        border-radius: 12px;
        padding: 28px 32px;
        text-align: center;
    }

    .rec-price {
        font-size: 3.2rem;
        font-weight: 700;
        color: #ffffff;
        font-family: 'Georgia', serif;
        display: block;
        line-height: 1.1;
    }

    .rec-adjustment-positive {
        font-size: 1.3rem;
        color: #4ade80;
        font-family: 'Courier New', monospace;
        font-weight: 600;
        display: block;
        margin-top: 6px;
    }

    .rec-adjustment-negative {
        font-size: 1.3rem;
        color: #f87171;
        font-family: 'Courier New', monospace;
        font-weight: 600;
        display: block;
        margin-top: 6px;
    }

    /* Ethics badge */
    .ethics-pass {
        background: #052e1c;
        border: 1px solid #16a34a;
        border-radius: 8px;
        padding: 10px 16px;
        color: #4ade80;
        font-size: 0.85rem;
        font-family: 'Courier New', monospace;
        margin: 6px 0;
        display: block;
    }

    .ethics-warn {
        background: #2d1b00;
        border: 1px solid #d97706;
        border-radius: 8px;
        padding: 10px 16px;
        color: #fbbf24;
        font-size: 0.85rem;
        font-family: 'Courier New', monospace;
        margin: 6px 0;
        display: block;
    }

    /* Section headers */
    .section-header {
        font-family: 'Georgia', serif;
        font-size: 1.3rem;
        font-weight: 700;
        color: #c8d0e8;
        border-bottom: 1px solid #1e2640;
        padding-bottom: 10px;
        margin-bottom: 20px;
    }

    /* Sidebar styling */
    .css-1d391kg, [data-testid="stSidebar"] {
        background-color: #080c18;
        border-right: 1px solid #1e2640;
    }

    /* Override Streamlit defaults */
    .stSelectbox > div > div {
        background-color: #111627;
        border-color: #2a3050;
        color: #e8eaf0;
    }

    .stSlider > div > div > div {
        background-color: #3d5af1;
    }

    div[data-testid="metric-container"] {
        background-color: #111627;
        border: 1px solid #1e2640;
        border-radius: 10px;
        padding: 16px;
    }

    /* Explanation text */
    .explain-text {
        color: #8892b0;
        font-size: 0.88rem;
        line-height: 1.6;
        font-family: 'Courier New', monospace;
        background: #080c18;
        border-left: 3px solid #3d5af1;
        padding: 12px 16px;
        border-radius: 0 8px 8px 0;
        margin: 12px 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD MODEL AND DATA
# ============================================================

@st.cache_resource
def load_model():
    """Load model and encoders once and cache them."""
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    with open(os.path.join(base, "model", "ticket_pricing_model.pkl"), "rb") as f:
        model = pickle.load(f)
    with open(os.path.join(base, "model", "label_encoders.pkl"), "rb") as f:
        encoders = pickle.load(f)
    with open(os.path.join(base, "model", "feature_list.pkl"), "rb") as f:
        features = pickle.load(f)

    df = pd.read_csv(os.path.join(base, "data", "ticket_transactions.csv"))

    return model, encoders, features, df

model, label_encoders, FEATURES, df_data = load_model()

# Venue and genre options
VENUES = [
    "O2 Arena, London",
    "Manchester Arena",
    "First Direct Arena, Leeds",
    "Utilita Arena, Birmingham",
    "SSE Hydro, Glasgow",
    "Motorpoint Arena, Cardiff",
    "Glastonbury Festival, Somerset",
    "Boardmasters, Newquay",
]

VENUE_PREMIUMS = {
    "O2 Arena, London":               0.22,
    "Manchester Arena":               0.08,
    "First Direct Arena, Leeds":      0.02,
    "Utilita Arena, Birmingham":      0.05,
    "SSE Hydro, Glasgow":             0.03,
    "Motorpoint Arena, Cardiff":      0.00,
    "Glastonbury Festival, Somerset": 0.35,
    "Boardmasters, Newquay":          0.15,
}

VENUE_WEATHER_SENSITIVE = {
    "O2 Arena, London":               False,
    "Manchester Arena":               False,
    "First Direct Arena, Leeds":      False,
    "Utilita Arena, Birmingham":      False,
    "SSE Hydro, Glasgow":             True,
    "Motorpoint Arena, Cardiff":      False,
    "Glastonbury Festival, Somerset": True,
    "Boardmasters, Newquay":          True,
}

GENRES = ["Pop", "Rock", "R&B", "Electronic/Dance",
          "Hip-Hop", "Classical", "Jazz", "Musical Theatre"]

TIER_BASE_PRICES = {
    "Standing (General)": 48.0,
    "Standard (Seated)":  78.0,
    "Premium (Seated)":   125.0,
    "VIP Package":        210.0,
}

# ============================================================
# PREDICTION FUNCTION
# ============================================================

def make_prediction(venue, genre, ticket_tier, days_to_event,
                    artist_popularity, temperature_c, is_weekend,
                    is_saturday, has_competing_event, is_peak_season,
                    viral_shock, transport_disruption, month, day_of_week):

    is_cold  = int(temperature_c < 8)
    is_rainy = int(temperature_c < 13)
    venue_location_premium = VENUE_PREMIUMS[venue]

    input_data = {
        "days_to_event":          days_to_event,
        "artist_popularity":      artist_popularity,
        "temperature_c":          temperature_c,
        "is_cold":                is_cold,
        "is_rainy":               is_rainy,
        "is_weekend":             int(is_weekend),
        "is_saturday":            int(is_saturday),
        "month":                  month,
        "day_of_week":            day_of_week,
        "has_competing_event":    int(has_competing_event),
        "is_peak_season":         int(is_peak_season),
        "viral_shock":            int(viral_shock),
        "transport_disruption":   int(transport_disruption),
        "venue_location_premium": venue_location_premium,
        "venue":                  venue,
        "genre":                  genre,
    }

    df_input = pd.DataFrame([input_data])

    # Encode categoricals
    for col in ["venue", "genre"]:
        le = label_encoders[col]
        df_input[col] = le.transform(df_input[col])

    df_input = df_input[FEATURES]

    prediction = model.predict(df_input)[0]
    prediction = np.clip(prediction, -28.0, 22.0)

    # SHAP explanation
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(df_input)[0]

    return prediction, shap_values, df_input, explainer.expected_value

# ============================================================
# SHAP WATERFALL CHART
# ============================================================

def plot_shap_waterfall(shap_values, feature_names, title="Why this price?"):
    feature_impacts = list(zip(feature_names, shap_values))
    feature_impacts.sort(key=lambda x: abs(x[1]), reverse=True)
    top = feature_impacts[:8]

    labels = [f[0].replace("_", " ").title() for f in top]
    values = [f[1] for f in top]
    colors = ['#4ade80' if v > 0 else '#f87171' for v in values]

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor('#111627')
    ax.set_facecolor('#111627')

    bars = ax.barh(labels, values, color=colors, height=0.6, edgecolor='none')
    ax.axvline(x=0, color='#3d5af1', linewidth=1.5, alpha=0.8)

    for bar, val in zip(bars, values):
        offset = 0.08 if val > 0 else -0.08
        ax.text(
            val + offset,
            bar.get_y() + bar.get_height() / 2,
            f'{val:+.2f}%',
            va='center',
            ha='left' if val > 0 else 'right',
            color='#e8eaf0',
            fontsize=9.5,
            fontfamily='monospace'
        )

    ax.set_xlabel("Price adjustment impact (percentage points)",
                  color='#8892b0', fontsize=9)
    ax.set_title(title, color='#ffffff', fontsize=11,
                 fontweight='bold', pad=12)
    ax.tick_params(colors='#8892b0', labelsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#1e2640')
    ax.spines['bottom'].set_color('#1e2640')
    ax.xaxis.label.set_color('#8892b0')

    plt.tight_layout()
    return fig

# ============================================================
# HEADER
# ============================================================

st.markdown("""
<div class="main-header">
    <p class="main-title">🎟️ AI Dynamic Ticket Pricing</p>
    <p class="main-subtitle">
        XGBoost + SHAP explainability — UK Live Events Pricing Engine
        &nbsp;|&nbsp; CMA-compliant &nbsp;|&nbsp; Built by Ogbebor Osaheni
    </p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# TOP METRICS ROW
# ============================================================

m1, m2, m3, m4 = st.columns(4)

with m1:
    st.markdown("""
    <div class="metric-card">
        <span class="metric-value">0.79</span>
        <span class="metric-label">Model R² Score</span>
    </div>
    """, unsafe_allow_html=True)

with m2:
    st.markdown("""
    <div class="metric-card">
        <span class="metric-value">3.11%</span>
        <span class="metric-label">Avg Prediction Error</span>
    </div>
    """, unsafe_allow_html=True)

with m3:
    st.markdown("""
    <div class="metric-card">
        <span class="metric-value">50</span>
        <span class="metric-label">Optuna Trials Run</span>
    </div>
    """, unsafe_allow_html=True)

with m4:
    st.markdown("""
    <div class="metric-card">
        <span class="metric-value">±22%</span>
        <span class="metric-label">CMA Price Cap</span>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ============================================================
# SIDEBAR — INPUT CONTROLS
# ============================================================

st.sidebar.markdown("## Event Configuration")
st.sidebar.markdown("---")

st.sidebar.markdown("**Venue & Genre**")
venue = st.sidebar.selectbox("Venue", VENUES, index=0)
genre = st.sidebar.selectbox("Genre", GENRES, index=0)
ticket_tier = st.sidebar.selectbox(
    "Ticket Tier", list(TIER_BASE_PRICES.keys()), index=1
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Demand Signals**")

artist_popularity = st.sidebar.slider(
    "Artist Popularity (1-10)", 1, 10, 7,
    help="1 = unknown act, 10 = global superstar"
)

days_to_event = st.sidebar.slider(
    "Days to Event", 1, 180, 21,
    help="How many days until the event"
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Date & Weather**")

month = st.sidebar.slider("Month (1=Jan, 12=Dec)", 1, 12, 7)
day_of_week = st.sidebar.slider("Day of Week (0=Mon, 6=Sun)", 0, 6, 5)
temperature_c = st.sidebar.slider("Temperature (°C)", -5, 32, 15)

is_weekend  = day_of_week >= 4
is_saturday = day_of_week == 5
is_peak_season = month in [2, 7, 8, 12]

st.sidebar.markdown("---")
st.sidebar.markdown("**Market Conditions**")

has_competing_event = st.sidebar.checkbox("Competing event same night?", value=False)
viral_shock         = st.sidebar.checkbox("Viral moment (TikTok/press)?", value=False)
transport_disruption = st.sidebar.checkbox("Transport disruption?", value=False)

# ============================================================
# MAIN CONTENT — TWO COLUMNS
# ============================================================

left_col, right_col = st.columns([1, 1], gap="large")

# --- LEFT: RECOMMENDATION ---
with left_col:
    st.markdown('<p class="section-header">Price Recommendation</p>',
                unsafe_allow_html=True)

    base_price = TIER_BASE_PRICES[ticket_tier]

    prediction, shap_vals, input_df, base_value = make_prediction(
        venue, genre, ticket_tier, days_to_event,
        artist_popularity, temperature_c, is_weekend,
        is_saturday, has_competing_event, is_peak_season,
        viral_shock, transport_disruption, month, day_of_week
    )

    recommended_price = base_price * (1 + prediction / 100)
    recommended_price = round(recommended_price * 2) / 2  # Round to 50p

    box_class   = "rec-box-positive" if prediction >= 0 else "rec-box-negative"
    adj_class   = "rec-adjustment-positive" if prediction >= 0 else "rec-adjustment-negative"
    adj_symbol  = "+" if prediction >= 0 else ""

    st.markdown(f"""
    <div class="{box_class}">
        <span style="color:#8892b0; font-size:0.8rem;
              text-transform:uppercase; letter-spacing:1px;
              font-family:monospace;">Recommended Price</span>
        <span class="rec-price">£{recommended_price:.2f}</span>
        <span class="{adj_class}">{adj_symbol}{prediction:.1f}% from base (£{base_price:.2f})</span>
        <span style="color:#8892b0; font-size:0.78rem;
              font-family:monospace; margin-top:8px; display:block;">
            {ticket_tier} &nbsp;·&nbsp; {venue.split(",")[0]}
        </span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Ethics panel
    st.markdown('<p class="section-header">Ethics & Compliance</p>',
                unsafe_allow_html=True)

    MAX_INCREASE = 22.0
    MAX_DECREASE = -28.0

    if prediction <= MAX_INCREASE:
        st.markdown(
            f'<span class="ethics-pass">✓ CMA Cap Compliant — {prediction:.1f}% is within +{MAX_INCREASE}% limit</span>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f'<span class="ethics-warn">⚠ Near CMA Cap — {prediction:.1f}% approaches +{MAX_INCREASE}% limit</span>',
            unsafe_allow_html=True
        )

    if prediction >= MAX_DECREASE:
        st.markdown(
            '<span class="ethics-pass">✓ Floor Compliant — above minimum discount threshold</span>',
            unsafe_allow_html=True
        )

    if not viral_shock:
        st.markdown(
            '<span class="ethics-pass">✓ No surge pricing detected — stable demand conditions</span>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<span class="ethics-warn">⚠ Viral demand spike detected — human review recommended</span>',
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Plain English explanation
    st.markdown('<p class="section-header">Plain English Explanation</p>',
                unsafe_allow_html=True)

    feature_impacts = list(zip(FEATURES, shap_vals))
    feature_impacts.sort(key=lambda x: abs(x[1]), reverse=True)
    top3 = feature_impacts[:3]

    explanation_parts = []
    for feat, val in top3:
        direction = "increased" if val > 0 else "decreased"
        feat_readable = feat.replace("_", " ")
        explanation_parts.append(
            f"{feat_readable} {direction} the price by {abs(val):.1f}%"
        )

    explanation = f"The model recommends £{recommended_price:.2f} " \
                  f"({adj_symbol}{prediction:.1f}% adjustment). " \
                  f"Primarily because: {explanation_parts[0]}. " \
                  f"Additionally, {explanation_parts[1]}. " \
                  f"And {explanation_parts[2]}."

    st.markdown(f'<div class="explain-text">{explanation}</div>',
                unsafe_allow_html=True)

# --- RIGHT: SHAP CHART ---
with right_col:
    st.markdown('<p class="section-header">SHAP Explanation — Why This Price?</p>',
                unsafe_allow_html=True)

    fig = plot_shap_waterfall(
        shap_vals,
        FEATURES,
        title=f"Price drivers — {ticket_tier} at {venue.split(',')[0]}"
    )
    st.pyplot(fig, use_container_width=True)
    plt.close()

    st.markdown("""
    <div class="explain-text">
        Green bars pushed the price UP.
        Red bars pushed the price DOWN.
        Bar length = magnitude of impact in percentage points.
        Every recommendation is fully auditable.
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# MARKET OVERVIEW SECTION
# ============================================================

st.markdown("---")
st.markdown('<p class="section-header">Market Overview — Price Adjustment Distribution</p>',
            unsafe_allow_html=True)

chart1, chart2 = st.columns(2)

with chart1:
    st.markdown("**Average Adjustment by Venue**")
    venue_avg = df_data.groupby("venue")["price_adjustment_pct"].mean().sort_values()
    venue_names_short = [v.split(",")[0] for v in venue_avg.index]

    fig2, ax2 = plt.subplots(figsize=(7, 4))
    fig2.patch.set_facecolor('#111627')
    ax2.set_facecolor('#111627')

    colors2 = ['#3d5af1' if v >= 0 else '#f87171' for v in venue_avg.values]
    ax2.barh(venue_names_short, venue_avg.values, color=colors2, height=0.6)
    ax2.axvline(x=0, color='#8892b0', linewidth=0.8)
    ax2.set_xlabel("Avg price adjustment (%)", color='#8892b0', fontsize=9)
    ax2.tick_params(colors='#8892b0', labelsize=8.5)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_color('#1e2640')
    ax2.spines['bottom'].set_color('#1e2640')
    plt.tight_layout()
    st.pyplot(fig2, use_container_width=True)
    plt.close()

with chart2:
    st.markdown("**Adjustment Distribution by Artist Popularity**")
    pop_avg = df_data.groupby("artist_popularity")["price_adjustment_pct"].mean()

    fig3, ax3 = plt.subplots(figsize=(7, 4))
    fig3.patch.set_facecolor('#111627')
    ax3.set_facecolor('#111627')

    ax3.plot(pop_avg.index, pop_avg.values,
             color='#3d5af1', linewidth=2.5, marker='o',
             markersize=6, markerfacecolor='#ffffff')
    ax3.fill_between(pop_avg.index, pop_avg.values,
                     alpha=0.15, color='#3d5af1')
    ax3.set_xlabel("Artist Popularity Score", color='#8892b0', fontsize=9)
    ax3.set_ylabel("Avg Adjustment (%)", color='#8892b0', fontsize=9)
    ax3.tick_params(colors='#8892b0', labelsize=8.5)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.spines['left'].set_color('#1e2640')
    ax3.spines['bottom'].set_color('#1e2640')
    plt.tight_layout()
    st.pyplot(fig3, use_container_width=True)
    plt.close()

# ============================================================
# FOOTER
# ============================================================

st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#8892b0; font-size:0.8rem;
     font-family:monospace; padding:16px 0;">
    Built by Ogbebor Osaheni &nbsp;·&nbsp;
    XGBoost + SHAP + Optuna &nbsp;·&nbsp;
    CMA-compliant dynamic pricing &nbsp;·&nbsp;
    <a href="https://github.com/aiirveon" style="color:#3d5af1;">GitHub</a>
    &nbsp;·&nbsp;
    <a href="https://www.linkedin.com/in/osaheni-o-94565421a/"
       style="color:#3d5af1;">LinkedIn</a>
</div>
""", unsafe_allow_html=True)