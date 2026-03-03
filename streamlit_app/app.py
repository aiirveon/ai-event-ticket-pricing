# ============================================================
# app.py
# AI Dynamic Ticket Pricing — Interactive Dashboard
# Author: Ogbebor Osaheni
# Design: Premium fintech — deep navy + gold/amber accents
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
import matplotlib.font_manager as fm

st.set_page_config(
    page_title="AI Ticket Pricing — Ogbebor Osaheni",
    page_icon="🎟",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# DESIGN SYSTEM — Navy + Gold
# ============================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=DM+Sans:wght@300;400;500&family=DM+Mono:wght@400;500&display=swap');

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background-color: #06091a;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background-color: #080c1f;
    border-right: 1px solid #1a2040;
}

[data-testid="stSidebar"] .block-container {
    padding-top: 2rem;
}

/* ── Sidebar labels ── */
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stMarkdown p {
    color: #8892b8 !important;
    font-size: 0.78rem !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-family: 'DM Mono', monospace !important;
}

/* ── Sidebar section titles ── */
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #c9a84c !important;
    font-family: 'Playfair Display', serif !important;
    font-size: 1.0rem !important;
    border-bottom: 1px solid #1a2040;
    padding-bottom: 8px;
    margin-bottom: 16px;
}

/* ── Sliders ── */
[data-testid="stSlider"] > div > div > div > div {
    background: #c9a84c !important;
}

/* ── Selectbox ── */
[data-testid="stSelectbox"] > div > div {
    background-color: #0d1230 !important;
    border: 1px solid #1a2040 !important;
    color: #e8eaf0 !important;
    border-radius: 8px !important;
}

/* ── Checkboxes ── */
[data-testid="stCheckbox"] label {
    color: #8892b8 !important;
    font-size: 0.82rem !important;
}

/* ── Divider ── */
hr {
    border-color: #1a2040 !important;
    margin: 1.5rem 0 !important;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding-top: 1.5rem !important;
    padding-bottom: 2rem !important;
    max-width: 1400px;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
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
    "O2 Arena, London": 0.22,
    "Manchester Arena": 0.08,
    "First Direct Arena, Leeds": 0.02,
    "Utilita Arena, Birmingham": 0.05,
    "SSE Hydro, Glasgow": 0.03,
    "Motorpoint Arena, Cardiff": 0.00,
    "Glastonbury Festival, Somerset": 0.35,
    "Boardmasters, Newquay": 0.15,
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
# PREDICTION
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
    for col in ["venue", "genre"]:
        df_input[col] = label_encoders[col].transform(df_input[col])
    df_input = df_input[FEATURES]

    prediction   = float(model.predict(df_input)[0])
    prediction   = np.clip(prediction, -28.0, 22.0)
    explainer    = shap.TreeExplainer(model)
    shap_values  = explainer.shap_values(df_input)[0]

    return prediction, shap_values, df_input, explainer.expected_value

# ============================================================
# SHAP CHART — navy/gold palette
# ============================================================

def plot_shap(shap_values, feature_names, title=""):
    impacts = sorted(zip(feature_names, shap_values),
                     key=lambda x: abs(x[1]), reverse=True)[:8]
    labels = [f[0].replace("_", " ").title() for f in impacts]
    values = [f[1] for f in impacts]
    colors = ['#c9a84c' if v > 0 else '#e05c5c' for v in values]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    fig.patch.set_facecolor('#0d1230')
    ax.set_facecolor('#0d1230')

    bars = ax.barh(labels, values, color=colors,
                   height=0.55, edgecolor='none')
    ax.axvline(x=0, color='#2a3060', linewidth=1.2)

    for bar, val in zip(bars, values):
        offset = 0.06 if val > 0 else -0.06
        ax.text(
            val + offset,
            bar.get_y() + bar.get_height() / 2,
            f'{val:+.2f}%',
            va='center',
            ha='left' if val > 0 else 'right',
            color='#e8eaf0', fontsize=9,
        )

    ax.set_xlabel("Price adjustment impact (pp)",
                  color='#5a6080', fontsize=8.5)
    ax.set_title(title, color='#c9a84c',
                 fontsize=10, fontweight='bold', pad=10)
    ax.tick_params(colors='#8892b8', labelsize=8.5)
    for spine in ax.spines.values():
        spine.set_color('#1a2040')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    return fig

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown("## Configure Event")
    st.markdown("---")

    st.markdown("### Venue & Product")
    venue       = st.selectbox("Venue", VENUES)
    genre       = st.selectbox("Genre", GENRES)
    ticket_tier = st.selectbox("Ticket Tier",
                               list(TIER_BASE_PRICES.keys()), index=1)

    st.markdown("---")
    st.markdown("### Demand Signals")
    artist_popularity = st.slider("Artist Popularity", 1, 10, 7)
    days_to_event     = st.slider("Days to Event", 1, 180, 21)

    st.markdown("---")
    st.markdown("### Date & Weather")
    month        = st.slider("Month", 1, 12, 7)
    day_of_week  = st.slider("Day of Week  (0=Mon · 6=Sun)", 0, 6, 5)
    temperature_c = st.slider("Temperature (°C)", -5, 32, 15)

    is_weekend     = day_of_week >= 4
    is_saturday    = day_of_week == 5
    is_peak_season = month in [2, 7, 8, 12]

    st.markdown("---")
    st.markdown("### Market Conditions")
    has_competing_event  = st.checkbox("Competing event tonight?")
    viral_shock          = st.checkbox("Viral moment active?")
    transport_disruption = st.checkbox("Transport disruption?")

    st.markdown("---")
    st.markdown("""
    <p style='font-family:DM Mono,monospace;font-size:0.7rem;
    color:#3a4060;line-height:1.6;'>
    XGBoost · SHAP · Optuna<br>
    R² = 0.79 · MAE = 3.11%<br>
    CMA-compliant · ±22% cap<br><br>
    <a href='https://github.com/aiirveon/ai-event-ticket-pricing'
    style='color:#c9a84c;'>GitHub →</a>
    </p>
    """, unsafe_allow_html=True)

# ============================================================
# RUN PREDICTION
# ============================================================

prediction, shap_vals, input_df, base_value = make_prediction(
    venue, genre, ticket_tier, days_to_event,
    artist_popularity, temperature_c, is_weekend,
    is_saturday, has_competing_event, is_peak_season,
    viral_shock, transport_disruption, month, day_of_week
)

base_price        = TIER_BASE_PRICES[ticket_tier]
recommended_price = round(base_price * (1 + prediction / 100) * 2) / 2
adj_sign          = "+" if prediction >= 0 else ""
adj_color         = "#c9a84c" if prediction >= 0 else "#e05c5c"

# ============================================================
# HEADER
# ============================================================

# Hero price card using Streamlit native components
    st.markdown(f"""
    <div style='background:linear-gradient(135deg,#0d1230 0%,#080c1f 100%);
    border:1px solid #c9a84c;border-radius:14px;
    padding:36px 32px;text-align:center;margin-bottom:20px;'>
    <p style='font-family:DM Mono,monospace;font-size:0.68rem;
    color:#5a6080;text-transform:uppercase;letter-spacing:0.15em;
    margin:0 0 12px 0;'>Recommended Price</p>
    <p style='font-family:Playfair Display,serif;font-size:3.6rem;
    font-weight:700;color:#ffffff;margin:0;line-height:1;'>£{recommended_price:.2f}</p>
    <p style='font-family:DM Mono,monospace;font-size:1.1rem;
    color:{adj_color};margin:10px 0 0 0;font-weight:500;'>{adj_sign}{prediction:.1f}% from base price of £{base_price:.2f}</p>
    <div style='height:1px;background:#1a2040;margin:16px 0;'></div>
    <p style='font-family:DM Mono,monospace;font-size:0.72rem;
    color:#3a4060;margin:0;'>{ticket_tier} &nbsp;·&nbsp; {venue.split(",")[0]} &nbsp;·&nbsp; {genre}</p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# METRICS ROW
# ============================================================

c1, c2, c3, c4 = st.columns(4)

def metric_card(col, value, label, sublabel=""):
    col.markdown(f"""
    <div style='background:#0d1230;border:1px solid #1a2040;
    border-radius:10px;padding:18px 20px;'>
        <p style='font-family:DM Mono,monospace;font-size:0.68rem;
        color:#5a6080;text-transform:uppercase;letter-spacing:0.1em;
        margin:0 0 6px 0;'>{label}</p>
        <p style='font-family:Playfair Display,serif;font-size:1.8rem;
        font-weight:700;color:#c9a84c;margin:0;line-height:1;'>{value}</p>
        <p style='font-family:DM Mono,monospace;font-size:0.68rem;
        color:#3a4060;margin:4px 0 0 0;'>{sublabel}</p>
    </div>
    """, unsafe_allow_html=True)

metric_card(c1, "0.79", "Model R²", "Price adj. accuracy")
metric_card(c2, "3.11%", "Mean Abs Error", "Avg prediction error")
metric_card(c3, "50", "Optuna Trials", "Hyperparameter search")
metric_card(c4, "±22%", "CMA Cap", "Ethical price boundary")

st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)

# ============================================================
# MAIN — TWO COLUMNS
# ============================================================

left, right = st.columns([1, 1], gap="large")

# ── LEFT ──
with left:

    # Hero price card
    st.markdown(f"""
    <div style='background:linear-gradient(135deg,#0d1230 0%,#080c1f 100%);
    border:1px solid #c9a84c;border-radius:14px;
    padding:36px 32px;text-align:center;margin-bottom:20px;'>

        <p style='font-family:DM Mono,monospace;font-size:0.68rem;
        color:#5a6080;text-transform:uppercase;letter-spacing:0.15em;
        margin:0 0 12px 0;'>Recommended Price</p>

        <p style='font-family:Playfair Display,serif;font-size:3.6rem;
        font-weight:700;color:#ffffff;margin:0;line-height:1;'>
        £{recommended_price:.2f}</p>

        <p style='font-family:DM Mono,monospace;font-size:1.1rem;
        color:{adj_color};margin:10px 0 0 0;font-weight:500;'>
        {adj_sign}{prediction:.1f}% from base price of £{base_price:.2f}</p>

        <div style='height:1px;background:#1a2040;margin:16px 0;'></div>

        <p style='font-family:DM Mono,monospace;font-size:0.72rem;
        color:#3a4060;margin:0;'>
        {ticket_tier} &nbsp;·&nbsp; {venue.split(",")[0]}
        &nbsp;·&nbsp; {genre}
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Ethics panel
    st.markdown("""
    <p style='font-family:DM Mono,monospace;font-size:0.72rem;
    color:#c9a84c;text-transform:uppercase;letter-spacing:0.12em;
    margin:0 0 12px 0;'>Ethics & Compliance</p>
    """, unsafe_allow_html=True)

    def ethics_row(passed, text):
        icon  = "✓" if passed else "⚠"
        bg    = "#05180f" if passed else "#1f1000"
        border= "#1a5c30" if passed else "#7a4500"
        color = "#4ade80" if passed else "#fbbf24"
        st.markdown(f"""
        <div style='background:{bg};border:1px solid {border};
        border-radius:8px;padding:10px 14px;margin-bottom:8px;
        font-family:DM Mono,monospace;font-size:0.78rem;color:{color};'>
        {icon} &nbsp; {text}
        </div>
        """, unsafe_allow_html=True)

    ethics_row(prediction <= 22.0,
               f"CMA Cap — {prediction:.1f}% within +22% limit")
    ethics_row(prediction >= -28.0,
               "Floor compliant — above minimum threshold")
    ethics_row(not viral_shock,
               "No surge pricing — stable demand conditions"
               if not viral_shock else
               "Viral demand spike — human review recommended")

    # Plain English
    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
    st.markdown("""
    <p style='font-family:DM Mono,monospace;font-size:0.72rem;
    color:#c9a84c;text-transform:uppercase;letter-spacing:0.12em;
    margin:0 0 12px 0;'>Plain English Explanation</p>
    """, unsafe_allow_html=True)

    impacts = sorted(zip(FEATURES, shap_vals),
                     key=lambda x: abs(x[1]), reverse=True)[:3]
    parts = []
    for feat, val in impacts:
        direction = "increased" if val > 0 else "decreased"
        parts.append(
            f"{feat.replace('_',' ')} {direction} "
            f"the price by {abs(val):.1f}pp"
        )

    explanation = (
        f"Recommending £{recommended_price:.2f} "
        f"({adj_sign}{prediction:.1f}% adjustment). "
        f"{parts[0].capitalize()}. "
        f"{parts[1].capitalize()}. "
        f"{parts[2].capitalize()}."
    )

    st.markdown(f"""
    <div style='background:#080c1f;border-left:3px solid #c9a84c;
    border-radius:0 8px 8px 0;padding:14px 18px;
    font-family:DM Mono,monospace;font-size:0.82rem;
    color:#8892b8;line-height:1.7;'>
    {explanation}
    </div>
    """, unsafe_allow_html=True)

# ── RIGHT ──
with right:
    st.markdown("""
    <p style='font-family:DM Mono,monospace;font-size:0.72rem;
    color:#c9a84c;text-transform:uppercase;letter-spacing:0.12em;
    margin:0 0 12px 0;'>SHAP Explanation — Why This Price?</p>
    """, unsafe_allow_html=True)

    fig = plot_shap(
        shap_vals, FEATURES,
        title=f"Price drivers — {ticket_tier} · {venue.split(',')[0]}"
    )
    st.pyplot(fig, use_container_width=True)
    plt.close()

    st.markdown("""
    <div style='background:#080c1f;border-left:3px solid #1a2040;
    border-radius:0 8px 8px 0;padding:12px 16px;margin-top:12px;
    font-family:DM Mono,monospace;font-size:0.76rem;
    color:#3a4060;line-height:1.7;'>
    Gold bars pushed the price UP &nbsp;·&nbsp;
    Red bars pushed the price DOWN<br>
    Bar length = magnitude in percentage points &nbsp;·&nbsp;
    Every recommendation is fully auditable
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# MARKET OVERVIEW
# ============================================================

st.markdown("""
<div style='height:1px;background:linear-gradient(
90deg,#c9a84c 0%,#1a2040 60%);margin:36px 0 28px 0;'></div>
<p style='font-family:Playfair Display,serif;font-size:1.3rem;
font-weight:600;color:#ffffff;margin:0 0 20px 0;'>
Market Overview
</p>
""", unsafe_allow_html=True)

mc1, mc2 = st.columns(2)

def market_fig():
    fig, ax = plt.subplots(figsize=(7, 3.8))
    fig.patch.set_facecolor('#0d1230')
    ax.set_facecolor('#0d1230')
    for spine in ax.spines.values():
        spine.set_color('#1a2040')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(colors='#8892b8', labelsize=8)
    return fig, ax

with mc1:
    st.markdown("""
    <p style='font-family:DM Mono,monospace;font-size:0.72rem;
    color:#5a6080;text-transform:uppercase;letter-spacing:0.1em;
    margin:0 0 10px 0;'>Avg Adjustment by Venue</p>
    """, unsafe_allow_html=True)

    venue_avg = (df_data.groupby("venue")["price_adjustment_pct"]
                 .mean().sort_values())
    short     = [v.split(",")[0] for v in venue_avg.index]

    fig2, ax2 = market_fig()
    colors2   = ['#c9a84c' if v >= 15 else '#3d5af1'
                 for v in venue_avg.values]
    ax2.barh(short, venue_avg.values, color=colors2, height=0.55)
    ax2.axvline(x=venue_avg.mean(), color='#c9a84c',
                linewidth=1, linestyle='--', alpha=0.5)
    ax2.set_xlabel("Avg price adjustment (%)",
                   color='#5a6080', fontsize=8)
    plt.tight_layout()
    st.pyplot(fig2, use_container_width=True)
    plt.close()

with mc2:
    st.markdown("""
    <p style='font-family:DM Mono,monospace;font-size:0.72rem;
    color:#5a6080;text-transform:uppercase;letter-spacing:0.1em;
    margin:0 0 10px 0;'>Price Adjustment vs Artist Popularity</p>
    """, unsafe_allow_html=True)

    pop_avg = (df_data.groupby("artist_popularity")
               ["price_adjustment_pct"].mean())

    fig3, ax3 = market_fig()
    ax3.plot(pop_avg.index, pop_avg.values,
             color='#c9a84c', linewidth=2.5,
             marker='o', markersize=5,
             markerfacecolor='#06091a',
             markeredgecolor='#c9a84c',
             markeredgewidth=2)
    ax3.fill_between(pop_avg.index, pop_avg.values,
                     alpha=0.08, color='#c9a84c')
    ax3.set_xlabel("Artist Popularity (1–10)",
                   color='#5a6080', fontsize=8)
    ax3.set_ylabel("Avg Adjustment (%)",
                   color='#5a6080', fontsize=8)
    plt.tight_layout()
    st.pyplot(fig3, use_container_width=True)
    plt.close()

# ============================================================
# FOOTER
# ============================================================

st.markdown("""
<div style='height:1px;background:#1a2040;margin:40px 0 20px 0;'></div>
<div style='display:flex;justify-content:space-between;
align-items:center;padding:0 0 32px 0;'>
    <p style='font-family:DM Mono,monospace;font-size:0.72rem;
    color:#3a4060;margin:0;'>
    Built by Ogbebor Osaheni &nbsp;·&nbsp;
    XGBoost + SHAP + Optuna &nbsp;·&nbsp;
    CMA-compliant dynamic pricing
    </p>
    <p style='font-family:DM Mono,monospace;font-size:0.72rem;margin:0;'>
    <a href='https://github.com/aiirveon/ai-event-ticket-pricing'
    style='color:#c9a84c;text-decoration:none;'>GitHub</a>
    &nbsp;&nbsp;
    <a href='https://www.linkedin.com/in/osaheni-o-94565421a/'
    style='color:#c9a84c;text-decoration:none;'>LinkedIn</a>
    </p>
</div>
""", unsafe_allow_html=True)