# ============================================================
# 01_data_generation.py
# AI Dynamic Ticket Pricing for UK Live Events
# Author: Ogbebor Osaheni
#
# PURPOSE: Generate a realistic synthetic dataset of UK live
# event ticket transactions with genuine market complexity.
#
# DESIGN PHILOSOPHY:
# This dataset encodes real UK live events market knowledge —
# not a simple formula. It includes demand shocks, irrational
# fan behaviour, venue-specific premiums, and genuine noise
# so the model must learn real patterns, not reverse-engineer
# a formula. Expected model R²: 0.78-0.88 (honest range).
# ============================================================

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

np.random.seed(42)

N_TRANSACTIONS = 5000

# ============================================================
# VENUE PROFILES
# Each venue has a capacity tier and a location premium.
# O2 London commands more than Cardiff — real market reality.
# Indoor vs outdoor affects weather sensitivity.
# ============================================================

VENUE_PROFILES = {
    "O2 Arena, London": {
        "location_premium": 0.22,   # London commands 22% premium
        "capacity": 20000,
        "weather_sensitive": False,  # Indoor
        "prestige": 10,
    },
    "Manchester Arena": {
        "location_premium": 0.08,
        "capacity": 21000,
        "weather_sensitive": False,
        "prestige": 8,
    },
    "First Direct Arena, Leeds": {
        "location_premium": 0.02,
        "capacity": 13500,
        "weather_sensitive": False,
        "prestige": 6,
    },
    "Utilita Arena, Birmingham": {
        "location_premium": 0.05,
        "capacity": 15800,
        "weather_sensitive": False,
        "prestige": 7,
    },
    "SSE Hydro, Glasgow": {
        "location_premium": 0.03,
        "capacity": 13000,
        "weather_sensitive": True,   # Glasgow weather is brutal
        "prestige": 7,
    },
    "Motorpoint Arena, Cardiff": {
        "location_premium": 0.00,
        "capacity": 7500,
        "weather_sensitive": False,
        "prestige": 5,
    },
    "Glastonbury Festival, Somerset": {
        "location_premium": 0.35,   # Festival premium is real
        "capacity": 135000,
        "weather_sensitive": True,   # Outdoor
        "prestige": 10,
    },
    "Boardmasters, Newquay": {
        "location_premium": 0.15,
        "capacity": 50000,
        "weather_sensitive": True,
        "prestige": 7,
    },
}

# ============================================================
# TICKET TIER PROFILES
# Base prices reflect real UK market rates (2024).
# Each tier has its own buyer psychology:
# - Standing buyers are price sensitive, buy late
# - VIP buyers are price insensitive, buy early
# ============================================================

TIER_PROFILES = {
    "Standing (General)": {
        "base_price": 48.0,
        "price_sensitivity": 0.85,  # High sensitivity — fans shop around
        "buy_early_probability": 0.25,  # Most buy late
        "weight": 0.33,
    },
    "Standard (Seated)": {
        "base_price": 78.0,
        "price_sensitivity": 0.60,
        "buy_early_probability": 0.45,
        "weight": 0.35,
    },
    "Premium (Seated)": {
        "base_price": 125.0,
        "price_sensitivity": 0.35,  # Less sensitive
        "buy_early_probability": 0.55,
        "weight": 0.20,
    },
    "VIP Package": {
        "base_price": 210.0,
        "price_sensitivity": 0.10,  # Almost insensitive
        "buy_early_probability": 0.75,  # VIP buyers plan ahead
        "weight": 0.12,
    },
}

# ============================================================
# GENRE PROFILES
# Different genres have genuinely different demand curves.
# Classical audiences are small but wealthy — high floor prices.
# Pop has mass appeal but also mass competition.
# ============================================================

GENRE_PROFILES = {
    "Pop":              {"demand_multiplier": 1.15, "volatility": 0.18},
    "Rock":             {"demand_multiplier": 1.05, "volatility": 0.14},
    "R&B":              {"demand_multiplier": 1.08, "volatility": 0.16},
    "Electronic/Dance": {"demand_multiplier": 1.10, "volatility": 0.20},
    "Hip-Hop":          {"demand_multiplier": 1.12, "volatility": 0.22},
    "Classical":        {"demand_multiplier": 0.90, "volatility": 0.08},
    "Jazz":             {"demand_multiplier": 0.82, "volatility": 0.06},
    "Musical Theatre":  {"demand_multiplier": 1.20, "volatility": 0.10},
}

print("=" * 55)
print("  Generating Realistic UK Live Events Dataset")
print("=" * 55)
print(f"\n  Target: {N_TRANSACTIONS} transactions")
print("  Encoding real UK market patterns...\n")

# ============================================================
# GENERATE BASE TRANSACTION ATTRIBUTES
# ============================================================

venue_names = list(VENUE_PROFILES.keys())
# Realistic venue weights — more London/Manchester shows
venue_weights = [0.25, 0.20, 0.10, 0.12, 0.10, 0.08, 0.10, 0.05]

tier_names = list(TIER_PROFILES.keys())
tier_weights = [p["weight"] for p in TIER_PROFILES.values()]

genre_names = list(GENRE_PROFILES.keys())
genre_weights = [0.20, 0.15, 0.12, 0.13, 0.12, 0.08, 0.06, 0.14]

# Date range: 2 years of UK events
start_date = datetime(2023, 1, 1)
end_date   = datetime(2024, 12, 31)
date_range = (end_date - start_date).days

event_dates = [
    start_date + timedelta(days=int(np.random.randint(0, date_range)))
    for _ in range(N_TRANSACTIONS)
]

venues = np.random.choice(venue_names, size=N_TRANSACTIONS, p=venue_weights)
ticket_tiers = np.random.choice(tier_names, size=N_TRANSACTIONS, p=tier_weights)
genres = np.random.choice(genre_names, size=N_TRANSACTIONS, p=genre_weights)

# ============================================================
# DAYS TO EVENT — buyer psychology baked in
# VIP buyers plan ahead. Standing buyers are last-minute.
# This creates a realistic distribution per tier.
# ============================================================

days_to_event = np.array([
    int(np.random.beta(
        2 if TIER_PROFILES[tier]["buy_early_probability"] > 0.5 else 1.2,
        2 if TIER_PROFILES[tier]["buy_early_probability"] < 0.5 else 1.2,
    ) * 179 + 1)
    for tier in ticket_tiers
])

sale_dates = [
    event_date - timedelta(days=int(d))
    for event_date, d in zip(event_dates, days_to_event)
]

# ============================================================
# TEMPORAL FEATURES
# ============================================================

day_of_week  = np.array([d.weekday() for d in event_dates])
month        = np.array([d.month for d in event_dates])
is_weekend   = (day_of_week >= 4).astype(int)  # Fri/Sat/Sun = weekend
is_friday    = (day_of_week == 4).astype(int)
is_saturday  = (day_of_week == 5).astype(int)

# Real UK peak seasons:
# July/August = festival adjacency + school holidays
# December = Christmas shows + end of year
# February = Valentine's shows
is_peak_season = np.isin(month, [2, 7, 8, 12]).astype(int)

# ============================================================
# ARTIST POPULARITY
# Real distribution: most acts are mid-tier (4-6).
# Superstar acts (9-10) are rare — maybe 3% of shows.
# Unknown acts (1-2) are common in smaller venues.
# ============================================================

artist_popularity = np.random.choice(
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    size=N_TRANSACTIONS,
    p=[0.06, 0.09, 0.11, 0.14, 0.16, 0.15, 0.13, 0.09, 0.05, 0.02]
)

# ============================================================
# WEATHER — realistic UK seasonal patterns
# Glasgow is colder and wetter than London.
# Outdoor festival venues are much more weather sensitive.
# ============================================================

seasonal_base = 10 + 8 * np.sin((month - 3) * np.pi / 6)

# Glasgow gets a cold penalty
glasgow_mask = (venues == "SSE Hydro, Glasgow").astype(float)
seasonal_base = seasonal_base - (glasgow_mask * 3)

temperature_c = seasonal_base + np.random.normal(0, 3.5, N_TRANSACTIONS)
temperature_c = np.clip(temperature_c, -5, 32)

is_cold  = (temperature_c < 8).astype(int)
is_rainy = (temperature_c < 13).astype(int)

# ============================================================
# COMPETING EVENTS
# Higher in London (more shows compete for same audience)
# Lower in Cardiff (fewer alternatives)
# ============================================================

london_mask = (venues == "O2 Arena, London").astype(float)
base_competition_prob = 0.20 + (london_mask * 0.15)
has_competing_event = np.array([
    np.random.binomial(1, p) for p in base_competition_prob
])

# ============================================================
# DEMAND SHOCKS — this is what makes data realistic
#
# Real events have unpredictable demand spikes and drops.
# These cannot be predicted from features alone — they are
# genuinely random, like real life.
# This is what forces R² down to an honest level.
# ============================================================

# Viral moment shock — artist goes viral on TikTok/radio
# Affects ~8% of events. Unpredictable. Big impact.
viral_shock = np.random.choice(
    [0, 1], size=N_TRANSACTIONS,
    p=[0.92, 0.08]
)

# Transport disruption — rail strike, road closure
# Affects ~6% of events. Suppresses demand.
transport_disruption = np.random.choice(
    [0, 1], size=N_TRANSACTIONS,
    p=[0.94, 0.06]
)

# Last minute announcement shock — supports or kills demand
# A surprise support act announced = demand spike
# A headline story about artist = could go either way
announcement_shock = np.random.normal(0, 0.04, N_TRANSACTIONS)

# ============================================================
# CALCULATE REALISTIC OPTIMAL PRICE
#
# This is intentionally more complex and less clean than
# the previous version. Real pricing has multiple interacting
# factors that don't add up neatly.
# ============================================================

# Start with base price for each tier
base_prices = np.array([TIER_PROFILES[t]["base_price"] for t in ticket_tiers])

# --- VENUE PREMIUM ---
venue_premium = np.array([
    VENUE_PROFILES[v]["location_premium"] for v in venues
])

# --- ARTIST POPULARITY EFFECT ---
# Non-linear — superstar artists have disproportionate impact
popularity_effect = np.where(artist_popularity >= 9,  0.28,
                    np.where(artist_popularity >= 7,  0.14,
                    np.where(artist_popularity >= 5,  0.05,
                    np.where(artist_popularity <= 2, -0.12,
                    -0.03))))

# --- URGENCY EFFECT ---
# Non-linear curve — drops off sharply in last 7 days
# VIP buyers don't respond to urgency the same way
tier_sensitivity = np.array([
    TIER_PROFILES[t]["price_sensitivity"] for t in ticket_tiers
])

urgency_effect = np.where(days_to_event <= 3,   0.18,
                 np.where(days_to_event <= 7,   0.12,
                 np.where(days_to_event <= 14,  0.08,
                 np.where(days_to_event <= 30,  0.04,
                 np.where(days_to_event <= 60,  0.01,
                 np.where(days_to_event >= 120, -0.06,
                 0.0)))))) * tier_sensitivity

# --- WEEKEND EFFECT ---
# Saturday > Friday > Sunday (Sunday is travel day = lower demand)
weekend_effect = (
    is_saturday * 0.07 +
    is_friday   * 0.04 +
    (day_of_week == 6).astype(int) * 0.01  # Sunday small premium
)

# --- WEATHER EFFECT ---
# Only meaningful for weather-sensitive venues
weather_sensitive = np.array([
    VENUE_PROFILES[v]["weather_sensitive"] for v in venues
]).astype(float)

weather_effect = -(is_cold & is_rainy) * 0.06 * weather_sensitive

# --- SEASONAL EFFECT ---
seasonal_effect = is_peak_season * 0.07

# --- GENRE EFFECT ---
genre_demand = np.array([
    GENRE_PROFILES[g]["demand_multiplier"] - 1.0 for g in genres
])

# --- COMPETITION EFFECT ---
competition_effect = -has_competing_event * 0.06

# --- DEMAND SHOCKS (unpredictable) ---
shock_effect = (
    viral_shock * np.random.uniform(0.10, 0.25, N_TRANSACTIONS) +
    (-transport_disruption * np.random.uniform(0.05, 0.15, N_TRANSACTIONS)) +
    announcement_shock
)

# --- COMBINE ALL EFFECTS ---
total_adjustment = (
    venue_premium +
    popularity_effect +
    urgency_effect +
    weekend_effect +
    weather_effect +
    seasonal_effect +
    genre_demand * 0.4 +
    competition_effect +
    shock_effect
)

# --- REALISTIC NOISE ---
# Human pricing decisions are never perfectly rational.
# Pricing managers make judgment calls, round numbers,
# apply gut instinct. This captures that messiness.
human_noise = np.random.normal(0, 0.035, N_TRANSACTIONS)
total_adjustment += human_noise

# --- ETHICS CAPS (CMA-compliant) ---
MAX_INCREASE = 0.22
MAX_DECREASE = 0.28
total_adjustment = np.clip(total_adjustment, -MAX_DECREASE, MAX_INCREASE)

# --- FINAL PRICE ---
optimal_price = base_prices * (1 + total_adjustment)

# Real venues round to nearest 50p or £1
optimal_price = np.round(optimal_price * 2) / 2  # Round to nearest 50p
optimal_price = np.clip(optimal_price, 15.0, 500.0)

# ============================================================
# ASSEMBLE DATAFRAME
# ============================================================

df = pd.DataFrame({
    "event_date":            event_dates,
    "sale_date":             sale_dates,
    "days_to_event":         days_to_event,
    "venue":                 venues,
    "genre":                 genres,
    "ticket_tier":           ticket_tiers,
    "base_price":            base_prices,
    "artist_popularity":     artist_popularity,
    "temperature_c":         np.round(temperature_c, 1),
    "is_cold":               is_cold,
    "is_rainy":              is_rainy,
    "is_weekend":            is_weekend,
    "is_saturday":           is_saturday,
    "month":                 month,
    "day_of_week":           day_of_week,
    "has_competing_event":   has_competing_event,
    "is_peak_season":        is_peak_season,
    "viral_shock":           viral_shock,
    "transport_disruption":  transport_disruption,
    "venue_location_premium": np.round(venue_premium, 3),
    "price_adjustment_pct":  np.round(total_adjustment * 100, 2),
    "optimal_price":         optimal_price,
})

# ============================================================
# SAVE
# ============================================================

script_dir   = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
output_path  = os.path.join(project_root, "data", "ticket_transactions.csv")

df.to_csv(output_path, index=False)

# ============================================================
# VALIDATION REPORT
# ============================================================

print(f"Dataset saved: {output_path}")
print(f"Total transactions: {len(df)}\n")

print("--- Price Distribution by Tier ---")
for tier in tier_names:
    subset = df[df["ticket_tier"] == tier]["optimal_price"]
    print(f"  {tier:<25} min=GBP{subset.min():.0f}  avg=GBP{subset.mean():.0f}  max=GBP{subset.max():.0f}")

print("\n--- Price Distribution by Venue ---")
for venue in venue_names:
    subset = df[df["venue"] == venue]["optimal_price"]
    print(f"  {venue:<35} avg=GBP{subset.mean():.0f}")

print("\n--- Demand Shocks ---")
print(f"  Viral moment events:       {viral_shock.sum()} ({viral_shock.mean()*100:.1f}%)")
print(f"  Transport disruptions:     {transport_disruption.sum()} ({transport_disruption.mean()*100:.1f}%)")

print("\n--- Price Adjustment Range ---")
print(f"  Max increase: {df['price_adjustment_pct'].max():.1f}%")
print(f"  Max decrease: {df['price_adjustment_pct'].min():.1f}%")
print(f"  Average:      {df['price_adjustment_pct'].mean():.1f}%")
print(f"  Std dev:      {df['price_adjustment_pct'].std():.1f}% (higher = more realistic noise)")

print(f"\n--- Sample Transactions ---")
print(df[["ticket_tier", "venue", "days_to_event",
          "artist_popularity", "optimal_price"]].head(8).to_string(index=False))

print(f"\n{'=' * 55}")
print(f"  Data generation complete.")
print(f"  Next: Run 02_model_training.py")
print(f"{'=' * 55}\n")