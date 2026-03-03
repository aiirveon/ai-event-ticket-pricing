# ============================================================
# 04_shap_analysis.py
# AI Dynamic Ticket Pricing for UK Live Events
# Author: Ogbebor Osaheni
#
# PURPOSE: Generate SHAP explanations for our pricing model.
#
# WHAT IS SHAP?
# SHAP (SHapley Additive exPlanations) answers one question:
# "Why did the model recommend THIS price adjustment?"
#
# Without SHAP: "The model recommends +12% adjustment"
# With SHAP: "The model recommends +12% because:
#   - Artist popularity score of 9 contributed +8%
#   - It's a Saturday which contributed +2%
#   - Peak season contributed +1.5%
#   - Weather had minimal impact (-0.5%)"
#
# WHY THIS MATTERS FOR YOUR PORTFOLIO:
# As The AI Product Manager's Handbook states — explainability
# is not a nice-to-have, it's a product requirement. A venue
# manager will not accept a black-box recommendation. They
# need to understand WHY before they act on it.
#
# SHAP also directly addresses the CMA's concerns about
# dynamic pricing — every price change has a documented,
# auditable reason.
#
# PORTFOLIO SECTION THIS FEEDS:
# -> Solution (SHAP Explainability feature)
# -> Ethics (transparency principle)
# -> Results (SHAP visualisations)
# -> Implementation (Phase 2)
# ============================================================

import pandas as pd
import numpy as np
import shap
import pickle
import os
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving files
import matplotlib.pyplot as plt

print("=" * 55)
print("  Phase 2 — SHAP Explainability Analysis")
print("=" * 55)

# ============================================================
# STEP 1: LOAD MODEL AND DATA
# ============================================================

script_dir   = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

# Load the optimised model we saved in Step 3
model_path = os.path.join(project_root, "model", "ticket_pricing_model.pkl")
with open(model_path, "rb") as f:
    model = pickle.load(f)

# Load encoders
encoders_path = os.path.join(project_root, "model", "label_encoders.pkl")
with open(encoders_path, "rb") as f:
    label_encoders = pickle.load(f)

# Load feature list
features_path = os.path.join(project_root, "model", "feature_list.pkl")
with open(features_path, "rb") as f:
    FEATURES = pickle.load(f)

# Load data
data_path = os.path.join(project_root, "data", "ticket_transactions.csv")
df = pd.read_csv(data_path)

print(f"\nModel loaded: {model.__class__.__name__}")
print(f"Features: {len(FEATURES)}")
print(f"Transactions: {len(df)}")

# ============================================================
# STEP 2: PREPARE FEATURES
# Same preparation as training — encode categoricals
# ============================================================

df_model = df[FEATURES].copy()

for col in ["venue", "genre"]:
    le = label_encoders[col]
    df_model[col] = le.transform(df_model[col])

# Use a sample of 500 for SHAP — computing for all 5000
# would take too long. 500 is statistically representative.
sample_size = 500
df_sample = df_model.sample(n=sample_size, random_state=42)

print(f"\nComputing SHAP values for {sample_size} transactions...")
print("(This takes 30-60 seconds)")

# ============================================================
# STEP 3: COMPUTE SHAP VALUES
#
# TreeExplainer is optimised specifically for tree-based
# models like XGBoost. Much faster than the generic explainer.
#
# shap_values is a matrix: one row per transaction,
# one column per feature. Each value represents how much
# that feature pushed the prediction up or down from
# the average prediction.
#
# Example: shap_values[0][1] = +3.2 means for transaction 0,
# feature 1 (artist_popularity) pushed the price up by 3.2
# percentage points above the average prediction.
# ============================================================

explainer   = shap.TreeExplainer(model)
shap_values = explainer.shap_values(df_sample)

print("SHAP values computed successfully")

# ============================================================
# STEP 4: GENERATE AND SAVE VISUALISATIONS
#
# We create three charts — each tells a different story
# and each is used in different parts of your portfolio.
# ============================================================

output_dir = os.path.join(project_root, "model", "shap_plots")
os.makedirs(output_dir, exist_ok=True)

# --- CHART 1: SUMMARY BAR PLOT ---
# Shows average absolute SHAP value per feature.
# This is your "which features matter most" chart.
# Cleaner than the raw importance scores from XGBoost.
# USE IN: Portfolio Results section, LinkedIn post image

print("\nGenerating Chart 1: Feature Importance Summary...")

plt.figure(figsize=(10, 7))
shap.summary_plot(
    shap_values,
    df_sample,
    feature_names=FEATURES,
    plot_type="bar",
    show=False
)
plt.title("SHAP Feature Importance — Price Adjustment Drivers", 
          fontsize=13, fontweight='bold', pad=15)
plt.xlabel("Mean |SHAP Value| (percentage points impact)", fontsize=11)
plt.tight_layout()
chart1_path = os.path.join(output_dir, "shap_feature_importance.png")
plt.savefig(chart1_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: model/shap_plots/shap_feature_importance.png")

# --- CHART 2: SUMMARY DOT PLOT ---
# Shows not just WHICH features matter but HOW they matter.
# Red dots = high feature value. Blue dots = low feature value.
# Right side = pushed price UP. Left side = pushed price DOWN.
#
# Example: If artist_popularity shows red dots on the right,
# it means HIGH popularity scores push prices UP.
# This is the chart that proves your model learned correctly.
# USE IN: Portfolio methodology section, Loom walkthrough

print("Generating Chart 2: SHAP Dot Plot (Direction of Impact)...")

plt.figure(figsize=(10, 8))
shap.summary_plot(
    shap_values,
    df_sample,
    feature_names=FEATURES,
    show=False
)
plt.title("SHAP Values — Direction and Magnitude of Price Drivers",
          fontsize=13, fontweight='bold', pad=15)
plt.tight_layout()
chart2_path = os.path.join(output_dir, "shap_dot_plot.png")
plt.savefig(chart2_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: model/shap_plots/shap_dot_plot.png")

# --- CHART 3: SINGLE PREDICTION WATERFALL ---
# Shows exactly why the model made ONE specific recommendation.
# This is the most powerful chart for stakeholder communication.
# "Here is why we recommended +14% for THIS event tonight."
# USE IN: Streamlit demo, ethics section, LinkedIn carousel

print("Generating Chart 3: Single Prediction Waterfall...")

# Pick an interesting example — high popularity, last minute
# Find a transaction with high artist popularity and low days_to_event
df_interesting = df[
    (df["artist_popularity"] >= 8) &
    (df["days_to_event"] <= 10)
].head(1)

if len(df_interesting) > 0:
    example_idx = df_interesting.index[0]
    example_row = df_model.loc[[example_idx]]
    example_shap = explainer.shap_values(example_row)[0]
    base_value   = explainer.expected_value

    # Build waterfall data manually for clean plotting
    feature_impacts = list(zip(FEATURES, example_shap))
    feature_impacts.sort(key=lambda x: abs(x[1]), reverse=True)
    top_features = feature_impacts[:8]  # Top 8 drivers

    features_labels = [f[0] for f in top_features]
    shap_vals        = [f[1] for f in top_features]
    colors           = ['#e74c3c' if v > 0 else '#3498db' for v in shap_vals]

    plt.figure(figsize=(10, 6))
    bars = plt.barh(features_labels, shap_vals, color=colors)
    plt.axvline(x=0, color='black', linewidth=0.8)
    plt.xlabel("SHAP Value (percentage point impact on price adjustment)", fontsize=10)
    plt.title(
        f"Why this price? — Artist Popularity: {df.loc[example_idx, 'artist_popularity']}, "
        f"Days to Event: {df.loc[example_idx, 'days_to_event']}",
        fontsize=11, fontweight='bold'
    )

    # Add value labels on bars
    for bar, val in zip(bars, shap_vals):
        plt.text(
            val + (0.1 if val > 0 else -0.1),
            bar.get_y() + bar.get_height()/2,
            f'{val:+.2f}%',
            va='center',
            ha='left' if val > 0 else 'right',
            fontsize=9
        )

    plt.tight_layout()
    chart3_path = os.path.join(output_dir, "shap_waterfall_example.png")
    plt.savefig(chart3_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: model/shap_plots/shap_waterfall_example.png")

    # Print the explanation in plain English
    print(f"\n--- Plain English Explanation (for portfolio) ---")
    print(f"Event: Artist popularity {df.loc[example_idx, 'artist_popularity']}/10, "
          f"{df.loc[example_idx, 'days_to_event']} days to event")
    print(f"Venue: {df.loc[example_idx, 'venue']}")
    print(f"Genre: {df.loc[example_idx, 'genre']}")
    print(f"\nTop price drivers:")
    for feat, val in top_features[:5]:
        direction = "pushed price UP" if val > 0 else "pushed price DOWN"
        print(f"  {feat:<25} {val:+.2f}% ({direction})")

    actual_adj = df.loc[example_idx, 'price_adjustment_pct']
    base_price = df.loc[example_idx, 'base_price']
    recommended_price = base_price * (1 + actual_adj/100)
    print(f"\nBase price:          GBP {base_price:.2f}")
    print(f"Recommended adj:     {actual_adj:+.1f}%")
    print(f"Recommended price:   GBP {recommended_price:.2f}")

# ============================================================
# STEP 5: GLOBAL INSIGHTS SUMMARY
# These numbers feed your portfolio Results section directly
# ============================================================

print("\n" + "=" * 55)
print("  SHAP Global Insights")
print("=" * 55)

mean_abs_shap = np.abs(shap_values).mean(axis=0)
total_impact  = mean_abs_shap.sum()

print("\nFeature contribution to total price variation:")
for feat, val in sorted(zip(FEATURES, mean_abs_shap),
                         key=lambda x: x[1], reverse=True):
    pct_contribution = val / total_impact * 100
    print(f"  {feat:<25} {val:.3f}pp  ({pct_contribution:.1f}% of total)")

print(f"\nAverage prediction baseline: {explainer.expected_value:.2f}%")
print(f"(This is the average price adjustment across all events)")

print("\n" + "=" * 55)
print("  Phase 2 SHAP Analysis Complete")
print("=" * 55)
print("\n  3 charts saved to: model/shap_plots/")
print("  1. shap_feature_importance.png  (use in portfolio)")
print("  2. shap_dot_plot.png            (use in Loom video)")
print("  3. shap_waterfall_example.png   (use in LinkedIn)")
print("\n  Next: Build streamlit_app/app.py")
print("=" * 55)