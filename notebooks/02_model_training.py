# ============================================================
# 02_model_training.py
# AI Dynamic Ticket Pricing for UK Live Events
# Author: Ogbebor Osaheni
#
# PURPOSE: Train an XGBoost model to predict optimal ticket
# prices based on demand signals.
#
# WHAT THIS FILE DOES:
# 1. Loads the data we created in 01_data_generation.py
# 2. Prepares features (the inputs) and target (the output)
# 3. Splits data into training and testing sets
# 4. Trains an XGBoost model
# 5. Evaluates how accurate the model is
# 6. Saves the trained model to disk
#
# PORTFOLIO SECTION THIS FEEDS:
# → Implementation (Phase 1 + Phase 2)
# → Results (model accuracy metrics)
# → Data & Methodology (train/test split, validation)
# ============================================================

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import pickle
import os

# ============================================================
# STEP 1: LOAD THE DATA
# ============================================================

print("=" * 55)
print("  AI Dynamic Ticket Pricing — Model Training")
print("=" * 55)

# Build path to data file
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
data_path = os.path.join(project_root, "data", "ticket_transactions.csv")

print(f"\n📂 Loading data from: {data_path}")
df = pd.read_csv(data_path)
print(f"✅ Loaded {len(df)} transactions with {len(df.columns)} columns")

# ============================================================
# STEP 2: FEATURE SELECTION
#
# PM DECISION: Which columns do we feed the model?
#
# We have 18 columns but NOT all of them should be features.
# Rules for exclusion:
# - event_date, sale_date: Raw dates aren't useful as numbers.
#   We already extracted month, day_of_week from them.
# - base_price: This is the starting price BEFORE adjustment.
#   If we include it, the model just learns "base + adjustment =
#   optimal" which is trivial and teaches it nothing useful.
# - price_adjustment_pct: This is a direct intermediate step
#   TO our target. Including it would be "cheating" — the model
#   would just learn to reverse-engineer it.
# - optimal_price: This IS our target. Never include the target
#   as a feature — the model would just memorise it perfectly
#   (called "data leakage" — a critical ML mistake).
#
# What we KEEP are the real-world signals the model should
# learn to respond to.
# ============================================================

FEATURES = [
    "days_to_event",
    "artist_popularity",
    "temperature_c",
    "is_cold",
    "is_rainy",
    "is_weekend",
    "is_saturday",
    "month",
    "day_of_week",
    "has_competing_event",
    "is_peak_season",
    "viral_shock",
    "transport_disruption",
    "venue_location_premium",
    "venue",
    "genre",
]

TARGET = "price_adjustment_pct"

print(f"\n🎯 Target variable: {TARGET}")
print(f"📊 Features selected: {len(FEATURES)}")
print(f"   Numeric: days_to_event, artist_popularity, temperature_c, etc.")
print(f"   Categorical: ticket_tier, venue, genre (will be encoded)")

# ============================================================
# STEP 3: ENCODE CATEGORICAL FEATURES
#
# WHY? XGBoost only understands numbers, not text.
# "Standing (General)" needs to become 0, 1, 2, 3 etc.
# LabelEncoder does this automatically.
#
# We save the encoders to disk so the Streamlit app can use
# the SAME encoding when a user makes a prediction.
# If we don't save them, the app would encode differently
# and the model would give wrong predictions.
# ============================================================

print("\n🔄 Encoding categorical features...")

df_model = df[FEATURES + [TARGET]].copy()

label_encoders = {}
categorical_cols = ["venue", "genre"]

for col in categorical_cols:
    le = LabelEncoder()
    df_model[col] = le.fit_transform(df_model[col])
    label_encoders[col] = le
    print(f"   {col}: {list(le.classes_)} → {list(range(len(le.classes_)))}")

# ============================================================
# STEP 4: TRAIN / TEST SPLIT
#
# WHY? We need to test our model on data it has NEVER seen.
# If we test on the same data we trained on, the model will
# look artificially perfect — it just memorised the answers.
#
# 80% of data → training (model learns from this)
# 20% of data → testing (we evaluate on this)
#
# random_state=42 ensures the same split every time you run.
# This is reproducibility — same result, every machine.
#
# PM INSIGHT: This is like testing a new pricing strategy
# on a sample of stores before rolling it out everywhere.
# ============================================================

X = df_model[FEATURES]
y = df_model[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,      # 20% held out for testing
    random_state=42     # Reproducible split
)

print(f"\n✂️  Train/Test Split:")
print(f"   Training set:  {len(X_train)} transactions (80%)")
print(f"   Testing set:   {len(X_test)} transactions (20%)")

# ============================================================
# STEP 5: TRAIN THE XGBOOST MODEL
#
# WHY XGBOOST?
# XGBoost (Extreme Gradient Boosting) is consistently the
# best-performing algorithm on tabular (table-shaped) data.
# It builds many small decision trees, each one correcting
# the mistakes of the previous one. This "boosting" approach
# makes it extremely accurate.
#
# As Leong Chan notes in Applied AI — the right algorithm
# choice depends on your data type. For structured tabular
# data like ours, tree-based models like XGBoost outperform
# neural networks because they handle mixed data types
# (numbers + categories) naturally.
#
# HYPERPARAMETERS EXPLAINED:
# - n_estimators: How many trees to build (100 = 100 trees)
# - max_depth: How deep each tree can grow (6 = medium)
# - learning_rate: How much each tree corrects the last (0.1)
# - subsample: Use 80% of data per tree (prevents overfitting)
# - random_state: Reproducibility
#
# These are starting values. We'll optimise them with Optuna
# in the next script.
# ============================================================

print("\n🤖 Training XGBoost model...")
print("   (This is your Phase 1 baseline model)")

model = xgb.XGBRegressor(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    random_state=42,
    verbosity=0         # Suppress XGBoost's own output
)

model.fit(X_train, y_train)
print("✅ Model trained successfully")

# ============================================================
# STEP 6: EVALUATE THE MODEL
#
# THREE METRICS — what they mean in plain English:
#
# R² (R-squared): "How much of the price variation does our
# model explain?" 1.0 = perfect. 0.0 = useless.
# Target: > 0.95
#
# MAE (Mean Absolute Error): "On average, how many pounds
# off is each prediction?" Lower = better.
# Target: < £5.00 (less than 5% error on a £100 ticket)
#
# RMSE (Root Mean Squared Error): Like MAE but penalises
# large errors more heavily. A prediction that's £20 off
# hurts your RMSE score more than two predictions £10 off.
# Target: < £8.00
#
# PM INSIGHT: These metrics feed directly into your portfolio
# Results section and give you credible numbers to quote in
# interviews. "My model achieves R² = X" is a concrete
# achievement, not a vague claim.
# ============================================================

print("\n📊 Evaluating model performance...")

y_pred = model.predict(X_test)

r2   = r2_score(y_test, y_pred)
mae  = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"\n  --- BASELINE MODEL RESULTS (Phase 1) ---")
print(f"  R2   = {r2:.4f}  (target: > 0.95)")
print(f"  MAE  = GBP {mae:.2f}   (avg prediction error)")
print(f"  RMSE = GBP {rmse:.2f}   (penalised error)")
print(f"  -----------------------------------------")

if r2 > 0.95:
    print(f"\n✅ R² target met! Model explains {r2*100:.1f}% of price variation.")
else:
    print(f"\n⚠️  R² below target. We'll improve this with Optuna tuning.")

# ============================================================
# STEP 7: FEATURE IMPORTANCE
#
# This tells us which features the model found most useful.
# XGBoost gives each feature an "importance score" based on
# how often it used that feature to make decisions.
#
# WHY THIS MATTERS FOR YOUR PORTFOLIO:
# Feature importance validates your PM decisions.
# If "days_to_event" has high importance, it confirms your
# hypothesis that urgency is a key pricing signal.
# This is the bridge between your data science work and
# your product thinking.
# ============================================================

print("\n🔍 Feature Importance (which signals matter most):")

importance_dict = dict(zip(FEATURES, model.feature_importances_))
sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)

for feature, importance in sorted_importance:
    bar = "|" * int(importance * 50)
    print(f"   {feature:<25} {bar} {importance:.3f}")

# ============================================================
# STEP 8: SAMPLE PREDICTIONS
#
# Let's see actual examples — what does the model predict
# vs what the real price should be?
# This makes the model tangible and interview-ready.
# ============================================================

print("\n🎟️  Sample Predictions (first 5 test transactions):")
print(f"   {'Actual Price':>14} | {'Predicted Price':>15} | {'Difference':>12}")
print(f"   {'-'*14}-+-{'-'*15}-+-{'-'*12}")

for actual, predicted in zip(list(y_test[:5]), list(y_pred[:5])):
    diff = predicted - actual
    print(f"   £{actual:>12.2f} | £{predicted:>13.2f} | £{diff:>+11.2f}")

# ============================================================
# STEP 9: SAVE EVERYTHING
#
# We save three things:
# 1. The trained model → used by Streamlit app for predictions
# 2. The label encoders → so app uses same text→number mapping
# 3. The feature list → so app sends features in correct order
#
# WHY PICKLE? Pickle is Python's built-in way to save objects.
# It "freezes" the model exactly as it is so we can "thaw"
# it later in the Streamlit app without retraining.
# ============================================================

print("\n💾 Saving model and encoders...")

model_dir = os.path.join(project_root, "model")

# Save model
model_path = os.path.join(model_dir, "ticket_pricing_model.pkl")
with open(model_path, "wb") as f:
    pickle.dump(model, f)
print(f"   ✅ Model saved → model/ticket_pricing_model.pkl")

# Save label encoders
encoders_path = os.path.join(model_dir, "label_encoders.pkl")
with open(encoders_path, "wb") as f:
    pickle.dump(label_encoders, f)
print(f"   ✅ Encoders saved → model/label_encoders.pkl")

# Save feature list
features_path = os.path.join(model_dir, "feature_list.pkl")
with open(features_path, "wb") as f:
    pickle.dump(FEATURES, f)
print(f"   ✅ Feature list saved → model/feature_list.pkl")

print(f"\n{'=' * 55}")
print(f"  Phase 1 Complete — Baseline Model Trained")
print(f"{'=' * 55}")
print(f"\n  R²   = {r2:.4f}")
print(f"  MAE  = £{mae:.2f}")
print(f"  RMSE = £{rmse:.2f}")
print(f"\n  Next: Run 03_optuna_tuning.py to optimise the model")
print(f"{'=' * 55}\n")