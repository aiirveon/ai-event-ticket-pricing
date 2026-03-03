# ============================================================
# 03_optuna_tuning.py
# AI Dynamic Ticket Pricing for UK Live Events
# Author: Ogbebor Osaheni
#
# PURPOSE: Use Optuna to automatically find the best
# hyperparameters for our XGBoost model.
#
# WHAT ARE HYPERPARAMETERS?
# Think of XGBoost like a chef. The recipe (algorithm) is
# fixed. But the chef has knobs to adjust:
# - How many trees to build (n_estimators)
# - How deep each tree grows (max_depth)
# - How fast it learns (learning_rate)
# - How much data each tree sees (subsample)
#
# In Phase 1 we guessed these settings. They were reasonable
# but not optimal. Optuna runs 50 experiments, trying
# different combinations, and finds the best ones.
#
# PORTFOLIO SECTION THIS FEEDS:
# -> Implementation (Phase 2 — Production ML)
# -> Results (before vs after improvement)
# -> Data & Methodology (hyperparameter optimisation)
# ============================================================

import pandas as pd
import numpy as np
import xgboost as xgb
import optuna
import pickle
import os
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

# Suppress Optuna's verbose logging — we want clean output
optuna.logging.set_verbosity(optuna.logging.WARNING)

print("=" * 55)
print("  Phase 2 — Optuna Hyperparameter Tuning")
print("=" * 55)

# ============================================================
# STEP 1: LOAD AND PREPARE DATA
# Identical to 02_model_training.py — same features, same
# target, same encoding. Consistency is critical.
# ============================================================

script_dir   = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
data_path    = os.path.join(project_root, "data", "ticket_transactions.csv")

df = pd.read_csv(data_path)
print(f"\nLoaded {len(df)} transactions")

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

df_model = df[FEATURES + [TARGET]].copy()

# Encode categoricals — same as before
label_encoders = {}
for col in ["venue", "genre"]:
    le = LabelEncoder()
    df_model[col] = le.fit_transform(df_model[col])
    label_encoders[col] = le

X = df_model[FEATURES]
y = df_model[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training set: {len(X_train)} | Test set: {len(X_test)}")

# ============================================================
# STEP 2: RECALL PHASE 1 BASELINE
# We print the baseline so we can compare improvement clearly.
# This becomes your "before vs after" story in the portfolio.
# ============================================================

print("\n--- Phase 1 Baseline (for comparison) ---")
baseline_model = xgb.XGBRegressor(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    random_state=42,
    verbosity=0
)
baseline_model.fit(X_train, y_train)
baseline_pred = baseline_model.predict(X_test)

baseline_r2   = r2_score(y_test, baseline_pred)
baseline_mae  = mean_absolute_error(y_test, baseline_pred)
baseline_rmse = np.sqrt(mean_squared_error(y_test, baseline_pred))

print(f"  R2   = {baseline_r2:.4f}")
print(f"  MAE  = {baseline_mae:.2f}%")
print(f"  RMSE = {baseline_rmse:.2f}%")

# ============================================================
# STEP 3: DEFINE THE OPTUNA OBJECTIVE FUNCTION
#
# This is the heart of Optuna. We define a function that:
# 1. Receives a "trial" object from Optuna
# 2. Uses the trial to suggest hyperparameter values
# 3. Trains a model with those values
# 4. Returns a score (we want to MAXIMISE R²)
#
# Optuna runs this function 50 times, each time trying
# different hyperparameter combinations. It uses a smart
# algorithm (TPE — Tree-structured Parzen Estimator) to
# learn which combinations tend to work better and focuses
# its search there.
#
# WHY THIS MATTERS AS A PM:
# Manual hyperparameter tuning would take hours of guessing.
# Optuna does it systematically in minutes. This is the
# difference between artisanal ML and production ML.
# ============================================================

def objective(trial):
    params = {
        # How many trees? More = better but slower
        "n_estimators": trial.suggest_int("n_estimators", 100, 600),

        # How deep can each tree grow?
        # Deeper = learns more complex patterns but risks
        # memorising training data (overfitting)
        "max_depth": trial.suggest_int("max_depth", 3, 8),

        # How fast does it learn?
        # Smaller = more careful but needs more trees
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),

        # What fraction of data does each tree see?
        # Less than 1.0 adds randomness — prevents overfitting
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),

        # What fraction of features does each tree use?
        # Forces trees to learn different patterns
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),

        # Minimum improvement needed to add a branch
        # Higher = simpler, more conservative trees
        "gamma": trial.suggest_float("gamma", 0, 0.5),

        # Regularisation — penalises model complexity
        # Prevents overfitting to training data
        "reg_alpha":  trial.suggest_float("reg_alpha", 0, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 2.0),

        "random_state": 42,
        "verbosity": 0,
    }

    model = xgb.XGBRegressor(**params)

    # Use 3-fold cross validation instead of single train/test
    # WHY? Cross validation is more reliable — it tests the
    # model on 3 different subsets and averages the results.
    # A model that scores well on all 3 folds is genuinely
    # good, not just lucky on one particular split.
    scores = cross_val_score(
        model, X_train, y_train,
        cv=3,
        scoring="r2",
        n_jobs=-1  # Use all CPU cores
    )

    return scores.mean()

# ============================================================
# STEP 4: RUN THE OPTIMISATION
#
# We create a "study" and tell it to maximise our objective.
# n_trials=50 means 50 experiments.
#
# WHY 50? Enough to explore the search space meaningfully
# without taking too long on a laptop.
# Professional projects might run 500-1000 trials overnight.
# ============================================================

print("\nRunning 50 Optuna trials...")
print("(Each trial trains and evaluates a different model)")
print("This takes 2-4 minutes — this is normal.\n")

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50, show_progress_bar=False)

# Print progress every 10 trials manually
print("Trials complete.")
print(f"Best R2 found during search: {study.best_value:.4f}")

# ============================================================
# STEP 5: TRAIN FINAL MODEL WITH BEST PARAMETERS
#
# The study found the best hyperparameters but only tested
# them with cross-validation on training data.
# Now we train a final model with those parameters on the
# full training set and evaluate on the held-out test set.
# ============================================================

print("\n--- Training final model with best parameters ---")
print("Best parameters found:")
for param, value in study.best_params.items():
    print(f"  {param}: {value}")

best_params = study.best_params
best_params["random_state"] = 42
best_params["verbosity"] = 0

final_model = xgb.XGBRegressor(**best_params)
final_model.fit(X_train, y_train)
final_pred = final_model.predict(X_test)

final_r2   = r2_score(y_test, final_pred)
final_mae  = mean_absolute_error(y_test, final_pred)
final_rmse = np.sqrt(mean_squared_error(y_test, final_pred))

# ============================================================
# STEP 6: COMPARE BASELINE VS OPTIMISED
# This is your Phase 1 vs Phase 2 story.
# These numbers go directly into your portfolio Results section.
# ============================================================

print("\n" + "=" * 55)
print("  RESULTS: Phase 1 vs Phase 2")
print("=" * 55)
print(f"  Metric     Phase 1    Phase 2    Change")
print(f"  --------   -------    -------    ------")

r2_change   = final_r2 - baseline_r2
mae_change  = final_mae - baseline_mae
rmse_change = final_rmse - baseline_rmse

print(f"  R2         {baseline_r2:.4f}     {final_r2:.4f}     {r2_change:+.4f}")
print(f"  MAE        {baseline_mae:.2f}%      {final_mae:.2f}%      {mae_change:+.2f}%")
print(f"  RMSE       {baseline_rmse:.2f}%      {final_rmse:.2f}%      {rmse_change:+.2f}%")
print("=" * 55)

if final_r2 > baseline_r2:
    improvement = (final_r2 - baseline_r2) / baseline_r2 * 100
    print(f"\nR2 improved by {improvement:.1f}% through hyperparameter tuning")
else:
    print("\nNote: Minimal improvement — baseline was already well-tuned")

# ============================================================
# STEP 7: FEATURE IMPORTANCE (OPTIMISED MODEL)
# Compare this to Phase 1 — did the same features win?
# ============================================================

print("\nFeature Importance (Optimised Model):")
importance_dict = dict(zip(FEATURES, final_model.feature_importances_))
sorted_importance = sorted(
    importance_dict.items(), key=lambda x: x[1], reverse=True
)

for feature, importance in sorted_importance:
    bar = "|" * int(importance * 50)
    print(f"  {feature:<25} {bar} {importance:.3f}")

# ============================================================
# STEP 8: SAVE OPTIMISED MODEL
# Overwrites the Phase 1 model — this is now our best model
# ============================================================

print("\nSaving optimised model...")
model_dir = os.path.join(project_root, "model")

model_path = os.path.join(model_dir, "ticket_pricing_model.pkl")
with open(model_path, "wb") as f:
    pickle.dump(final_model, f)

# Save best params for documentation
params_path = os.path.join(model_dir, "best_params.pkl")
with open(params_path, "wb") as f:
    pickle.dump(study.best_params, f)

# Save encoders and features (same as before)
encoders_path = os.path.join(model_dir, "label_encoders.pkl")
with open(encoders_path, "wb") as f:
    pickle.dump(label_encoders, f)

features_path = os.path.join(model_dir, "feature_list.pkl")
with open(features_path, "wb") as f:
    pickle.dump(FEATURES, f)

print("  Optimised model saved → model/ticket_pricing_model.pkl")
print("  Best parameters saved → model/best_params.pkl")

print("\n" + "=" * 55)
print("  Phase 2 Complete — Production Model Ready")
print("=" * 55)
print(f"  Final R2   = {final_r2:.4f}")
print(f"  Final MAE  = {final_mae:.2f}%")
print(f"  Final RMSE = {final_rmse:.2f}%")
print("\n  Next: Run 04_shap_analysis.py")
print("=" * 55)