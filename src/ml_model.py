"""
ML Elasticity Model — LightGBM
================================
Predicts occupancy_rate from pricing, competitive, temporal, and
event features.  Uses SHAP to explain feature importance and
compares implied elasticities with the log-log OLS estimates.
"""

import json
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
import shap
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

FEATURE_COLS = [
    "adr",
    "adr_vs_competitor_pct",
    "competitor_avg_rate",
    "weather_index",
    "lagged_occupancy_7d",
    "direct_pct",
    "ota_booking_pct",
    "gds_pct",
    "groups_pct",
    "leisure_pct",
    "business_pct",
    "group_pct",
    "cancellation_rate_pct",
    # Encoded categoricals
    "property_tier_enc",
    "lt_band_enc",
    "dow_enc",
    "week_of_year",
    # Flags
    "local_events_flag",
    "school_holiday_flag",
    "bank_holiday_flag",
    "promo_flag",
]

TIER_MAP = {"budget": 0, "midscale": 1, "upscale": 2, "luxury": 3}
LT_MAP = {"0-7d": 0, "8-30d": 1, "31-90d": 2, "90d+": 3}
DOW_MAP = {d: i for i, d in enumerate(
    ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
)}


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create ML-ready features."""
    out = df.copy()
    out["adr_vs_competitor_pct"] = (
        (out["adr"] - out["competitor_avg_rate"]) / out["competitor_avg_rate"].clip(lower=1)
    ) * 100
    out["property_tier_enc"] = out["property_tier"].map(TIER_MAP)
    out["lt_band_enc"] = out["booking_lead_time_band"].map(LT_MAP)
    out["dow_enc"] = out["day_of_week"].map(DOW_MAP)
    out["week_of_year"] = pd.to_datetime(out["week_start_date"]).dt.isocalendar().week.astype(int)

    for col in ["local_events_flag", "school_holiday_flag", "bank_holiday_flag", "promo_flag"]:
        out[col] = out[col].astype(int)

    return out


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_model(
    df: pd.DataFrame,
    target: str = "occupancy_rate",
    n_splits: int = 4,
) -> tuple[lgb.LGBMRegressor, pd.DataFrame, dict]:
    """
    Train LightGBM with time-series cross-validation.
    Returns: fitted model, feature importances DF, metrics dict.
    """
    df = df.sort_values("week_start_date").reset_index(drop=True)
    X = df[FEATURE_COLS].copy()
    y = df[target].copy()

    tscv = TimeSeriesSplit(n_splits=n_splits)
    oof_preds = np.full(len(y), np.nan)

    params = {
        "n_estimators": 600,
        "learning_rate": 0.05,
        "max_depth": 7,
        "num_leaves": 63,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_samples": 30,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": 42,
        "verbosity": -1,
    }

    model = lgb.LGBMRegressor(**params)

    fold_metrics = []
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        preds = model.predict(X_val)
        oof_preds[val_idx] = preds

        mae = mean_absolute_error(y_val, preds)
        fold_metrics.append({"fold": fold, "mae": round(mae, 5)})
        print(f"  Fold {fold}: MAE = {mae:.5f}")

    # Final fit on all data
    model.fit(X, y)

    # OOF revenue MAPE
    valid_mask = ~np.isnan(oof_preds)
    actual_rev = df.loc[valid_mask, "rooms_revenue"]
    pred_occ = pd.Series(oof_preds[valid_mask], index=actual_rev.index)
    pred_rev = pred_occ * df.loc[valid_mask, "adr"] * df.loc[valid_mask, "rooms_available"]
    rev_mape = mean_absolute_percentage_error(actual_rev, pred_rev)

    occ_mae = mean_absolute_error(y[valid_mask], oof_preds[valid_mask])
    metrics = {
        "occupancy_mae": round(float(occ_mae), 5),
        "revenue_mape": round(float(rev_mape), 5),
        "fold_metrics": fold_metrics,
    }

    importances = pd.DataFrame({
        "feature": FEATURE_COLS,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    return model, importances, metrics


# ---------------------------------------------------------------------------
# SHAP explanation
# ---------------------------------------------------------------------------

def compute_shap(model: lgb.LGBMRegressor, df: pd.DataFrame, max_samples: int = 3000) -> shap.Explanation:
    """Compute SHAP values for the LightGBM model."""
    X = df[FEATURE_COLS].copy()
    if len(X) > max_samples:
        X = X.sample(max_samples, random_state=42)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X)
    return shap_values


def implied_elasticity_from_lgbm(
    model: lgb.LGBMRegressor,
    df: pd.DataFrame,
    delta_pct: float = 0.05,
) -> pd.DataFrame:
    """
    Compute implied own-price elasticity per property tier by perturbing ADR
    and measuring predicted occupancy change.
    """
    results = []
    for tier, grp in df.groupby("property_tier"):
        X_base = grp[FEATURE_COLS].copy()
        occ_base = model.predict(X_base).mean()

        X_up = X_base.copy()
        X_up["adr"] = X_up["adr"] * (1 + delta_pct)
        X_up["adr_vs_competitor_pct"] = (
            (X_up["adr"] - grp["competitor_avg_rate"].values) / grp["competitor_avg_rate"].values.clip(min=1)
        ) * 100
        occ_up = model.predict(X_up).mean()

        pct_occ_change = (occ_up - occ_base) / occ_base
        implied_elas = pct_occ_change / delta_pct

        results.append({
            "property_tier": tier,
            "base_occupancy": round(float(occ_base), 4),
            "perturbed_occupancy": round(float(occ_up), 4),
            "implied_elasticity": round(float(implied_elas), 4),
        })
    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(data_path: str = "data/hotel_data.csv", output_dir: str = "models") -> dict:
    """Full ML pipeline: train, explain, compare."""
    df = pd.read_csv(data_path, parse_dates=["week_start_date"])
    df = prepare_features(df)

    print("Training LightGBM occupancy model …")
    model, importances, metrics = train_model(df)

    out_dir = Path(output_dir)
    out_dir.mkdir(exist_ok=True)

    with open(out_dir / "lgbm_model.pkl", "wb") as f:
        pickle.dump(model, f)

    importances.to_csv(out_dir / "lgbm_feature_importances.csv", index=False)

    # SHAP
    print("Computing SHAP values …")
    shap_values = compute_shap(model, df)
    shap_summary = pd.DataFrame({
        "feature": FEATURE_COLS,
        "mean_abs_shap": np.abs(shap_values.values).mean(axis=0).round(5),
    }).sort_values("mean_abs_shap", ascending=False)
    shap_summary.to_csv(out_dir / "shap_summary.csv", index=False)

    # Implied elasticity comparison
    print("Computing implied elasticities …")
    impl_elas = implied_elasticity_from_lgbm(model, df)
    impl_elas.to_csv(out_dir / "lgbm_implied_elasticity.csv", index=False)

    # Save metrics
    with open(out_dir / "lgbm_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nML model artefacts saved → {out_dir}")
    print(f"Occupancy MAE: {metrics['occupancy_mae']:.5f}")
    print(f"Revenue MAPE:  {metrics['revenue_mape']:.5f}")
    print("\nTop-10 features by SHAP:")
    print(shap_summary.head(10).to_string(index=False))
    print("\nImplied elasticity per tier (LightGBM):")
    print(impl_elas.to_string(index=False))

    return {
        "model": model,
        "metrics": metrics,
        "importances": importances,
        "shap_summary": shap_summary,
        "implied_elasticity": impl_elas,
    }


def main():
    run()


if __name__ == "__main__":
    main()
