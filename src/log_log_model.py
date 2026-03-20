"""
Log-Log Price Elasticity Model
================================
Estimates own-price elasticity of demand for hotel rooms using an
OLS log-log specification:

    ln(occupancy) = α + β·ln(adr) + γ·ln(competitor_avg_rate)
                    + δ·local_events + ε·school_holiday
                    + ζ·booking_lead_time_band + η·segment_mix + error

β  = own-price elasticity
γ  = competitive (cross-price) elasticity

Fits models per property AND per property_tier, with segmentation
by booking window, day-of-week, and segment type.
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create log-transformed and dummy-coded features for the log-log model."""
    out = df.copy()

    # Log transforms (clip to avoid log(0))
    out["ln_occupancy"] = np.log(out["occupancy_rate"].clip(lower=0.01))
    out["ln_adr"] = np.log(out["adr"].clip(lower=1))
    out["ln_competitor_rate"] = np.log(out["competitor_avg_rate"].clip(lower=1))

    # Dummies for lead-time bands (reference: 31-90d)
    lt_dummies = pd.get_dummies(out["booking_lead_time_band"], prefix="lt", dtype=float)
    for col in ["lt_0-7d", "lt_8-30d", "lt_90d+"]:
        if col not in lt_dummies.columns:
            lt_dummies[col] = 0.0
    lt_dummies = lt_dummies.drop(columns=["lt_31-90d"], errors="ignore")
    out = pd.concat([out, lt_dummies], axis=1)

    # Day-of-week dummies (reference: Monday)
    dow_dummies = pd.get_dummies(out["day_of_week"], prefix="dow", dtype=float)
    dow_dummies = dow_dummies.drop(columns=["dow_Monday"], errors="ignore")

    out = pd.concat([out, dow_dummies], axis=1)

    # Flags as float
    out["local_events_flag"] = out["local_events_flag"].astype(float)
    out["school_holiday_flag"] = out["school_holiday_flag"].astype(float)
    out["bank_holiday_flag"] = out["bank_holiday_flag"].astype(float)
    out["promo_flag"] = out["promo_flag"].astype(float)

    return out


# ---------------------------------------------------------------------------
# Model fitting
# ---------------------------------------------------------------------------

CORE_X_COLS = [
    "ln_adr",
    "ln_competitor_rate",
    "local_events_flag",
    "school_holiday_flag",
    "bank_holiday_flag",
    "promo_flag",
    "business_pct",
    "leisure_pct",
    "weather_index",
]

LT_COLS = ["lt_0-7d", "lt_8-30d", "lt_90d+"]


def _fit_ols(X: pd.DataFrame, y: pd.Series, label: str) -> dict:
    """Run OLS and return a summary dict."""
    X_const = sm.add_constant(X)
    model = sm.OLS(y, X_const).fit()

    beta = model.params.get("ln_adr", np.nan)
    ci = model.conf_int().loc["ln_adr"] if "ln_adr" in model.params.index else [np.nan, np.nan]
    gamma = model.params.get("ln_competitor_rate", np.nan)

    return {
        "segment": label,
        "n_obs": int(model.nobs),
        "own_price_elasticity": round(float(beta), 4),
        "elasticity_ci_lower": round(float(ci[0]), 4),
        "elasticity_ci_upper": round(float(ci[1]), 4),
        "cross_price_elasticity": round(float(gamma), 4),
        "r_squared": round(float(model.rsquared), 4),
        "adj_r_squared": round(float(model.rsquared_adj), 4),
        "p_value_adr": round(float(model.pvalues.get("ln_adr", np.nan)), 6),
        "p_value_competitor": round(float(model.pvalues.get("ln_competitor_rate", np.nan)), 6),
    }


def fit_per_property(df: pd.DataFrame) -> list[dict]:
    """Fit log-log model for each property individually."""
    results = []
    x_cols = [c for c in CORE_X_COLS + LT_COLS if c in df.columns]
    for pid, grp in df.groupby("property_id"):
        if len(grp) < 30:
            continue
        tier = grp["property_tier"].iloc[0]
        pname = grp["property_name"].iloc[0]
        X = grp[x_cols].copy()
        y = grp["ln_occupancy"]
        res = _fit_ols(X, y, label=f"property_{pid}")
        res["property_id"] = pid
        res["property_name"] = pname
        res["property_tier"] = tier
        results.append(res)
    return results


def fit_per_tier(df: pd.DataFrame) -> list[dict]:
    """Fit log-log model per property tier (pooled)."""
    results = []
    x_cols = [c for c in CORE_X_COLS + LT_COLS if c in df.columns]
    for tier, grp in df.groupby("property_tier"):
        X = grp[x_cols].copy()
        y = grp["ln_occupancy"]
        res = _fit_ols(X, y, label=f"tier_{tier}")
        res["property_tier"] = tier
        results.append(res)
    return results


def fit_segmented(df: pd.DataFrame) -> list[dict]:
    """Fit elasticity sliced by booking window, day-of-week, segment."""
    results = []
    x_cols = [c for c in CORE_X_COLS if c in df.columns]

    # By booking window
    for band, grp in df.groupby("booking_lead_time_band"):
        if len(grp) < 30:
            continue
        res = _fit_ols(grp[x_cols], grp["ln_occupancy"], label=f"lead_time_{band}")
        results.append(res)

    # By day-of-week
    for dow, grp in df.groupby("day_of_week"):
        if len(grp) < 30:
            continue
        res = _fit_ols(grp[x_cols], grp["ln_occupancy"], label=f"dow_{dow}")
        results.append(res)

    # By dominant segment (>50% of mix)
    df = df.copy()
    df["dominant_segment"] = np.where(
        df["business_pct"] > 0.5, "business",
        np.where(df["group_pct"] > 0.4, "group", "leisure"),
    )
    for seg, grp in df.groupby("dominant_segment"):
        if len(grp) < 30:
            continue
        res = _fit_ols(grp[x_cols], grp["ln_occupancy"], label=f"segment_{seg}")
        results.append(res)

    return results


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run(data_path: str = "data/hotel_data.csv", output_dir: str = "models") -> pd.DataFrame:
    """Fit all log-log models and save results."""
    df = pd.read_csv(data_path, parse_dates=["week_start_date"])
    df = prepare_features(df)

    prop_results = fit_per_property(df)
    tier_results = fit_per_tier(df)
    seg_results = fit_segmented(df)

    all_results = prop_results + tier_results + seg_results
    results_df = pd.DataFrame(all_results)

    out_dir = Path(output_dir)
    out_dir.mkdir(exist_ok=True)
    results_df.to_csv(out_dir / "log_log_elasticity.csv", index=False)

    # Save a compact JSON for downstream consumption
    summary = {
        "per_property": prop_results,
        "per_tier": tier_results,
        "segmented": seg_results,
    }
    with open(out_dir / "log_log_elasticity.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"Log-log model results saved → {out_dir}")
    print(f"\nElasticity by tier:")
    tier_df = pd.DataFrame(tier_results)
    if not tier_df.empty:
        print(tier_df[["property_tier", "own_price_elasticity", "elasticity_ci_lower",
                        "elasticity_ci_upper", "r_squared", "p_value_adr"]].to_string(index=False))

    return results_df


def main():
    run()


if __name__ == "__main__":
    main()
