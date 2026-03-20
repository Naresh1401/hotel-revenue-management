"""
RevPAR Optimizer
=================
For each property + booking-window combination, find the ADR that
maximises RevPAR(p) = p × occupancy(p) subject to:
  - Rate parity:  within ±5 % of OTA / competitor rate
  - Brand floor:  budget ≥ £50, midscale ≥ £80, upscale ≥ £150, luxury ≥ £250
  - Competitor proximity:  within ±20 % of comp-set average

Also computes the occupancy-maximising rate for distress inventory.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

# ---------------------------------------------------------------------------
# Constraints / floors
# ---------------------------------------------------------------------------

BRAND_FLOOR = {
    "budget": 50,
    "midscale": 80,
    "upscale": 150,
    "luxury": 250,
}

BRAND_CEILING = {
    "budget": 200,
    "midscale": 350,
    "upscale": 600,
    "luxury": 1200,
}

RATE_PARITY_TOLERANCE = 0.05      # ±5 %
COMPETITOR_PROXIMITY = 0.20        # ±20 %


def _demand_curve(price: float, base_occ: float, base_adr: float,
                  elasticity: float) -> float:
    """Log-log demand: occupancy = base_occ × (price / base_adr) ^ elasticity."""
    if price <= 0 or base_adr <= 0:
        return 0.0
    occ = base_occ * (price / base_adr) ** elasticity
    return float(np.clip(occ, 0.0, 1.0))


def _revpar(price: float, base_occ: float, base_adr: float,
            elasticity: float, rooms_available: int) -> float:
    """RevPAR = ADR × occupancy."""
    occ = _demand_curve(price, base_occ, base_adr, elasticity)
    return price * occ


# ---------------------------------------------------------------------------
# Single-property optimiser
# ---------------------------------------------------------------------------

def optimize_rate(
    base_occ: float,
    base_adr: float,
    elasticity: float,
    rooms_available: int,
    competitor_rate: float,
    tier: str,
) -> dict:
    """
    Find the ADR that maximises RevPAR within business constraints.

    Returns dict with: recommended_rate, expected_occupancy, expected_revpar,
    occupancy_max_rate, and constraint metadata.
    """
    floor = max(
        BRAND_FLOOR[tier],
        competitor_rate * (1 - COMPETITOR_PROXIMITY),
    )
    ceiling = min(
        BRAND_CEILING[tier],
        competitor_rate * (1 + COMPETITOR_PROXIMITY),
    )

    # Rate parity bounds (tighter)
    parity_lo = competitor_rate * (1 - RATE_PARITY_TOLERANCE)
    parity_hi = competitor_rate * (1 + RATE_PARITY_TOLERANCE)

    # Use the intersection of all bounds
    lb = max(floor, parity_lo)
    ub = min(ceiling, parity_hi)

    # Safety: if bounds are inverted, relax parity
    if lb > ub:
        lb = floor
        ub = ceiling

    # --- RevPAR maximisation ---
    result = minimize_scalar(
        lambda p: -_revpar(p, base_occ, base_adr, elasticity, rooms_available),
        bounds=(lb, ub),
        method="bounded",
    )
    opt_rate = round(result.x, 2)
    opt_occ = round(_demand_curve(opt_rate, base_occ, base_adr, elasticity), 4)
    opt_revpar = round(opt_rate * opt_occ, 2)

    # --- Occupancy-maximising rate (lowest feasible rate) ---
    occ_max_rate = round(lb, 2)
    occ_max_occ = round(_demand_curve(occ_max_rate, base_occ, base_adr, elasticity), 4)
    occ_max_revpar = round(occ_max_rate * occ_max_occ, 2)

    # Current performance
    current_revpar = round(base_adr * base_occ, 2)
    rate_change_pct = round((opt_rate - base_adr) / base_adr * 100, 2)

    return {
        "recommended_rate": opt_rate,
        "expected_occupancy": opt_occ,
        "expected_revpar": opt_revpar,
        "occupancy_max_rate": occ_max_rate,
        "occupancy_max_occ": occ_max_occ,
        "occupancy_max_revpar": occ_max_revpar,
        "current_adr": round(base_adr, 2),
        "current_revpar": current_revpar,
        "vs_current_rate_change_pct": rate_change_pct,
        "revpar_uplift_pct": round((opt_revpar - current_revpar) / max(current_revpar, 0.01) * 100, 2),
        "constraint_lb": round(lb, 2),
        "constraint_ub": round(ub, 2),
        "brand_floor": BRAND_FLOOR[tier],
    }


# ---------------------------------------------------------------------------
# Portfolio-level optimisation
# ---------------------------------------------------------------------------

def optimize_portfolio(
    df: pd.DataFrame,
    elasticity_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Optimise rates for every property × booking-window combination.

    Parameters
    ----------
    df : hotel data (aggregated to property × lead-time level)
    elasticity_df : log-log elasticity per property (from log_log_model)

    Returns DataFrame of recommendations.
    """
    # Build elasticity lookup: use per-property if available, else tier
    elas_by_prop = {}
    prop_rows = elasticity_df[elasticity_df["segment"].str.startswith("property_")]
    for _, row in prop_rows.iterrows():
        elas_by_prop[row.get("property_id", "")] = row["own_price_elasticity"]

    elas_by_tier = {}
    tier_rows = elasticity_df[elasticity_df["segment"].str.startswith("tier_")]
    for _, row in tier_rows.iterrows():
        elas_by_tier[row["property_tier"]] = row["own_price_elasticity"]

    # Aggregate data to property × lead-time band
    agg = (
        df.groupby(["property_id", "property_name", "property_tier",
                     "booking_lead_time_band", "rooms_available"])
        .agg(
            mean_adr=("adr", "mean"),
            mean_occ=("occupancy_rate", "mean"),
            mean_comp_rate=("competitor_avg_rate", "mean"),
        )
        .reset_index()
    )

    recommendations = []
    for _, row in agg.iterrows():
        pid = row["property_id"]
        tier = row["property_tier"]
        elasticity = elas_by_prop.get(pid, elas_by_tier.get(tier, -1.2))

        rec = optimize_rate(
            base_occ=row["mean_occ"],
            base_adr=row["mean_adr"],
            elasticity=elasticity,
            rooms_available=row["rooms_available"],
            competitor_rate=row["mean_comp_rate"],
            tier=tier,
        )
        rec["property_id"] = pid
        rec["property_name"] = row["property_name"]
        rec["property_tier"] = tier
        rec["booking_lead_time_band"] = row["booking_lead_time_band"]
        rec["elasticity_used"] = elasticity
        recommendations.append(rec)

    return pd.DataFrame(recommendations)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(
    data_path: str = "data/hotel_data.csv",
    elasticity_path: str = "models/log_log_elasticity.csv",
    output_dir: str = "models",
) -> pd.DataFrame:
    """Run portfolio optimisation and save results."""
    df = pd.read_csv(data_path, parse_dates=["week_start_date"])
    elas = pd.read_csv(elasticity_path)

    recs = optimize_portfolio(df, elas)

    out_dir = Path(output_dir)
    out_dir.mkdir(exist_ok=True)
    recs.to_csv(out_dir / "rate_recommendations.csv", index=False)

    summary = recs.groupby("property_tier").agg(
        avg_current_revpar=("current_revpar", "mean"),
        avg_optimal_revpar=("expected_revpar", "mean"),
        avg_uplift_pct=("revpar_uplift_pct", "mean"),
    ).round(2)

    print(f"Rate recommendations saved → {out_dir / 'rate_recommendations.csv'}")
    print(f"\nRevPAR uplift summary by tier:")
    print(summary.to_string())

    return recs


def main():
    run()


if __name__ == "__main__":
    main()
