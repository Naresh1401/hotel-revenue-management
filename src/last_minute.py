"""
Last-Minute Pricing Engine
============================
Monitors unsold inventory in the 0 – 7 day booking window and triggers
distress pricing when occupancy falls below threshold.

Strategy:
  - Trigger: occupancy < 60 % at T − 5 days
  - Distress rate = max(brand_floor, competitor_rate × 0.85)
  - Channel: push distress inventory to OTAs only (protect direct rate)
  - Goal: recover revenue vs. leaving rooms empty (perishable inventory)
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.optimizer import BRAND_FLOOR, _demand_curve

OCCUPANCY_TRIGGER = 0.60       # fire distress pricing below this
DISTRESS_DISCOUNT = 0.85       # competitor_rate × 0.85
DAYS_BEFORE_ARRIVAL = 5        # monitoring window


def identify_distressed_inventory(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flag property-weeks where last-minute occupancy is below threshold.
    Filters to the 0-7d booking window only.
    """
    last_min = df[df["booking_lead_time_band"] == "0-7d"].copy()

    # Aggregate to property × week
    agg = (
        last_min.groupby(["property_id", "property_name", "property_tier",
                          "week_start_date", "rooms_available"])
        .agg(
            current_occupancy=("occupancy_rate", "mean"),
            current_adr=("adr", "mean"),
            competitor_rate=("competitor_avg_rate", "mean"),
            rooms_sold=("rooms_sold", "mean"),
        )
        .reset_index()
    )
    agg["rooms_sold"] = agg["rooms_sold"].round().astype(int)
    agg["unsold_rooms"] = agg["rooms_available"] - agg["rooms_sold"]
    agg["is_distressed"] = agg["current_occupancy"] < OCCUPANCY_TRIGGER

    return agg


def compute_distress_rates(distressed: pd.DataFrame) -> pd.DataFrame:
    """
    For each distressed property-week, compute the recommended distress rate
    and expected revenue recovery.
    """
    rows = []
    for _, r in distressed[distressed["is_distressed"]].iterrows():
        tier = r["property_tier"]
        floor = BRAND_FLOOR[tier]
        distress_rate = max(floor, r["competitor_rate"] * DISTRESS_DISCOUNT)

        # Expected additional rooms sold at distress rate
        # Use a high elasticity (last-minute = -2.5)
        elasticity = -2.5
        base_occ = r["current_occupancy"]
        base_adr = r["current_adr"]
        new_occ = _demand_curve(distress_rate, base_occ, base_adr, elasticity)
        additional_occ = max(0, new_occ - base_occ)
        additional_rooms = int(round(additional_occ * r["rooms_available"]))

        revenue_at_nothing = 0  # unsold rooms = £0 (perishable)
        revenue_at_distress = additional_rooms * distress_rate

        rows.append({
            "property_id":       r["property_id"],
            "property_name":     r["property_name"],
            "property_tier":     tier,
            "week_start_date":   r["week_start_date"],
            "rooms_available":   r["rooms_available"],
            "current_occupancy": round(r["current_occupancy"], 4),
            "unsold_rooms":      r["unsold_rooms"],
            "current_adr":       round(r["current_adr"], 2),
            "competitor_rate":   round(r["competitor_rate"], 2),
            "distress_rate":     round(distress_rate, 2),
            "rate_discount_pct": round((1 - distress_rate / r["current_adr"]) * 100, 2),
            "expected_new_occ":  round(new_occ, 4),
            "additional_rooms":  additional_rooms,
            "recovered_revenue": round(revenue_at_distress, 2),
            "channel_strategy":  "OTA-only (protect direct rate)",
        })

    return pd.DataFrame(rows)


def portfolio_recovery_summary(distress_df: pd.DataFrame) -> dict:
    """Aggregate distressed revenue recovery across the portfolio."""
    if distress_df.empty:
        return {"distressed_weeks": 0, "total_recovered_revenue": 0, "avg_discount_pct": 0}
    return {
        "distressed_weeks": len(distress_df),
        "total_recovered_revenue": round(distress_df["recovered_revenue"].sum(), 2),
        "avg_discount_pct": round(distress_df["rate_discount_pct"].mean(), 2),
        "avg_additional_rooms_per_week": round(distress_df["additional_rooms"].mean(), 1),
        "total_additional_rooms": int(distress_df["additional_rooms"].sum()),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(
    data_path: str = "data/hotel_data.csv",
    output_dir: str = "models",
) -> pd.DataFrame:
    """Identify distressed inventory and compute recovery rates."""
    df = pd.read_csv(data_path, parse_dates=["week_start_date"])

    print("Scanning for distressed inventory (0-7d window, occ < 60 %) …")
    inventory = identify_distressed_inventory(df)
    distress = compute_distress_rates(inventory)

    out_dir = Path(output_dir)
    out_dir.mkdir(exist_ok=True)
    inventory.to_csv(out_dir / "last_minute_inventory.csv", index=False)
    distress.to_csv(out_dir / "last_minute_distress_rates.csv", index=False)

    summary = portfolio_recovery_summary(distress)

    print(f"Last-minute pricing saved → {out_dir}")
    print(f"Distressed property-weeks: {summary['distressed_weeks']}")
    print(f"Total recovered revenue:   £{summary['total_recovered_revenue']:,.2f}")
    print(f"Avg rate discount:         {summary['avg_discount_pct']:.1f} %")

    return distress


def main():
    run()


if __name__ == "__main__":
    main()
