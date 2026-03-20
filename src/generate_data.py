"""
Synthetic Hotel Dataset Generator
==================================
Generates 2 years of weekly hotel performance data for 15 properties
with realistic demand-elasticity relationships embedded per tier, segment,
and booking window.
"""

import numpy as np
import pandas as pd
from itertools import product

SEED = 42
np.random.seed(SEED)

# ---------------------------------------------------------------------------
# Property configuration
# ---------------------------------------------------------------------------
PROPERTIES = [
    {"id": "P01", "name": "City Centre Budget Inn",      "tier": "budget",   "rooms": 120, "base_adr": 72},
    {"id": "P02", "name": "Airport Economy Lodge",        "tier": "budget",   "rooms": 100, "base_adr": 65},
    {"id": "P03", "name": "Suburban Value Stay",          "tier": "budget",   "rooms": 80,  "base_adr": 58},
    {"id": "P04", "name": "Downtown Midscale Hotel",      "tier": "midscale", "rooms": 150, "base_adr": 110},
    {"id": "P05", "name": "Convention Centre Suites",     "tier": "midscale", "rooms": 200, "base_adr": 125},
    {"id": "P06", "name": "Riverside Midscale Resort",    "tier": "midscale", "rooms": 140, "base_adr": 105},
    {"id": "P07", "name": "Business Park Hotel",          "tier": "midscale", "rooms": 160, "base_adr": 115},
    {"id": "P08", "name": "Heritage Upscale Hotel",       "tier": "upscale",  "rooms": 180, "base_adr": 185},
    {"id": "P09", "name": "Marina Upscale Resort",        "tier": "upscale",  "rooms": 200, "base_adr": 210},
    {"id": "P10", "name": "Golf & Spa Upscale",           "tier": "upscale",  "rooms": 160, "base_adr": 195},
    {"id": "P11", "name": "Mountain View Upscale",        "tier": "upscale",  "rooms": 130, "base_adr": 175},
    {"id": "P12", "name": "Grand Luxury Palace",          "tier": "luxury",   "rooms": 120, "base_adr": 380},
    {"id": "P13", "name": "Waterfront Luxury Collection", "tier": "luxury",   "rooms": 90,  "base_adr": 420},
    {"id": "P14", "name": "Royal Penthouse Suites",       "tier": "luxury",   "rooms": 60,  "base_adr": 500},
    {"id": "P15", "name": "Island Boutique Luxury",       "tier": "luxury",   "rooms": 50,  "base_adr": 450},
]

# Elasticity parameters (own-price)
TIER_ELASTICITY = {
    "budget":   -1.8,
    "midscale": -1.2,
    "upscale":  -0.9,
    "luxury":   -0.7,
}

LEAD_TIME_ELASTICITY = {
    "0-7d":   -2.5,
    "8-30d":  -1.5,
    "31-90d": -1.0,
    "90d+":   -0.9,
}

SEGMENT_ELASTICITY_ADJ = {
    "business": 0.5,   # additive — makes demand less elastic
    "leisure":  0.0,
    "group":   -0.2,   # groups negotiate hard
}

LEAD_TIME_BANDS = ["0-7d", "8-30d", "31-90d", "90d+"]
LEAD_TIME_WEIGHTS = [0.20, 0.35, 0.30, 0.15]

DAYS_OF_WEEK = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

# ---------------------------------------------------------------------------
# Seasonality helpers
# ---------------------------------------------------------------------------

def _season_factor(week: int) -> float:
    """Peak in summer (weeks 22-35) and Christmas (weeks 50-52)."""
    summer = np.exp(-0.5 * ((week - 28) / 6) ** 2) * 0.25
    christmas = np.exp(-0.5 * ((week - 51) / 1.5) ** 2) * 0.18
    return 1.0 + summer + christmas


def _dow_factor(dow: str, tier: str) -> float:
    """Weekend premium for leisure; midweek for business."""
    weekend = dow in ("Friday", "Saturday", "Sunday")
    if tier in ("luxury", "upscale"):
        return 1.10 if weekend else 0.95
    if tier == "midscale":
        return 1.05 if weekend else 1.00
    return 1.08 if weekend else 0.96  # budget


def _school_holiday(week: int) -> bool:
    return week in list(range(1, 3)) + list(range(14, 17)) + list(range(29, 35)) + list(range(51, 53))


def _bank_holiday(week: int) -> bool:
    return week in [1, 14, 18, 22, 35, 51, 52]


def _local_event(week: int, prop_id: str) -> bool:
    """Deterministic pseudo-random events ~12% of weeks."""
    h = hash((week, prop_id))
    return (h % 100) < 12

# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------

def generate_dataset() -> pd.DataFrame:
    """Return a DataFrame with all required hospitality columns."""

    date_range = pd.date_range("2024-01-01", "2025-12-28", freq="W-MON")
    rows = []

    for prop in PROPERTIES:
        pid = prop["id"]
        tier = prop["tier"]
        base_adr = prop["base_adr"]
        rooms_avail = prop["rooms"]
        base_elasticity = TIER_ELASTICITY[tier]

        for week_start in date_range:
            wk = week_start.isocalendar().week
            season = _season_factor(wk)
            school_hol = _school_holiday(wk)
            bank_hol = _bank_holiday(wk)
            event = _local_event(wk, pid)
            weather = np.clip(int(3 * season + np.random.normal(0, 0.6)), 1, 5)

            # Competitor rate — loosely tracks our rate with noise
            comp_rate = base_adr * season * np.random.uniform(0.90, 1.12)

            # Decide booking-window distribution for this week
            lt_band = np.random.choice(LEAD_TIME_BANDS, p=LEAD_TIME_WEIGHTS)

            # Dominant segment (varies with day-of-week proxy and tier)
            if tier in ("midscale",) and wk % 2 == 0:
                dominant_seg = "business"
            elif tier == "luxury":
                dominant_seg = "leisure"
            else:
                dominant_seg = np.random.choice(["leisure", "business", "group"], p=[0.55, 0.30, 0.15])

            # Effective elasticity
            eff_elasticity = (
                base_elasticity
                + (LEAD_TIME_ELASTICITY[lt_band] - base_elasticity) * 0.4
                + SEGMENT_ELASTICITY_ADJ[dominant_seg]
            )

            # --- Pricing decision (varies around base) ---
            rate_mult = season * np.random.uniform(0.88, 1.15)
            if event:
                rate_mult *= np.random.uniform(1.08, 1.25)
            if school_hol:
                rate_mult *= np.random.uniform(1.02, 1.10)

            adr = round(base_adr * rate_mult, 2)

            # --- Demand response via elasticity ---
            log_price_ratio = np.log(adr / (base_adr * season))
            base_occ = 0.72 if tier == "budget" else 0.68 if tier == "midscale" else 0.65 if tier == "upscale" else 0.60
            if event:
                base_occ += 0.10
            if school_hol:
                base_occ += 0.05
            if bank_hol:
                base_occ += 0.03

            log_occ = np.log(base_occ) + eff_elasticity * log_price_ratio
            # Competitive rate effect — positive cross-elasticity
            comp_ratio = np.log(comp_rate / (base_adr * season))
            log_occ += 0.3 * comp_ratio  # γ ≈ 0.3

            occ = np.clip(np.exp(log_occ) + np.random.normal(0, 0.04), 0.10, 0.99)

            # Channel & segment mixes
            if dominant_seg == "business":
                direct_pct, ota_pct, gds_pct, groups_pct = 0.25, 0.30, 0.35, 0.10
                leisure_pct, biz_pct, grp_pct = 0.20, 0.60, 0.20
            elif dominant_seg == "group":
                direct_pct, ota_pct, gds_pct, groups_pct = 0.15, 0.25, 0.15, 0.45
                leisure_pct, biz_pct, grp_pct = 0.25, 0.25, 0.50
            else:
                direct_pct, ota_pct, gds_pct, groups_pct = 0.35, 0.45, 0.10, 0.10
                leisure_pct, biz_pct, grp_pct = 0.65, 0.20, 0.15

            # Small noise on channel mix
            mix_noise = np.random.dirichlet([10, 10, 10, 10])
            direct_pct = round(0.7 * direct_pct + 0.3 * mix_noise[0], 3)
            ota_pct    = round(0.7 * ota_pct + 0.3 * mix_noise[1], 3)
            gds_pct    = round(0.7 * gds_pct + 0.3 * mix_noise[2], 3)
            groups_pct = round(1 - direct_pct - ota_pct - gds_pct, 3)

            seg_noise = np.random.dirichlet([10, 10, 10])
            leisure_pct = round(0.7 * leisure_pct + 0.3 * seg_noise[0], 3)
            biz_pct     = round(0.7 * biz_pct + 0.3 * seg_noise[1], 3)
            grp_pct     = round(1 - leisure_pct - biz_pct, 3)

            # Promo flag — ~15% of weeks
            promo = np.random.random() < 0.15

            # Rate parity — 90% of time parity holds
            rate_parity = abs(adr - comp_rate) / comp_rate < 0.05 or np.random.random() < 0.90

            rooms_sold = int(round(occ * rooms_avail))
            rooms_rev = round(rooms_sold * adr, 2)
            revpar = round(adr * occ, 2)

            # GOPPAR — simplified: ~40-55% of RevPAR depending on tier
            gop_margin = {"budget": 0.40, "midscale": 0.45, "upscale": 0.50, "luxury": 0.55}[tier]
            goppar = round(revpar * gop_margin * np.random.uniform(0.90, 1.10), 2)

            cancel_rate = round(np.clip(np.random.normal(0.15, 0.05), 0.02, 0.40), 3)

            for dow in DAYS_OF_WEEK:
                dow_f = _dow_factor(dow, tier)
                d_adr = round(adr * dow_f, 2)
                d_occ = np.clip(occ * (1 / dow_f) ** abs(eff_elasticity) * np.random.uniform(0.97, 1.03), 0.10, 0.99)
                d_occ = round(d_occ, 4)
                d_rooms_sold = int(round(d_occ * rooms_avail))
                d_rooms_rev = round(d_rooms_sold * d_adr, 2)
                d_revpar = round(d_adr * d_occ, 2)
                d_goppar = round(d_revpar * gop_margin * np.random.uniform(0.90, 1.10), 2)

                rows.append({
                    "property_id":            pid,
                    "property_name":          prop["name"],
                    "property_tier":          tier,
                    "week_start_date":        week_start.strftime("%Y-%m-%d"),
                    "day_of_week":            dow,
                    "adr":                    d_adr,
                    "occupancy_rate":         d_occ,
                    "revpar":                 d_revpar,
                    "goppar":                 d_goppar,
                    "rooms_available":        rooms_avail,
                    "rooms_sold":             d_rooms_sold,
                    "rooms_revenue":          d_rooms_rev,
                    "competitor_avg_rate":    round(comp_rate * dow_f, 2),
                    "rate_parity_flag":       rate_parity,
                    "booking_lead_time_band": lt_band,
                    "direct_pct":             direct_pct,
                    "ota_booking_pct":        ota_pct,
                    "gds_pct":                gds_pct,
                    "groups_pct":             groups_pct,
                    "leisure_pct":            leisure_pct,
                    "business_pct":           biz_pct,
                    "group_pct":              grp_pct,
                    "local_events_flag":      event,
                    "school_holiday_flag":    school_hol,
                    "bank_holiday_flag":      bank_hol,
                    "weather_index":          weather,
                    "promo_flag":             promo,
                    "cancellation_rate_pct":  cancel_rate,
                })

    df = pd.DataFrame(rows)
    df["week_start_date"] = pd.to_datetime(df["week_start_date"])
    return df


def add_lagged_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add 7-day lagged occupancy per property."""
    df = df.sort_values(["property_id", "week_start_date", "day_of_week"]).reset_index(drop=True)
    df["lagged_occupancy_7d"] = (
        df.groupby("property_id")["occupancy_rate"]
        .shift(7)
        .fillna(df["occupancy_rate"].mean())
    )
    return df


def main():
    print("Generating synthetic hotel dataset …")
    df = generate_dataset()
    df = add_lagged_features(df)
    out_path = "data/hotel_data.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df):,} rows → {out_path}")
    print(f"Properties: {df['property_id'].nunique()}")
    print(f"Date range: {df['week_start_date'].min()} → {df['week_start_date'].max()}")
    print(f"\nSample RevPAR by tier:")
    print(df.groupby("property_tier")["revpar"].mean().round(2).to_string())


if __name__ == "__main__":
    main()
