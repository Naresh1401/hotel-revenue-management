"""
Yield Management Scenarios
============================
Simulates 90-day forward-looking revenue outcomes under four strategies
per property, using the estimated elasticity curves.

Scenario A — Revenue max:     optimal rate per booking window
Scenario B — Market parity:   match competitor rate exactly
Scenario C — Occupancy fill:  10 % below competitor to fill rooms
Scenario D — Event premium:   +25 % surcharge on local-event weeks
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.optimizer import _demand_curve, BRAND_FLOOR

# ---------------------------------------------------------------------------
# Scenario definitions
# ---------------------------------------------------------------------------

def _run_scenario(
    weeks: pd.DataFrame,
    elasticity: float,
    base_adr: float,
    base_occ: float,
    rooms_available: int,
    strategy: str,
    tier: str,
) -> list[dict]:
    """Simulate one scenario across forward weeks."""
    records = []
    floor = BRAND_FLOOR[tier]

    for _, wk in weeks.iterrows():
        comp_rate = wk["competitor_avg_rate"]
        is_event = wk["local_events_flag"]

        if strategy == "revenue_max":
            # Use midpoint of a quick grid search within ±30 % of comp rate
            candidates = np.linspace(max(floor, comp_rate * 0.8), comp_rate * 1.3, 200)
            revpars = [p * _demand_curve(p, base_occ, base_adr, elasticity) for p in candidates]
            rate = candidates[np.argmax(revpars)]

        elif strategy == "market_parity":
            rate = max(floor, comp_rate)

        elif strategy == "occupancy_fill":
            rate = max(floor, comp_rate * 0.90)

        elif strategy == "event_premium":
            rate = max(floor, comp_rate * (1.25 if is_event else 1.0))

        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        occ = _demand_curve(rate, base_occ, base_adr, elasticity)
        revpar = rate * occ
        total_rev = revpar * rooms_available

        records.append({
            "week_start_date": wk["week_start_date"],
            "strategy": strategy,
            "rate": round(rate, 2),
            "expected_occupancy": round(occ, 4),
            "expected_adr": round(rate, 2),
            "expected_revpar": round(revpar, 2),
            "expected_total_revenue": round(total_rev, 2),
            "local_events_flag": is_event,
        })

    return records


# ---------------------------------------------------------------------------
# Per-property scenario runner
# ---------------------------------------------------------------------------

STRATEGIES = ["revenue_max", "market_parity", "occupancy_fill", "event_premium"]


def simulate_property(
    df_prop: pd.DataFrame,
    elasticity: float,
    forward_weeks: int = 13,  # ~90 days
) -> pd.DataFrame:
    """Run all four scenarios for a single property."""
    pid = df_prop["property_id"].iloc[0]
    tier = df_prop["property_tier"].iloc[0]
    rooms_available = df_prop["rooms_available"].iloc[0]
    base_adr = df_prop["adr"].mean()
    base_occ = df_prop["occupancy_rate"].mean()

    # Use the last N weeks of data as the "forward" period
    latest = df_prop.sort_values("week_start_date").drop_duplicates("week_start_date").tail(forward_weeks)

    all_records = []
    for strat in STRATEGIES:
        recs = _run_scenario(
            latest, elasticity, base_adr, base_occ,
            rooms_available, strat, tier,
        )
        for r in recs:
            r["property_id"] = pid
            r["property_tier"] = tier
        all_records.extend(recs)

    return pd.DataFrame(all_records)


def run_all_scenarios(
    df: pd.DataFrame,
    elasticity_df: pd.DataFrame,
) -> pd.DataFrame:
    """Run scenarios for all 15 properties."""
    # Build elasticity lookup
    elas_map = {}
    for _, row in elasticity_df.iterrows():
        if str(row.get("segment", "")).startswith("property_"):
            elas_map[row.get("property_id", "")] = row["own_price_elasticity"]
    tier_map = {}
    for _, row in elasticity_df.iterrows():
        if str(row.get("segment", "")).startswith("tier_"):
            tier_map[row["property_tier"]] = row["own_price_elasticity"]

    frames = []
    for pid, grp in df.groupby("property_id"):
        tier = grp["property_tier"].iloc[0]
        elas = elas_map.get(pid, tier_map.get(tier, -1.2))
        sc = simulate_property(grp, elas)
        frames.append(sc)

    return pd.concat(frames, ignore_index=True)


def summarise_scenarios(scenarios: pd.DataFrame) -> pd.DataFrame:
    """Portfolio-level summary per strategy."""
    return (
        scenarios.groupby("strategy")
        .agg(
            mean_occupancy=("expected_occupancy", "mean"),
            mean_adr=("expected_adr", "mean"),
            mean_revpar=("expected_revpar", "mean"),
            total_revenue=("expected_total_revenue", "sum"),
        )
        .round(2)
        .reset_index()
    )


def vs_current(scenarios: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """Add vs_current_strategy columns."""
    current_revpar = df["revpar"].mean()
    summary = summarise_scenarios(scenarios)
    summary["vs_current_revpar_pct"] = round(
        (summary["mean_revpar"] - current_revpar) / current_revpar * 100, 2
    )
    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(
    data_path: str = "data/hotel_data.csv",
    elasticity_path: str = "models/log_log_elasticity.csv",
    output_dir: str = "models",
) -> pd.DataFrame:
    """Execute all scenarios and save outputs."""
    df = pd.read_csv(data_path, parse_dates=["week_start_date"])
    elas = pd.read_csv(elasticity_path)

    print("Running yield-management scenarios (90-day forward) …")
    scenarios = run_all_scenarios(df, elas)

    out_dir = Path(output_dir)
    out_dir.mkdir(exist_ok=True)
    scenarios.to_csv(out_dir / "scenario_results.csv", index=False)

    summary = vs_current(scenarios, df)
    summary.to_csv(out_dir / "scenario_summary.csv", index=False)

    print(f"Scenarios saved → {out_dir}")
    print("\nPortfolio scenario comparison:")
    print(summary.to_string(index=False))

    return scenarios


def main():
    run()


if __name__ == "__main__":
    main()
