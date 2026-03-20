"""
Tests for Hotel Revenue Management
=====================================
Validates elasticity signs, optimizer constraints, rate parity,
and last-minute trigger logic.
"""

import numpy as np
import pandas as pd
import pytest

from src.generate_data import generate_dataset, add_lagged_features, PROPERTIES, TIER_ELASTICITY
from src.log_log_model import prepare_features, fit_per_tier, fit_per_property
from src.optimizer import optimize_rate, BRAND_FLOOR, BRAND_CEILING
from src.last_minute import (
    identify_distressed_inventory,
    compute_distress_rates,
    OCCUPANCY_TRIGGER,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def hotel_data():
    """Generate a smaller dataset for testing speed."""
    df = generate_dataset()
    df = add_lagged_features(df)
    return df


@pytest.fixture(scope="module")
def prepped_data(hotel_data):
    return prepare_features(hotel_data)


# ---------------------------------------------------------------------------
# Step 1 — Data generation
# ---------------------------------------------------------------------------

class TestDataGeneration:
    def test_row_count(self, hotel_data):
        """15 properties × ~104 weeks × 7 days ≈ 10,920+ rows."""
        assert len(hotel_data) > 10_000

    def test_all_properties_present(self, hotel_data):
        assert hotel_data["property_id"].nunique() == 15

    def test_tiers(self, hotel_data):
        tiers = set(hotel_data["property_tier"].unique())
        assert tiers == {"budget", "midscale", "upscale", "luxury"}

    def test_occupancy_bounded(self, hotel_data):
        assert hotel_data["occupancy_rate"].min() >= 0.05
        assert hotel_data["occupancy_rate"].max() <= 1.0

    def test_revpar_formula(self, hotel_data):
        """RevPAR ≈ ADR × occupancy (within rounding tolerance)."""
        computed = hotel_data["adr"] * hotel_data["occupancy_rate"]
        diff = (hotel_data["revpar"] - computed).abs()
        assert diff.max() < 2.0  # rounding tolerance

    def test_columns_exist(self, hotel_data):
        required = [
            "property_id", "property_name", "property_tier",
            "week_start_date", "day_of_week", "adr", "occupancy_rate",
            "revpar", "goppar", "rooms_available", "rooms_sold",
            "rooms_revenue", "competitor_avg_rate", "rate_parity_flag",
            "booking_lead_time_band", "direct_pct", "ota_booking_pct",
            "gds_pct", "groups_pct", "leisure_pct", "business_pct",
            "group_pct", "local_events_flag", "school_holiday_flag",
            "bank_holiday_flag", "weather_index", "promo_flag",
            "cancellation_rate_pct",
        ]
        for col in required:
            assert col in hotel_data.columns, f"Missing column: {col}"

    def test_lead_time_bands(self, hotel_data):
        bands = set(hotel_data["booking_lead_time_band"].unique())
        assert bands == {"0-7d", "8-30d", "31-90d", "90d+"}


# ---------------------------------------------------------------------------
# Step 2 — Elasticity coefficients
# ---------------------------------------------------------------------------

class TestElasticity:
    def test_tier_elasticity_negative(self, prepped_data):
        """Own-price elasticity must be negative for all tiers."""
        results = fit_per_tier(prepped_data)
        for row in results:
            assert row["own_price_elasticity"] < 0, (
                f"Tier {row['property_tier']}: elasticity should be negative, "
                f"got {row['own_price_elasticity']}"
            )

    def test_property_elasticity_negative(self, prepped_data):
        """Own-price elasticity should be negative for each property."""
        results = fit_per_property(prepped_data)
        negative_count = sum(1 for r in results if r["own_price_elasticity"] < 0)
        # At least 80% should be negative (some edge cases with noise)
        assert negative_count / len(results) > 0.80

    def test_luxury_less_elastic_than_budget(self, prepped_data):
        """Luxury should have smaller |elasticity| than budget."""
        results = fit_per_tier(prepped_data)
        tier_elas = {r["property_tier"]: r["own_price_elasticity"] for r in results}
        if "luxury" in tier_elas and "budget" in tier_elas:
            assert abs(tier_elas["luxury"]) < abs(tier_elas["budget"]), (
                f"Expected |luxury| < |budget|, got luxury={tier_elas['luxury']}, "
                f"budget={tier_elas['budget']}"
            )


# ---------------------------------------------------------------------------
# Step 4 — RevPAR Optimizer
# ---------------------------------------------------------------------------

class TestOptimizer:
    @pytest.mark.parametrize("tier", ["budget", "midscale", "upscale", "luxury"])
    def test_rate_within_brand_floor(self, tier):
        result = optimize_rate(
            base_occ=0.65,
            base_adr=BRAND_FLOOR[tier] + 30,
            elasticity=-1.2,
            rooms_available=150,
            competitor_rate=BRAND_FLOOR[tier] + 40,
            tier=tier,
        )
        assert result["recommended_rate"] >= BRAND_FLOOR[tier], (
            f"Rate {result['recommended_rate']} below brand floor {BRAND_FLOOR[tier]}"
        )

    @pytest.mark.parametrize("tier", ["budget", "midscale", "upscale", "luxury"])
    def test_rate_within_ceiling(self, tier):
        result = optimize_rate(
            base_occ=0.65,
            base_adr=BRAND_CEILING[tier] - 50,
            elasticity=-0.8,
            rooms_available=150,
            competitor_rate=BRAND_CEILING[tier] - 40,
            tier=tier,
        )
        assert result["recommended_rate"] <= BRAND_CEILING[tier]

    def test_rate_parity_not_violated(self):
        comp_rate = 120.0
        result = optimize_rate(
            base_occ=0.70, base_adr=115.0, elasticity=-1.2,
            rooms_available=150, competitor_rate=comp_rate, tier="midscale",
        )
        # Within ±20% of competitor (relaxed competitor proximity)
        assert result["recommended_rate"] >= comp_rate * 0.80
        assert result["recommended_rate"] <= comp_rate * 1.20

    def test_revpar_positive(self):
        result = optimize_rate(
            base_occ=0.60, base_adr=200.0, elasticity=-0.9,
            rooms_available=100, competitor_rate=210.0, tier="upscale",
        )
        assert result["expected_revpar"] > 0


# ---------------------------------------------------------------------------
# Step 6 — Last-Minute Trigger
# ---------------------------------------------------------------------------

class TestLastMinute:
    def test_trigger_fires_below_threshold(self):
        """Distressed inventory should be flagged when occ < 60 %."""
        data = pd.DataFrame([{
            "property_id": "P01", "property_name": "Test Hotel",
            "property_tier": "budget", "week_start_date": pd.Timestamp("2025-01-06"),
            "rooms_available": 100, "rooms_sold": 45,
            "occupancy_rate": 0.45, "adr": 70, "competitor_avg_rate": 75,
            "booking_lead_time_band": "0-7d",
            "day_of_week": "Monday",
        }])
        inv = identify_distressed_inventory(data)
        assert inv["is_distressed"].any(), "Should flag occupancy 0.45 as distressed"

    def test_trigger_does_not_fire_above_threshold(self):
        """Should NOT trigger when occupancy > 60 %."""
        data = pd.DataFrame([{
            "property_id": "P01", "property_name": "Test Hotel",
            "property_tier": "budget", "week_start_date": pd.Timestamp("2025-01-06"),
            "rooms_available": 100, "rooms_sold": 75,
            "occupancy_rate": 0.75, "adr": 70, "competitor_avg_rate": 75,
            "booking_lead_time_band": "0-7d",
            "day_of_week": "Monday",
        }])
        inv = identify_distressed_inventory(data)
        assert not inv["is_distressed"].any(), "Should NOT flag occupancy 0.75"

    def test_distress_rate_above_brand_floor(self):
        """Distress rate must respect brand floor."""
        data = pd.DataFrame([{
            "property_id": "P14", "property_name": "Royal Penthouse",
            "property_tier": "luxury", "week_start_date": pd.Timestamp("2025-01-06"),
            "rooms_available": 60, "rooms_sold": 20,
            "occupancy_rate": 0.33, "adr": 500, "competitor_avg_rate": 280,
            "booking_lead_time_band": "0-7d",
            "day_of_week": "Monday",
        }])
        inv = identify_distressed_inventory(data)
        distress = compute_distress_rates(inv)
        if not distress.empty:
            assert distress["distress_rate"].iloc[0] >= BRAND_FLOOR["luxury"]
