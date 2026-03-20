"""
Streamlit Pricing Simulator — Hotel Revenue Management
=========================================================
Five-tab dashboard:
  Tab 1 — Property Overview:    RevPAR heatmap, elasticity table
  Tab 2 — Rate Optimizer:       Interactive demand curve + rate slider
  Tab 3 — Scenario Planner:     Side-by-side 90-day comparison
  Tab 4 — Last-Minute Engine:   Distress inventory alerts
  Tab 5 — Competitive Position: Rate index / RGI analysis
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.optimizer import _demand_curve, BRAND_FLOOR, BRAND_CEILING

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Hotel Revenue Management — Pricing Simulator",
    page_icon="🏨",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------

@st.cache_data
def load_hotel_data():
    return pd.read_csv("data/hotel_data.csv", parse_dates=["week_start_date"])


@st.cache_data
def load_elasticity():
    return pd.read_csv("models/log_log_elasticity.csv")


@st.cache_data
def load_recommendations():
    return pd.read_csv("models/rate_recommendations.csv")


@st.cache_data
def load_scenarios():
    return pd.read_csv("models/scenario_results.csv", parse_dates=["week_start_date"])


@st.cache_data
def load_scenario_summary():
    return pd.read_csv("models/scenario_summary.csv")


@st.cache_data
def load_distress():
    return pd.read_csv("models/last_minute_distress_rates.csv", parse_dates=["week_start_date"])


@st.cache_data
def load_inventory():
    return pd.read_csv("models/last_minute_inventory.csv", parse_dates=["week_start_date"])


# ---------------------------------------------------------------------------
# Tab 1 — Property Overview
# ---------------------------------------------------------------------------

def tab_overview(df: pd.DataFrame, elas_df: pd.DataFrame):
    st.header("Property Overview")

    col1, col2 = st.columns(2)

    # RevPAR heatmap: property × week
    with col1:
        st.subheader("RevPAR Heatmap (Property × Week)")
        weekly = (
            df.groupby(["property_name", pd.Grouper(key="week_start_date", freq="W")])["revpar"]
            .mean()
            .reset_index()
        )
        pivot = weekly.pivot(index="property_name", columns="week_start_date", values="revpar")
        fig = px.imshow(
            pivot.values,
            labels=dict(x="Week", y="Property", color="RevPAR (£)"),
            y=pivot.index.tolist(),
            x=[d.strftime("%Y-%m-%d") for d in pivot.columns],
            color_continuous_scale="YlOrRd",
            aspect="auto",
        )
        fig.update_layout(height=500, xaxis_nticks=20)
        st.plotly_chart(fig, use_container_width=True)

    # Elasticity table
    with col2:
        st.subheader("Price Elasticity by Property")
        prop_elas = elas_df[elas_df["segment"].str.startswith("property_")].copy()
        if not prop_elas.empty:
            prop_elas["elasticity_label"] = prop_elas["own_price_elasticity"].apply(
                lambda e: "Elastic" if abs(e) > 1 else "Inelastic"
            )
            display_cols = [
                "property_id", "property_name", "property_tier",
                "own_price_elasticity", "elasticity_ci_lower", "elasticity_ci_upper",
                "r_squared", "elasticity_label",
            ]
            display_cols = [c for c in display_cols if c in prop_elas.columns]
            st.dataframe(prop_elas[display_cols].sort_values("property_tier"), use_container_width=True)

    # Tier-level summary
    st.subheader("Elasticity by Tier")
    tier_elas = elas_df[elas_df["segment"].str.startswith("tier_")].copy()
    if not tier_elas.empty:
        fig_tier = px.bar(
            tier_elas, x="property_tier", y="own_price_elasticity",
            error_y=tier_elas["elasticity_ci_upper"] - tier_elas["own_price_elasticity"],
            error_y_minus=tier_elas["own_price_elasticity"] - tier_elas["elasticity_ci_lower"],
            color="property_tier", title="Own-Price Elasticity by Tier (with 95% CI)",
        )
        fig_tier.update_layout(yaxis_title="Elasticity (β)", showlegend=False)
        st.plotly_chart(fig_tier, use_container_width=True)


# ---------------------------------------------------------------------------
# Tab 2 — Rate Optimizer
# ---------------------------------------------------------------------------

def tab_optimizer(df: pd.DataFrame, elas_df: pd.DataFrame, recs: pd.DataFrame):
    st.header("Rate Optimizer")

    col_sel, col_chart = st.columns([1, 2])

    with col_sel:
        properties = df[["property_id", "property_name", "property_tier"]].drop_duplicates()
        selected_prop = st.selectbox(
            "Select Property",
            properties["property_id"].tolist(),
            format_func=lambda pid: f"{pid} — {properties[properties['property_id']==pid]['property_name'].iloc[0]}",
        )
        selected_lt = st.selectbox("Booking Window", ["0-7d", "8-30d", "31-90d", "90d+"])

        prop_data = df[(df["property_id"] == selected_prop) & (df["booking_lead_time_band"] == selected_lt)]
        tier = prop_data["property_tier"].iloc[0] if len(prop_data) > 0 else "midscale"
        base_adr = prop_data["adr"].mean() if len(prop_data) > 0 else 100
        base_occ = prop_data["occupancy_rate"].mean() if len(prop_data) > 0 else 0.65
        comp_rate = prop_data["competitor_avg_rate"].mean() if len(prop_data) > 0 else base_adr

        # Get elasticity
        prop_elas_row = elas_df[elas_df["segment"] == f"property_{selected_prop}"]
        elasticity = prop_elas_row["own_price_elasticity"].iloc[0] if len(prop_elas_row) > 0 else -1.2

        st.metric("Current Avg ADR", f"£{base_adr:.2f}")
        st.metric("Current Avg Occupancy", f"{base_occ:.1%}")
        st.metric("Elasticity (β)", f"{elasticity:.2f}")
        st.metric("Competitor Rate", f"£{comp_rate:.2f}")

        # Rate slider
        floor = BRAND_FLOOR[tier]
        ceiling = BRAND_CEILING[tier]
        selected_rate = st.slider(
            "Set Rate (£)", min_value=float(floor), max_value=float(ceiling),
            value=float(round(base_adr)), step=1.0,
        )

    with col_chart:
        # Demand curve
        rates = np.linspace(floor, ceiling, 300)
        occs = [_demand_curve(p, base_occ, base_adr, elasticity) for p in rates]
        revpars = [p * o for p, o in zip(rates, occs)]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=rates, y=occs, name="Occupancy", yaxis="y1",
                                 line=dict(color="royalblue", width=2)))
        fig.add_trace(go.Scatter(x=rates, y=revpars, name="RevPAR", yaxis="y2",
                                 line=dict(color="firebrick", width=2)))

        # Selected rate marker
        sel_occ = _demand_curve(selected_rate, base_occ, base_adr, elasticity)
        sel_revpar = selected_rate * sel_occ
        fig.add_trace(go.Scatter(
            x=[selected_rate], y=[sel_occ], mode="markers",
            marker=dict(size=14, color="green", symbol="diamond"),
            name=f"Selected: £{selected_rate:.0f}", yaxis="y1",
        ))

        fig.update_layout(
            title=f"Demand Curve — {selected_prop} ({selected_lt})",
            xaxis_title="ADR (£)",
            yaxis=dict(title="Occupancy Rate", side="left", range=[0, 1]),
            yaxis2=dict(title="RevPAR (£)", overlaying="y", side="right"),
            legend=dict(x=0.01, y=0.99),
            height=450,
        )
        st.plotly_chart(fig, use_container_width=True)

        # Live KPIs
        kcol1, kcol2, kcol3 = st.columns(3)
        kcol1.metric("Expected Occupancy", f"{sel_occ:.1%}",
                     delta=f"{(sel_occ - base_occ)*100:+.1f}pp")
        kcol2.metric("ADR", f"£{selected_rate:.2f}",
                     delta=f"{(selected_rate - base_adr)/base_adr*100:+.1f}%")
        kcol3.metric("RevPAR", f"£{sel_revpar:.2f}",
                     delta=f"{(sel_revpar - base_adr*base_occ)/(base_adr*base_occ)*100:+.1f}%")


# ---------------------------------------------------------------------------
# Tab 3 — Scenario Planner
# ---------------------------------------------------------------------------

def tab_scenarios(scenarios: pd.DataFrame, summary: pd.DataFrame):
    st.header("90-Day Scenario Planner")

    # Summary table
    st.subheader("Portfolio Summary — All Strategies")
    st.dataframe(summary, use_container_width=True)

    # Side-by-side RevPAR by strategy
    fig = px.bar(
        summary, x="strategy", y="mean_revpar", color="strategy",
        title="Mean RevPAR by Strategy",
        text="mean_revpar",
    )
    fig.update_layout(showlegend=False, yaxis_title="RevPAR (£)")
    st.plotly_chart(fig, use_container_width=True)

    # Per-property timeline
    st.subheader("Weekly RevPAR by Strategy")
    props = scenarios["property_id"].unique()
    selected = st.selectbox("Select Property", props, key="scenario_prop")
    prop_sc = scenarios[scenarios["property_id"] == selected]

    fig2 = px.line(
        prop_sc, x="week_start_date", y="expected_revpar",
        color="strategy", title=f"RevPAR Trajectory — {selected}",
    )
    fig2.update_layout(xaxis_title="Week", yaxis_title="RevPAR (£)")
    st.plotly_chart(fig2, use_container_width=True)

    # Total revenue comparison
    st.subheader("Total Portfolio Revenue")
    fig3 = px.bar(
        summary, x="strategy", y="total_revenue", color="strategy",
        title="Total Revenue (90-day, all properties)",
        text="total_revenue",
    )
    fig3.update_layout(showlegend=False, yaxis_title="Revenue (£)")
    st.plotly_chart(fig3, use_container_width=True)


# ---------------------------------------------------------------------------
# Tab 4 — Last-Minute Engine
# ---------------------------------------------------------------------------

def tab_last_minute(distress: pd.DataFrame, inventory: pd.DataFrame):
    st.header("Last-Minute Pricing Engine")

    if distress.empty:
        st.info("No distressed inventory detected in current data.")
        return

    # KPIs
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Distressed Property-Weeks", len(distress))
    k2.metric("Total Unsold Rooms", int(distress["unsold_rooms"].sum()))
    k3.metric("Recovered Revenue", f"£{distress['recovered_revenue'].sum():,.0f}")
    k4.metric("Avg Discount", f"{distress['rate_discount_pct'].mean():.1f}%")

    # Alert table
    st.subheader("Distress Pricing Recommendations")
    display = distress[[
        "property_id", "property_name", "property_tier", "week_start_date",
        "current_occupancy", "unsold_rooms", "current_adr", "distress_rate",
        "rate_discount_pct", "additional_rooms", "recovered_revenue", "channel_strategy",
    ]].sort_values("recovered_revenue", ascending=False)
    st.dataframe(display, use_container_width=True)

    # Chart — recovered revenue by property
    by_prop = distress.groupby("property_name")["recovered_revenue"].sum().sort_values(ascending=True)
    fig = px.bar(
        by_prop.reset_index(), x="recovered_revenue", y="property_name",
        orientation="h", title="Recovered Revenue by Property",
        labels={"recovered_revenue": "Revenue (£)", "property_name": "Property"},
    )
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Tab 5 — Competitive Positioning
# ---------------------------------------------------------------------------

def tab_competitive(df: pd.DataFrame):
    st.header("Competitive Positioning")

    # Rate index = our ADR / competitor ADR
    comp = (
        df.groupby(["property_id", "property_name", "property_tier"])
        .agg(
            our_adr=("adr", "mean"),
            comp_adr=("competitor_avg_rate", "mean"),
            our_revpar=("revpar", "mean"),
            our_occ=("occupancy_rate", "mean"),
        )
        .reset_index()
    )
    comp["rate_index"] = round(comp["our_adr"] / comp["comp_adr"], 3)
    # RGI = Revenue Generation Index = our RevPAR / fair-share RevPAR
    # Approximate comp RevPAR as comp_adr × our_occ (simplified)
    comp["comp_revpar_est"] = comp["comp_adr"] * comp["our_occ"]
    comp["rgi"] = round(comp["our_revpar"] / comp["comp_revpar_est"].clip(lower=1), 3)

    st.subheader("Rate Index & RGI by Property")
    st.dataframe(comp, use_container_width=True)

    # Scatter: Rate Index vs RGI
    fig = px.scatter(
        comp, x="rate_index", y="rgi",
        color="property_tier", size="our_revpar",
        hover_name="property_name",
        title="Rate Index vs Revenue Generation Index (RGI)",
        labels={"rate_index": "Rate Index (ADR / Comp ADR)", "rgi": "RGI (RevPAR / Comp RevPAR)"},
    )
    fig.add_hline(y=1.0, line_dash="dash", line_color="grey", annotation_text="Fair share (RGI=1)")
    fig.add_vline(x=1.0, line_dash="dash", line_color="grey", annotation_text="Parity")
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

    # Our rate vs comp rate over time
    st.subheader("Rate Trend: Our ADR vs Competitor")
    weekly = (
        df.groupby([pd.Grouper(key="week_start_date", freq="W")])
        .agg(our_adr=("adr", "mean"), comp_adr=("competitor_avg_rate", "mean"))
        .reset_index()
    )
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=weekly["week_start_date"], y=weekly["our_adr"],
                              name="Our ADR", line=dict(color="royalblue")))
    fig2.add_trace(go.Scatter(x=weekly["week_start_date"], y=weekly["comp_adr"],
                              name="Competitor Avg", line=dict(color="firebrick", dash="dash")))
    fig2.update_layout(title="Portfolio ADR vs Comp Set", yaxis_title="ADR (£)", xaxis_title="Week")
    st.plotly_chart(fig2, use_container_width=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    st.title("🏨 Hotel Revenue Management — Pricing Simulator")
    st.markdown(
        "Dynamic pricing engine powered by price-elasticity estimation and "
        "yield management optimisation across **15 properties**."
    )

    # Load data
    try:
        df = load_hotel_data()
        elas = load_elasticity()
        recs = load_recommendations()
        scenarios = load_scenarios()
        sc_summary = load_scenario_summary()
        distress = load_distress()
        inventory = load_inventory()
    except FileNotFoundError as e:
        st.error(f"Data files not found. Run `make data && make train && make optimize` first.\n\n{e}")
        return

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Property Overview",
        "💰 Rate Optimizer",
        "📈 Scenario Planner",
        "⚡ Last-Minute Engine",
        "🏁 Competitive Positioning",
    ])

    with tab1:
        tab_overview(df, elas)
    with tab2:
        tab_optimizer(df, elas, recs)
    with tab3:
        tab_scenarios(scenarios, sc_summary)
    with tab4:
        tab_last_minute(distress, inventory)
    with tab5:
        tab_competitive(df)


if __name__ == "__main__":
    main()
