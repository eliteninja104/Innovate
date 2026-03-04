"""
Southern Company Network Equipment Lifecycle Dashboard
======================================================
Interactive web dashboard for network equipment lifecycle management.
Run with: python app.py
"""

import base64
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from dash import Dash, html, dcc, callback_context, dash_table, Input, Output, State, ALL, MATCH, no_update
import dash_bootstrap_components as dbc
from sklearn.cluster import DBSCAN

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - optional dependency
    OpenAI = None

from etl_pipeline import run_pipeline, save_exceptions, load_exceptions, OUTPUT_DIR, DATA_FILE

# ---------------------------------------------------------------------------
# Initialize
# ---------------------------------------------------------------------------
app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.FLATLY, dbc.icons.FONT_AWESOME],
    suppress_callback_exceptions=True,
    title="Southern Co. Network Lifecycle Dashboard",
)
server = app.server

BASE_DIR = Path(__file__).resolve().parent
APP_FONT_STACK = "Aptos, Segoe UI, Helvetica Neue, Arial, sans-serif"

pio.templates["southern_exec"] = go.layout.Template(
    layout=go.Layout(
        font=dict(family=APP_FONT_STACK, size=13, color="#24324A"),
        title=dict(font=dict(family=APP_FONT_STACK, size=18, color="#24324A")),
        legend=dict(font=dict(family=APP_FONT_STACK, size=12)),
    )
)
pio.templates.default = "plotly_white+southern_exec"
px.defaults.template = "plotly_white+southern_exec"

# ---------------------------------------------------------------------------
# Color schemes
# ---------------------------------------------------------------------------
RISK_COLORS = {
    "Critical": "#dc3545",
    "High": "#fd7e14",
    "Medium": "#ffc107",
    "Low": "#28a745",
}

LIFECYCLE_COLORS = {
    "Past EoL": "#dc3545",
    "Past EoS": "#fd7e14",
    "EoS within 1yr": "#ffc107",
    "Current": "#28a745",
    "Unknown": "#6c757d",
}

BRAND_CHART_COLORS = [
    "#ED1D24",
    "#00BDF2",
    "#007DBA",
    "#B2D235",
    "#9CC987",
]

BRAND_CONTINUOUS_SCALE = [
    [0.0, "#9CC987"],
    [0.25, "#B2D235"],
    [0.5, "#00BDF2"],
    [0.75, "#007DBA"],
    [1.0, "#ED1D24"],
]

DEVICE_TYPE_COLORS = {
    "Switch": "#007DBA",
    "Router": "#ED1D24",
    "Access Point": "#00BDF2",
    "Wireless LAN Controller": "#B2D235",
    "Voice Gateway": "#9CC987",
}

LOGO_PATHS = [
    BASE_DIR / "WHITE LOGO.png",
    BASE_DIR / "BLACK LOGO.png",
]

RISK_FOCUS_STATUSES = ["Past EoL", "Past EoS", "EoS within 1yr", "Current"]
HOURS_PER_FTE_YEAR = 1_800
SITE_MOBILIZATION_BUDGET = 100_000
SITE_MOBILIZATION_HOURS = 24
PROGRAM_SITE_DEVICE_THRESHOLD = 8
PROGRAM_SITE_COST_THRESHOLD = 100_000


def get_logo_src():
    """Embed the local logo file so Dash can render it without an assets folder."""
    for logo_path in LOGO_PATHS:
        if logo_path.exists():
            encoded = base64.b64encode(logo_path.read_bytes()).decode("ascii")
            return f"data:image/png;base64,{encoded}"
    return None

# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------
def load_data():
    """Load processed data, running pipeline if needed."""
    csv_path = OUTPUT_DIR / "unified_devices.csv"
    if not csv_path.exists():
        run_pipeline()
    df = pd.read_csv(csv_path, low_memory=False, keep_default_na=False)
    numeric_cols = [
        "free_ports", "total_ports", "ports_in_use", "latitude", "longitude",
        "device_cost", "dna_cost", "staging_cost", "labor_cost", "material_cost",
        "tax_overhead", "de_hours", "se_hours", "fot_hours", "risk_score",
        "total_refresh_cost",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ["eos_date", "eol_date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    # Ensure risk_tier is categorical with order
    df["risk_tier"] = pd.Categorical(df["risk_tier"], categories=["Low", "Medium", "High", "Critical"], ordered=True)
    return df


DF = load_data()

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def haversine(lat1, lon1, lat2, lon2):
    """Compute haversine distance in miles between two points."""
    R = 3958.8  # Earth radius in miles
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def get_filter_options(df):
    """Extract unique filter values."""
    return {
        "states": sorted(df["state"].dropna().unique().tolist()),
        "affiliates": sorted(df["affiliate"].dropna().unique().tolist()),
        "device_types": sorted(df["device_type"].dropna().unique().tolist()),
        "lifecycle_statuses": ["Past EoL", "Past EoS", "EoS within 1yr", "Current", "Unknown"],
        "risk_tiers": ["Critical", "High", "Medium", "Low"],
    }


FILTER_OPTS = get_filter_options(DF)


def apply_filters(df, states, affiliates, device_types, lifecycle_statuses):
    """Apply dashboard filters and live exception flags."""
    filtered = df.copy()

    # Apply live exceptions from JSON (not just the CSV column)
    exceptions = load_exceptions()
    if exceptions:
        filtered["exception_flagged"] = filtered["serial_number"].isin(exceptions.keys())
    else:
        filtered["exception_flagged"] = False

    if states:
        filtered = filtered[filtered["state"].isin(states)]
    if affiliates:
        filtered = filtered[filtered["affiliate"].isin(affiliates)]
    if device_types:
        filtered = filtered[filtered["device_type"].isin(device_types)]
    if lifecycle_statuses:
        filtered = filtered[filtered["lifecycle_status"].isin(lifecycle_statuses)]
    return filtered


def apply_risk_focus(df):
    """Exclude Unknown lifecycle rows from the risk narrative per sponsor guidance."""
    return df[df["lifecycle_status"].isin(RISK_FOCUS_STATUSES)].copy()


def fmt_money(value):
    return f"${value:,.0f}"


def fmt_millions(value):
    return f"${value / 1_000_000:.1f}M"


def blank_figure(message):
    fig = go.Figure()
    fig.update_layout(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        annotations=[dict(text=message, x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)],
        margin=dict(t=10, b=10, l=10, r=10),
        height=320,
    )
    return fig


def get_filtered_frames(states, affiliates, dtypes, lifecycle):
    full_df = apply_filters(DF, states, affiliates, dtypes, lifecycle)
    full_df = full_df[full_df["exception_flagged"] != True].copy()
    risk_df = apply_risk_focus(full_df)
    return full_df, risk_df


def build_site_portfolio(risk_df):
    if risk_df.empty:
        return pd.DataFrame()

    grouped = risk_df.groupby(
        ["site_code", "site_name", "state", "affiliate", "county"],
        dropna=False,
    ).agg(
        device_count=("device_id", "count"),
        avg_risk=("risk_score", "mean"),
        total_risk=("risk_score", "sum"),
        past_eol=("lifecycle_status", lambda x: int((x == "Past EoL").sum())),
        past_eos=("lifecycle_status", lambda x: int((x == "Past EoS").sum())),
        planning_count=("lifecycle_status", lambda x: int((x == "EoS within 1yr").sum())),
        current_count=("lifecycle_status", lambda x: int((x == "Current").sum())),
        total_cost=("total_refresh_cost", "sum"),
        device_cost=("device_cost", "sum"),
        dna_cost=("dna_cost", "sum"),
        staging_cost=("staging_cost", "sum"),
        labor_cost=("labor_cost", "sum"),
        tax_overhead=("tax_overhead", "sum"),
        de_hours=("de_hours", "sum"),
        se_hours=("se_hours", "sum"),
        fot_hours=("fot_hours", "sum"),
        latitude=("latitude", "first"),
        longitude=("longitude", "first"),
    ).reset_index()

    grouped["base_hours"] = grouped["de_hours"].fillna(0) + grouped["se_hours"].fillna(0) + grouped["fot_hours"].fillna(0)
    grouped["planning_budget"] = grouped["total_cost"].fillna(0) + SITE_MOBILIZATION_BUDGET
    grouped["planning_hours"] = grouped["base_hours"] + SITE_MOBILIZATION_HOURS
    grouped["fte"] = grouped["planning_hours"] / HOURS_PER_FTE_YEAR
    grouped["dominant_phase"] = np.select(
        [
            grouped["past_eol"] > 0,
            grouped["past_eos"] > 0,
            grouped["planning_count"] > 0,
        ],
        [
            "Immediate",
            "Near-Term",
            "Planning",
        ],
        default="Strategic",
    )

    county_counts = grouped["county"].fillna("Unknown").value_counts()
    grouped["coordination_count"] = grouped["county"].fillna("Unknown").map(county_counts).fillna(1) - 1
    grouped["risk_per_dollar"] = grouped["total_risk"] / grouped["planning_budget"].replace(0, np.nan)
    grouped["risk_per_fte"] = grouped["total_risk"] / grouped["fte"].replace(0, np.nan)
    grouped = grouped.replace([np.inf, -np.inf], np.nan).fillna(0)
    return grouped


def score_program_sites(site_df, objective_mode):
    if site_df.empty:
        return site_df

    program_df = site_df[
        (site_df["device_count"] >= PROGRAM_SITE_DEVICE_THRESHOLD) |
        (site_df["total_cost"] >= PROGRAM_SITE_COST_THRESHOLD)
    ].copy()

    if program_df.empty:
        program_df = site_df.copy()

    def normalize(series):
        if series.max() == series.min():
            return pd.Series(np.ones(len(series)), index=series.index)
        return (series - series.min()) / (series.max() - series.min())

    program_df["norm_total_risk"] = normalize(program_df["total_risk"])
    program_df["norm_overdue"] = normalize(program_df["past_eol"] * 2 + program_df["past_eos"])
    program_df["norm_efficiency_dollar"] = normalize(program_df["risk_per_dollar"])
    program_df["norm_efficiency_fte"] = normalize(program_df["risk_per_fte"])
    program_df["norm_coordination"] = normalize(program_df["coordination_count"])

    weights = {
        "Balanced": {
            "norm_total_risk": 0.30,
            "norm_overdue": 0.25,
            "norm_efficiency_dollar": 0.20,
            "norm_efficiency_fte": 0.15,
            "norm_coordination": 0.10,
        },
        "Urgency First": {
            "norm_total_risk": 0.35,
            "norm_overdue": 0.35,
            "norm_efficiency_dollar": 0.10,
            "norm_efficiency_fte": 0.10,
            "norm_coordination": 0.10,
        },
        "Efficiency First": {
            "norm_total_risk": 0.20,
            "norm_overdue": 0.15,
            "norm_efficiency_dollar": 0.30,
            "norm_efficiency_fte": 0.25,
            "norm_coordination": 0.10,
        },
    }[objective_mode]

    program_df["optimizer_score"] = sum(program_df[col] * weight for col, weight in weights.items())
    program_df = program_df.sort_values(
        ["optimizer_score", "total_risk", "coordination_count"],
        ascending=False,
    ).reset_index(drop=True)
    return program_df


def explain_site_selection(row):
    reasons = []
    if row["past_eol"] > 0:
        reasons.append(f"{int(row['past_eol'])} Past EoL device(s)")
    if row["coordination_count"] > 0:
        reasons.append("same-county bundling potential")
    if row["risk_per_dollar"] >= row.get("risk_per_dollar_median", row["risk_per_dollar"]):
        reasons.append("strong risk reduction per dollar")
    if row["risk_per_fte"] >= row.get("risk_per_fte_median", row["risk_per_fte"]):
        reasons.append("efficient staffing profile")
    if not reasons:
        reasons.append("high portfolio risk concentration")
    return "; ".join(reasons[:3])


def build_optimizer_plan(risk_df, budget_cap_millions, fte_cap, objective_mode):
    site_df = build_site_portfolio(risk_df)
    if site_df.empty:
        return pd.DataFrame(), pd.DataFrame(), site_df

    scored = score_program_sites(site_df, objective_mode)
    scored["risk_per_dollar_median"] = scored["risk_per_dollar"].median()
    scored["risk_per_fte_median"] = scored["risk_per_fte"].median()

    budget_cap = budget_cap_millions * 1_000_000
    fte_limit = max(fte_cap, 0)

    selected_rows = []
    used_budget = 0.0
    used_fte = 0.0
    total_risk = max(scored["total_risk"].sum(), 1)

    for _, row in scored.iterrows():
        next_budget = used_budget + row["planning_budget"]
        next_fte = used_fte + row["fte"]
        if next_budget <= budget_cap and next_fte <= fte_limit:
            selected_rows.append(row)
            used_budget = next_budget
            used_fte = next_fte

    if not selected_rows:
        for _, row in scored.iterrows():
            if row["planning_budget"] <= budget_cap and row["fte"] <= fte_limit:
                selected_rows.append(row)
                break

    selected_df = pd.DataFrame(selected_rows)
    if selected_df.empty:
        return selected_df, pd.DataFrame(), scored

    selected_df = selected_df.copy()
    selected_df["selection_rank"] = np.arange(1, len(selected_df) + 1)
    selected_df["risk_covered"] = selected_df["total_risk"].cumsum()
    selected_df["risk_covered_pct"] = selected_df["risk_covered"] / total_risk * 100
    selected_df["planning_budget_used"] = selected_df["planning_budget"].cumsum()
    selected_df["fte_used"] = selected_df["fte"].cumsum()
    selected_df["why_selected"] = selected_df.apply(explain_site_selection, axis=1)
    return selected_df, selected_df.head(8).copy(), scored


def build_chat_summary(full_df, risk_df):
    lifecycle_counts = risk_df["lifecycle_status"].value_counts().to_dict()
    type_counts = full_df["device_type"].value_counts().head(8).to_dict()
    state_cost = risk_df.groupby("state")["total_refresh_cost"].sum().sort_values(ascending=False).head(5).round(0).to_dict()
    affiliate_cost = risk_df.groupby("affiliate")["total_refresh_cost"].sum().sort_values(ascending=False).head(5).round(0).to_dict()
    top_models = risk_df["model"].value_counts().head(8).to_dict()

    return {
        "visible_devices": int(len(full_df)),
        "risk_focus_devices": int(len(risk_df)),
        "past_eol": int((risk_df["lifecycle_status"] == "Past EoL").sum()),
        "past_eos": int((risk_df["lifecycle_status"] == "Past EoS").sum()),
        "total_hardware_cost": float(risk_df["device_cost"].sum()),
        "total_refresh_cost": float(risk_df["total_refresh_cost"].sum()),
        "total_fte": float((risk_df[["de_hours", "se_hours", "fot_hours"]].fillna(0).sum().sum()) / HOURS_PER_FTE_YEAR),
        "lifecycle_counts": lifecycle_counts,
        "device_type_counts": type_counts,
        "top_state_costs": state_cost,
        "top_affiliate_costs": affiliate_cost,
        "top_models": top_models,
    }


# ---------------------------------------------------------------------------
# Layout Components
# ---------------------------------------------------------------------------
def make_kpi_card(title, value, icon, color="primary"):
    return dbc.Card(
        dbc.CardBody([
            html.Div([
                html.Div(
                    html.I(
                        className=f"fas fa-{icon} text-{color}",
                        style={"fontSize": "1.5rem", "opacity": "0.7"},
                    ),
                    className="mb-2",
                ),
                html.H3(
                    value,
                    className=f"text-{color} mb-1 fw-bold",
                    style={
                        "fontSize": "1.6rem",
                        "lineHeight": "1.1",
                        "wordBreak": "break-word",
                    },
                ),
                html.Small(title, className="text-muted d-block"),
            ]),
        ]),
        className="shadow-sm h-100",
    )


def make_sidebar():
    logo_src = get_logo_src()
    return html.Div([
        html.Div([
            html.Button(
                html.Img(
                    src=logo_src,
                    alt="Southern Company logo",
                    style={
                        "width": "185px",
                        "height": "auto",
                        "display": "block",
                    },
                ),
                id="logo-home",
                type="button",
                style={
                    "padding": "0",
                    "border": "0",
                    "background": "transparent",
                    "display": "block",
                    "margin": "0 auto",
                    "cursor": "pointer",
                },
            ) if logo_src else None,
        ], className="text-center", style={"padding": "12px 10px", "backgroundColor": "#2F3A3D"}),

        html.Hr(className="my-0"),

        html.Div([
            html.Label("Navigation", className="fw-bold text-muted small mb-2"),
            dbc.Nav([
                dbc.NavLink([html.I(className="fas fa-shield-alt me-2"), "Network Risk Assessment"],
                            href="#", id="nav-story", active=True, className="nav-link-custom"),
                dbc.NavLink([html.I(className="fas fa-calendar-alt me-2"), "EoS / EoL Timeline"],
                            href="#", id="nav-timeline", className="nav-link-custom"),
                dbc.NavLink([html.I(className="fas fa-project-diagram me-2"), "Proximity Analysis"],
                            href="#", id="nav-proximity", className="nav-link-custom"),
                dbc.NavLink([html.I(className="fas fa-dollar-sign me-2"), "Cost & Risk Analysis"],
                            href="#", id="nav-cost", className="nav-link-custom"),
                dbc.NavLink([html.I(className="fas fa-network-wired me-2"), "Port Utilization"],
                            href="#", id="nav-capacity", className="nav-link-custom"),
                dbc.NavLink([html.I(className="fas fa-flag me-2"), "Exception Management"],
                            href="#", id="nav-exceptions", className="nav-link-custom"),
            ], vertical=True, pills=True),
        ], className="p-3"),

        html.Hr(),

        html.Div([
            html.Label("Filters", className="fw-bold text-muted small mb-2"),

            html.Label("State", className="small fw-semibold mt-2"),
            dcc.Dropdown(
                id="filter-state",
                options=[{"label": s, "value": s} for s in FILTER_OPTS["states"]],
                multi=True,
                placeholder="All States",
                className="mb-2",
            ),

            html.Label("Affiliate", className="small fw-semibold"),
            dcc.Dropdown(
                id="filter-affiliate",
                options=[{"label": a, "value": a} for a in FILTER_OPTS["affiliates"]],
                multi=True,
                placeholder="All Affiliates",
                className="mb-2",
            ),

            html.Label("Device Type", className="small fw-semibold"),
            dcc.Dropdown(
                id="filter-device-type",
                options=[{"label": d, "value": d} for d in FILTER_OPTS["device_types"]],
                multi=True,
                placeholder="All Types",
                className="mb-2",
            ),

            html.Label("Lifecycle Status", className="small fw-semibold"),
            dcc.Dropdown(
                id="filter-lifecycle",
                options=[{"label": s, "value": s} for s in FILTER_OPTS["lifecycle_statuses"]],
                multi=True,
                placeholder="All Statuses",
                className="mb-2",
            ),

            dbc.Button([html.I(className="fas fa-sync me-1"), "Refresh Data"],
                       id="btn-refresh", color="outline-primary", size="sm",
                       className="w-100 mt-3"),
        ], className="p-3"),
    ], id="sidebar", className="bg-light border-end",
       style={"width": "280px", "minHeight": "100vh", "position": "fixed",
              "overflowY": "auto", "zIndex": "1000"})


# ---------------------------------------------------------------------------
# Page Views
# ---------------------------------------------------------------------------
def story_page():
    return html.Div([
        html.H4("Network Risk Assessment", className="mb-3"),
        html.Div(id="kpi-row", className="mb-4"),
        dbc.Card(
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.H6("Estimated Cost Stack", className="mb-1"),
                        html.P(
                            "The headline KPI uses hardware-only cost. Expand below to see the full planning estimate including licensing, staging, tax, and labor.",
                            className="text-muted mb-0",
                        ),
                    ], md=9),
                    dbc.Col(
                        dbc.Button(
                            [html.I(className="fas fa-chevron-down me-1"), "See full cost estimate"],
                            id="btn-toggle-cost-breakdown",
                            color="outline-primary",
                            size="sm",
                            className="w-100",
                        ),
                        md=3,
                        className="d-flex align-items-start",
                    ),
                ], className="mb-3"),
                dbc.Collapse(html.Div(id="story-cost-breakdown"), id="collapse-cost-breakdown", is_open=False),
            ]),
            className="shadow-sm mb-4",
        ),
        dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody([
                html.H6("Support Coverage & Security Story", className="text-muted"),
                html.Div(id="support-story"),
            ]), className="shadow-sm h-100"), md=7),
            dbc.Col(dbc.Card(dbc.CardBody([
                html.H6("Data Confidence", className="text-muted"),
                html.Div(id="data-confidence"),
            ]), className="shadow-sm h-100"), md=5),
        ], className="mb-4"),
        dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody([
                html.H6("Fleet Composition", className="text-muted"),
                dcc.Graph(id="chart-device-types", config={"displayModeBar": False}),
            ]), className="shadow-sm"), md=4),
            dbc.Col(dbc.Card(dbc.CardBody([
                html.H6("Lifecycle Status", className="text-muted"),
                dcc.Graph(id="chart-lifecycle", config={"displayModeBar": False}),
            ]), className="shadow-sm"), md=4),
            dbc.Col(dbc.Card(dbc.CardBody([
                html.H6("Risk Distribution", className="text-muted"),
                dcc.Graph(id="chart-risk-dist", config={"displayModeBar": False}),
            ]), className="shadow-sm"), md=4),
        ], className="mb-4"),
        dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody([
                html.H6("Devices by State", className="text-muted"),
                dcc.Graph(id="chart-by-state", config={"displayModeBar": False}),
            ]), className="shadow-sm"), md=6),
            dbc.Col(dbc.Card(dbc.CardBody([
                html.H6("Devices by Affiliate", className="text-muted"),
                dcc.Graph(id="chart-by-affiliate", config={"displayModeBar": False}),
            ]), className="shadow-sm"), md=6),
        ], className="mb-4"),
        dbc.Card(dbc.CardBody([
            html.H6("Geographic Risk Map", className="text-muted"),
            html.P("Each bubble is a site. Size reflects device count and color reflects average risk score.", className="text-muted"),
            dcc.Graph(id="geo-map", style={"height": "60vh"}),
        ]), className="shadow-sm mb-4"),
        dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody([
                html.H6("Risk by State", className="text-muted"),
                dcc.Graph(id="state-choropleth", config={"displayModeBar": False}),
            ]), className="shadow-sm"), md=6),
            dbc.Col(dbc.Card(dbc.CardBody([
                html.H6("Top 20 Counties by Risk", className="text-muted"),
                dcc.Graph(id="county-risk-bar", config={"displayModeBar": False}),
            ]), className="shadow-sm"), md=6),
        ], className="mb-4"),
        dbc.Card(dbc.CardBody([
            html.H6("Recommended Refresh Schedule", className="text-muted"),
            html.P("A four-phase planning table that translates lifecycle urgency into an execution sequence.", className="text-muted"),
            html.Div(id="refresh-schedule-table"),
        ]), className="shadow-sm mb-4"),
        dbc.Card(dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.H5("AI Refresh Plan Optimizer", className="mb-1"),
                    html.P(
                        "Tune budget, staffing, and objective mode to generate a first-wave site package.",
                        className="text-muted small mb-0",
                    ),
                ], md=12),
                dbc.Col([
                    html.Label("Budget Cap ($M)", className="small fw-semibold"),
                    dcc.Slider(
                        id="optimizer-budget",
                        min=2,
                        max=50,
                        step=1,
                        value=12,
                        marks={2: "2", 12: "12", 25: "25", 50: "50"},
                        tooltip={"placement": "bottom", "always_visible": False},
                    ),
                ], md=5, className="optimizer-slider-col"),
                dbc.Col([
                    html.Label("FTE Cap", className="small fw-semibold"),
                    dcc.Slider(
                        id="optimizer-fte",
                        min=2,
                        max=50,
                        step=1,
                        value=10,
                        marks={2: "2", 10: "10", 25: "25", 50: "50"},
                        tooltip={"placement": "bottom", "always_visible": False},
                    ),
                ], md=5, className="optimizer-slider-col"),
                dbc.Col([
                    html.Label("Objective Mode", className="small fw-semibold"),
                    dcc.Dropdown(
                        id="optimizer-objective",
                        options=[
                            {"label": "Balanced", "value": "Balanced"},
                            {"label": "Urgency First", "value": "Urgency First"},
                            {"label": "Efficiency First", "value": "Efficiency First"},
                        ],
                        value="Balanced",
                        clearable=False,
                    ),
                ], md=2),
            ], className="gy-4 align-items-end"),
            html.Hr(),
            html.Div(id="optimizer-kpis", className="mb-4"),
            dbc.Row([
                dbc.Col(dbc.Card(dbc.CardBody([
                    html.H6("How the Plan Builds", className="text-muted"),
                    dcc.Graph(id="optimizer-build-chart", config={"displayModeBar": False}),
                ]), className="border-0"), md=7),
                dbc.Col(dbc.Card(dbc.CardBody([
                    html.H6("Executive Decision Brief", className="text-muted"),
                    html.Div(id="optimizer-exec-brief"),
                ]), className="border-0 h-100"), md=5),
            ], className="mb-3"),
            dbc.Row([
                dbc.Col(dbc.Card(dbc.CardBody([
                    html.H6("Suggested First Wave", className="text-muted"),
                    html.Div(id="optimizer-first-wave"),
                ]), className="border-0"), md=8),
                dbc.Col(dbc.Card(dbc.CardBody([
                    html.H6("Final Recommendation", className="text-muted"),
                    html.Div(id="final-recommendation"),
                ]), className="border-0 h-100"), md=4),
            ], className="mb-3"),
            html.Div(id="optimizer-full-portfolio"),
        ]), className="shadow-sm mb-4"),
        dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody([
                html.H6("Top Risk Sites", className="text-muted"),
                dcc.Graph(id="priority-sites", config={"displayModeBar": False}),
            ]), className="shadow-sm"), md=6),
            dbc.Col(dbc.Card(dbc.CardBody([
                html.H6("Priority Models for Refresh", className="text-muted"),
                dcc.Graph(id="priority-models", config={"displayModeBar": False}),
            ]), className="shadow-sm"), md=6),
        ], className="mb-4"),
        html.Div(id="staffing-kpis", className="mb-4"),
        dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody([
                html.H6("Top 15 Sites by Engineering Hours", className="text-muted"),
                dcc.Graph(id="staffing-by-site", config={"displayModeBar": False}),
            ]), className="shadow-sm"), md=6),
            dbc.Col(dbc.Card(dbc.CardBody([
                html.H6("Refresh Cost by Affiliate", className="text-muted"),
                dcc.Graph(id="cost-by-affiliate", config={"displayModeBar": False}),
            ]), className="shadow-sm"), md=6),
        ], className="mb-4"),
        html.Details([
            html.Summary("Open detailed cost and staffing appendix", className="fw-semibold"),
            html.Div([
                dbc.Row([
                    dbc.Col(dbc.Card(dbc.CardBody([
                        html.H6("Total Refresh Cost by Lifecycle Status", className="text-muted"),
                        dcc.Graph(id="cost-by-lifecycle"),
                    ]), className="shadow-sm"), md=6),
                    dbc.Col(dbc.Card(dbc.CardBody([
                        html.H6("Average Cost per Device by Type", className="text-muted"),
                        dcc.Graph(id="cost-by-type"),
                    ]), className="shadow-sm"), md=6),
                ], className="mb-4 mt-3"),
                dbc.Row([
                    dbc.Col(dbc.Card(dbc.CardBody([
                        html.H6("Risk Score vs Refresh Cost", className="text-muted"),
                        dcc.Graph(id="risk-vs-cost-scatter"),
                    ]), className="shadow-sm"), md=6),
                    dbc.Col(dbc.Card(dbc.CardBody([
                        html.H6("Cumulative Refresh Investment Required", className="text-muted"),
                        dcc.Graph(id="cumulative-cost"),
                    ]), className="shadow-sm"), md=6),
                ], className="mb-4"),
                dbc.Row([
                    dbc.Col(dbc.Card(dbc.CardBody([
                        html.H6("Engineering Hours by Refresh Phase", className="text-muted"),
                        dcc.Graph(id="staffing-by-phase"),
                    ]), className="shadow-sm"), md=6),
                    dbc.Col(dbc.Card(dbc.CardBody([
                        html.H6("FTE Projection by Refresh Phase", className="text-muted"),
                        html.Div(id="staffing-fte-table"),
                    ]), className="shadow-sm"), md=6),
                ], className="mb-4"),
                dbc.Row([
                    dbc.Col(dbc.Card(dbc.CardBody([
                        html.H6("Top 15 Models by Count", className="text-muted"),
                        dcc.Graph(id="chart-top-models", config={"displayModeBar": False}),
                    ]), className="shadow-sm"), md=12),
                ]),
            ], className="pt-3"),
        ], className="mb-4"),
    ])


def map_page():
    return html.Div([
        html.H4("Geographic Risk Map", className="mb-3"),
        html.P("Device risk by location. Bubble size = device count, color = average risk score.",
               className="text-muted"),
        dbc.Card(dbc.CardBody([
            dcc.Graph(id="geo-map", style={"height": "70vh"}),
        ]), className="shadow-sm mb-4"),
        dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody([
                html.H6("Risk by State (Choropleth)", className="text-muted"),
                dcc.Graph(id="state-choropleth", config={"displayModeBar": False}),
            ]), className="shadow-sm"), md=6),
            dbc.Col(dbc.Card(dbc.CardBody([
                html.H6("Risk Heatmap by County", className="text-muted"),
                dcc.Graph(id="county-risk-bar", config={"displayModeBar": False}),
            ]), className="shadow-sm"), md=6),
        ]),
    ])


def timeline_page():
    return html.Div([
        html.H4("EoS / EoL Timeline", className="mb-3"),
        html.P("Upcoming lifecycle milestones over time.", className="text-muted"),
        dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody([
                html.H6("End-of-Sale Timeline", className="text-muted"),
                dcc.Graph(id="eos-timeline", config={"displayModeBar": False}),
            ]), className="shadow-sm"), md=12),
        ], className="mb-4"),
        dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody([
                html.H6("End-of-Life Timeline", className="text-muted"),
                dcc.Graph(id="eol-timeline", config={"displayModeBar": False}),
            ]), className="shadow-sm"), md=12),
        ], className="mb-4"),
        dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody([
                html.H6("Overdue Devices — Past EoL Still in Production", className="text-muted"),
                html.Div(id="overdue-table-container"),
            ]), className="shadow-sm"), md=12),
        ]),
    ])


def proximity_page():
    return html.Div([
        html.H4("Proximity-Based Refresh Planning", className="mb-3"),
        html.P("Identify sites within a specified radius for coordinated refresh projects.",
               className="text-muted"),
        dbc.Row([
            dbc.Col([
                html.Label("Cluster Radius (miles)", className="fw-semibold"),
                dcc.Slider(id="proximity-radius", min=1, max=25, step=1, value=5,
                           marks={1: "1mi", 5: "5mi", 10: "10mi", 15: "15mi", 25: "25mi"},
                           tooltip={"placement": "bottom", "always_visible": True}),
            ], md=6),
            dbc.Col([
                html.Label("Min Sites per Cluster", className="fw-semibold"),
                dcc.Slider(id="proximity-min-sites", min=2, max=10, step=1, value=2,
                           marks={2: "2", 5: "5", 10: "10"},
                           tooltip={"placement": "bottom", "always_visible": True}),
            ], md=6),
        ], className="mb-4"),
        dbc.Card(dbc.CardBody([
            dcc.Graph(id="proximity-map", style={"height": "60vh"}),
        ]), className="shadow-sm mb-4"),
        dbc.Card(dbc.CardBody([
            html.H6("Cluster Details", className="text-muted"),
            html.Div(id="cluster-details"),
        ]), className="shadow-sm"),
    ])


def cost_page():
    return html.Div([
        html.H4("Cost & Risk Analysis", className="mb-3"),
        html.P("How lifecycle status correlates with support coverage, security risk, and cost.",
               className="text-muted"),
        dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody([
                html.H6("Total Refresh Cost by Lifecycle Status", className="text-muted"),
                dcc.Graph(id="cost-by-lifecycle"),
            ]), className="shadow-sm"), md=6),
            dbc.Col(dbc.Card(dbc.CardBody([
                html.H6("Average Cost per Device by Type", className="text-muted"),
                dcc.Graph(id="cost-by-type"),
            ]), className="shadow-sm"), md=6),
        ], className="mb-4"),
        dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody([
                html.H6("Risk Score vs Refresh Cost", className="text-muted"),
                dcc.Graph(id="risk-vs-cost-scatter"),
            ]), className="shadow-sm"), md=6),
            dbc.Col(dbc.Card(dbc.CardBody([
                html.H6("Cost Distribution by Affiliate", className="text-muted"),
                dcc.Graph(id="cost-by-affiliate"),
            ]), className="shadow-sm"), md=6),
        ], className="mb-4"),
        dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody([
                html.H6("Cumulative Refresh Investment Required", className="text-muted"),
                dcc.Graph(id="cumulative-cost"),
            ]), className="shadow-sm"), md=12),
        ]),
    ])


def capacity_page():
    return html.Div([
        html.H4("Port Utilization & Capacity", className="mb-3"),
        html.P("Network port capacity utilization across switches, routers, and voice gateways.",
               className="text-muted"),
        html.Div(id="capacity-kpi-row"),
        dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody([
                html.H6("Utilization Distribution", className="text-muted"),
                dcc.Graph(id="capacity-histogram", config={"displayModeBar": False}),
            ]), className="shadow-sm"), md=6),
            dbc.Col(dbc.Card(dbc.CardBody([
                html.H6("Utilization by Device Type", className="text-muted"),
                dcc.Graph(id="capacity-by-type", config={"displayModeBar": False}),
            ]), className="shadow-sm"), md=6),
        ], className="mb-4"),
        dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody([
                html.H6("Top 20 Sites by Avg Port Utilization", className="text-muted"),
                dcc.Graph(id="capacity-by-site", config={"displayModeBar": False}),
            ]), className="shadow-sm"), md=12),
        ], className="mb-4"),
        dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody([
                html.H6("High-Utilization Devices (>90%)", className="text-muted"),
                html.Div(id="capacity-table"),
            ]), className="shadow-sm"), md=12),
        ]),
    ])


def exceptions_page():
    return html.Div([
        html.H4("Exception Management", className="mb-3"),
        html.P("Flag active devices that should be excluded from project scope. "
               "Flagged devices are persisted and respected in all reports.",
               className="text-muted"),
        dbc.Row([
            dbc.Col([
                html.Label("Search by Hostname or Serial Number", className="fw-semibold"),
                dbc.InputGroup([
                    dbc.Input(id="exception-search", placeholder="Enter hostname or serial..."),
                    dbc.Button("Search", id="exception-search-btn", color="primary"),
                ]),
            ], md=6),
            dbc.Col([
                html.Div(id="exception-count-badge", className="mt-4"),
            ], md=6),
        ], className="mb-3"),
        html.Div(id="exception-search-results"),
        html.Hr(),
        html.H6("Currently Flagged Exceptions", className="text-muted mt-3"),
        html.Div(id="exception-list"),
    ])


def priorities_page():
    return html.Div([
        html.H4("Refresh Planning & Prioritization", className="mb-3"),
        html.P("Sites and models prioritized by risk exposure and cost-efficiency.",
               className="text-muted"),
        dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody([
                html.H6("Top 20 Highest Risk Sites", className="text-muted"),
                dcc.Graph(id="priority-sites"),
            ]), className="shadow-sm"), md=6),
            dbc.Col(dbc.Card(dbc.CardBody([
                html.H6("Priority Models for Refresh", className="text-muted"),
                dcc.Graph(id="priority-models"),
            ]), className="shadow-sm"), md=6),
        ], className="mb-4"),
        dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody([
                html.H6("Recommended Refresh Schedule", className="text-muted"),
                html.Div(id="refresh-schedule-table"),
            ]), className="shadow-sm"), md=12),
        ], className="mb-4"),

        # ---- Staffing & Resource Planning Section ----
        html.Hr(className="my-4"),
        html.H4("Staffing & Resource Planning", className="mb-1"),
        html.P("Engineering hours required for each refresh phase, based on ModelData staffing estimates.",
               className="text-muted mb-3"),
        html.Div(id="staffing-kpis", className="mb-4"),
        dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody([
                html.H6("Engineering Hours by Refresh Phase", className="text-muted"),
                dcc.Graph(id="staffing-by-phase"),
            ]), className="shadow-sm"), md=6),
            dbc.Col(dbc.Card(dbc.CardBody([
                html.H6("Top 15 Sites by Engineering Hours Required", className="text-muted"),
                dcc.Graph(id="staffing-by-site"),
            ]), className="shadow-sm"), md=6),
        ], className="mb-4"),
        dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody([
                html.H6("FTE Projection by Refresh Phase", className="text-muted"),
                html.Div(id="staffing-fte-table"),
            ]), className="shadow-sm"), md=12),
        ]),
    ])


def make_chat_widget():
    return html.Div([
        dcc.Store(id="chat-open", data=False),
        dcc.Store(id="chat-history", data=[
            {"role": "assistant", "content": "Ask about the currently filtered fleet. I answer from aggregated dashboard statistics only."}
        ]),
        html.Button(
            [html.I(className="fas fa-comment-dots me-2"), "Fleet Assistant"],
            id="chat-toggle-btn",
            className="btn btn-primary shadow",
            style={
                "position": "fixed",
                "bottom": "22px",
                "right": "22px",
                "zIndex": "1100",
                "borderRadius": "999px",
                "padding": "10px 16px",
            },
        ),
        dbc.Collapse(
            dbc.Card([
                dbc.CardHeader(
                    dbc.Row([
                        dbc.Col(html.Strong("Fleet Assistant")),
                        dbc.Col(
                            html.Button(
                                html.I(className="fas fa-times"),
                                id="chat-close-btn",
                                className="btn btn-sm btn-link text-decoration-none p-0",
                            ),
                            width="auto",
                        ),
                    ], className="align-items-center"),
                ),
                dbc.CardBody([
                    html.Div(
                        id="chat-messages",
                        style={"height": "280px", "overflowY": "auto", "fontSize": "0.95rem"},
                    ),
                    dbc.InputGroup([
                        dbc.Input(
                            id="chat-input",
                            placeholder="Ask a question about the filtered fleet...",
                            debounce=True,
                            n_submit=0,
                        ),
                        dbc.Button("Send", id="chat-send-btn", color="primary"),
                    ], className="mt-3"),
                    html.Small(
                        "Only aggregated summary statistics are sent to the model. No hostnames, IPs, or serial numbers are shared.",
                        className="text-muted d-block mt-2",
                    ),
                ]),
            ], className="shadow"),
            id="chat-collapse",
            is_open=False,
            style={
                "position": "fixed",
                "bottom": "76px",
                "right": "22px",
                "width": "360px",
                "maxWidth": "calc(100vw - 30px)",
                "zIndex": "1099",
            },
        ),
    ])


# ---------------------------------------------------------------------------
# Main Layout
# ---------------------------------------------------------------------------
app.layout = html.Div([
    dcc.Store(id="current-page", data="story"),
    dcc.Store(id="filtered-data-signal", data=0),
    dcc.Store(id="exceptions-signal", data=0),
    dcc.Download(id="download-csv"),
    make_sidebar(),
    html.Div([
        html.Div([
            html.Div([
                html.H6("Southern Company — Network Services Lifecycle Analytics",
                         className="text-muted mb-0"),
                html.Small(id="filter-summary", className="text-muted"),
            ]),
            dbc.Button(
                [html.I(className="fas fa-download me-1"), "Export CSV"],
                id="btn-export-csv",
                color="outline-secondary",
                size="sm",
            ),
        ], className="d-flex justify-content-between align-items-center p-3 bg-white border-bottom"),
        html.Div(id="page-content", className="p-4"),
    ], style={"marginLeft": "280px"}),
    make_chat_widget(),
])


# ---------------------------------------------------------------------------
# Callbacks: Navigation
# ---------------------------------------------------------------------------
NAV_IDS = ["nav-story", "nav-timeline", "nav-proximity", "nav-cost", "nav-capacity", "nav-exceptions"]
PAGE_MAP = {
    "logo-home": "story",
    "nav-story": "story",
    "nav-timeline": "timeline",
    "nav-proximity": "proximity",
    "nav-cost": "cost",
    "nav-capacity": "capacity",
    "nav-exceptions": "exceptions",
}


@app.callback(
    Output("current-page", "data"),
    [Input("logo-home", "n_clicks")] + [Input(nid, "n_clicks") for nid in NAV_IDS],
    prevent_initial_call=True,
)
def switch_page(*args):
    ctx = callback_context
    if not ctx.triggered:
        return "story"
    triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]
    return PAGE_MAP.get(triggered_id, "story")


@app.callback(
    [Output(nid, "active") for nid in NAV_IDS],
    Input("current-page", "data"),
)
def update_nav_active(page):
    return [PAGE_MAP.get(nid) == page for nid in NAV_IDS]


@app.callback(
    Output("page-content", "children"),
    Input("current-page", "data"),
)
def render_page(page):
    pages = {
        "story": story_page,
        "timeline": timeline_page,
        "proximity": proximity_page,
        "cost": cost_page,
        "capacity": capacity_page,
        "exceptions": exceptions_page,
    }
    return pages.get(page, story_page)()


# ---------------------------------------------------------------------------
# Callbacks: Data refresh
# ---------------------------------------------------------------------------
@app.callback(
    Output("filtered-data-signal", "data"),
    Input("btn-refresh", "n_clicks"),
    prevent_initial_call=True,
)
def refresh_data(n):
    global DF, FILTER_OPTS
    run_pipeline()
    DF = load_data()
    FILTER_OPTS = get_filter_options(DF)
    return (n or 0)


@app.callback(
    Output("filter-summary", "children"),
    [Input("filter-state", "value"),
     Input("filter-affiliate", "value"),
     Input("filter-device-type", "value"),
     Input("filter-lifecycle", "value"),
     Input("exceptions-signal", "data")],
)
def update_filter_summary(states, affiliates, dtypes, lifecycle, _exc_signal):
    df = apply_filters(DF, states, affiliates, dtypes, lifecycle)
    active = df[~df["exception_flagged"]]
    excepted = df["exception_flagged"].sum()
    filter_count = sum(bool(x) for x in [states, affiliates, dtypes, lifecycle])
    parts = [f"{len(active):,} visible devices", f"{excepted} excepted", f"{filter_count} active filter(s)"]
    return " | ".join(parts)


# ---------------------------------------------------------------------------
# Callbacks: CSV Export
# ---------------------------------------------------------------------------
EXPORT_COLUMNS = [
    "hostname", "serial_number", "ip_address", "model", "device_type",
    "state", "site_code", "site_name", "affiliate", "county",
    "lifecycle_status", "eos_date", "eol_date", "risk_score", "risk_tier",
    "in_scope", "replacement_device",
    "total_refresh_cost", "device_cost", "labor_cost", "material_cost",
    "total_ports", "ports_in_use", "free_ports",
    "de_hours", "se_hours", "fot_hours",
    "exception_flagged", "source",
]


@app.callback(
    Output("download-csv", "data"),
    Input("btn-export-csv", "n_clicks"),
    [State("filter-state", "value"),
     State("filter-affiliate", "value"),
     State("filter-device-type", "value"),
     State("filter-lifecycle", "value")],
    prevent_initial_call=True,
)
def export_csv(n, states, affiliates, dtypes, lifecycle):
    df = apply_filters(DF, states, affiliates, dtypes, lifecycle)
    df = df[df["exception_flagged"] != True]

    # Select only columns that exist (handles edge cases)
    cols = [c for c in EXPORT_COLUMNS if c in df.columns]
    export = df[cols].copy()

    # Format dates as strings for clean CSV output
    for col in ["eos_date", "eol_date"]:
        if col in export.columns:
            export[col] = export[col].dt.strftime("%Y-%m-%d")

    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    return dcc.send_data_frame(export.to_csv, f"soco_lifecycle_export_{timestamp}.csv", index=False)


# ---------------------------------------------------------------------------
# Callbacks: Overview
# ---------------------------------------------------------------------------
@app.callback(
    [Output("kpi-row", "children"),
     Output("chart-device-types", "figure"),
     Output("chart-lifecycle", "figure"),
     Output("chart-risk-dist", "figure"),
     Output("chart-by-state", "figure"),
     Output("chart-by-affiliate", "figure"),
     Output("chart-top-models", "figure")],
    [Input("current-page", "data"),
     Input("filter-state", "value"),
     Input("filter-affiliate", "value"),
     Input("filter-device-type", "value"),
     Input("filter-lifecycle", "value"),
     Input("filtered-data-signal", "data"),
     Input("exceptions-signal", "data")],
)
def update_overview(page, states, affiliates, dtypes, lifecycle, _signal, _exc_signal):
    if page != "story":
        return [no_update] * 7

    full_df, risk_df = get_filtered_frames(states, affiliates, dtypes, lifecycle)

    at_risk = len(risk_df)
    past_eol = int((risk_df["lifecycle_status"] == "Past EoL").sum())
    past_eos = int((risk_df["lifecycle_status"] == "Past EoS").sum())
    hardware_cost = risk_df["device_cost"].sum()

    kpi_row = dbc.Row([
        dbc.Col(make_kpi_card("At-Risk Devices", f"{at_risk:,}", "server", "primary"), md=3),
        dbc.Col(make_kpi_card("Past End-of-Life", f"{past_eol:,}", "exclamation-triangle", "danger"), md=3),
        dbc.Col(make_kpi_card("Past End-of-Sale", f"{past_eos:,}", "clock", "warning"), md=3),
        dbc.Col(make_kpi_card("Est. Hardware Cost", fmt_money(hardware_cost), "microchip", "info"), md=3),
    ], className="g-3 mb-4")

    # Device types pie uses the full fleet, including Unknown lifecycle rows.
    dt_counts = full_df["device_type"].value_counts()
    fig_dt = px.pie(values=dt_counts.values, names=dt_counts.index,
                    color=dt_counts.index,
                    color_discrete_map=DEVICE_TYPE_COLORS,
                    hole=0.4)
    fig_dt.update_traces(
        marker=dict(line=dict(color="#003087", width=1)),
        hovertemplate="<b>%{label}</b><br>%{value:,} devices<br>%{percent}<extra></extra>"
    )
    fig_dt.update_layout(
        margin=dict(t=10, b=10, l=10, r=10),
        height=300,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.12,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(0,0,0,0)",
        ),
    )

    # Lifecycle pie uses only the risk-focused population.
    lc_counts = risk_df["lifecycle_status"].value_counts()
    fig_lc = px.pie(values=lc_counts.values, names=lc_counts.index,
                    color=lc_counts.index,
                    color_discrete_map=LIFECYCLE_COLORS,
                    hole=0.4)
    fig_lc.update_traces(
        hovertemplate="<b>%{label}</b><br>%{value:,} devices<br>%{percent}<extra></extra>"
    )
    fig_lc.update_layout(
        margin=dict(t=10, b=10, l=10, r=10),
        height=300,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.12,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(0,0,0,0)",
        ),
    )

    # Risk distribution
    risk_counts = risk_df["risk_tier"].value_counts().reindex(["Low", "Medium", "High", "Critical"], fill_value=0)
    fig_risk = px.bar(x=risk_counts.index, y=risk_counts.values,
                      color=risk_counts.index,
                      color_discrete_map=RISK_COLORS,
                      labels={"x": "Risk Tier", "y": "Device Count"})
    fig_risk.update_traces(
        marker_line_color="#003087",
        marker_line_width=1,
        hovertemplate="<b>%{x}</b><br>%{y:,} devices<extra></extra>"
    )
    fig_risk.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=300, showlegend=False)

    # By state
    state_counts = risk_df.groupby("state").agg(
        count=("device_id", "count"),
        avg_risk=("risk_score", "mean")
    ).sort_values("count", ascending=True).tail(15)
    fig_state = go.Figure(go.Bar(
        x=state_counts["count"],
        y=state_counts.index,
        orientation="h",
        marker=dict(color=state_counts["avg_risk"], colorscale=BRAND_CONTINUOUS_SCALE,
                    colorbar=dict(title="Avg Risk")),
        hovertemplate="<b>%{y}</b><br>Devices: %{x:,}<br>Avg Risk: %{marker.color:.1f}<extra></extra>",
    ))
    fig_state.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=400,
                            xaxis_title="Devices", yaxis_title="State")

    # By affiliate
    aff_counts = risk_df.groupby("affiliate").agg(
        count=("device_id", "count"),
        avg_risk=("risk_score", "mean")
    ).sort_values("count", ascending=True)
    fig_aff = go.Figure(go.Bar(
        x=aff_counts["count"],
        y=aff_counts.index,
        orientation="h",
        marker=dict(color=aff_counts["avg_risk"], colorscale=BRAND_CONTINUOUS_SCALE,
                    colorbar=dict(title="Avg Risk")),
        hovertemplate="<b>%{y}</b><br>Devices: %{x:,}<br>Avg Risk: %{marker.color:.1f}<extra></extra>",
    ))
    fig_aff.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=400,
                          xaxis_title="Devices", yaxis_title="Affiliate")

    # Top models
    model_counts = risk_df["model"].value_counts().head(15)
    model_lifecycle = risk_df[risk_df["model"].isin(model_counts.index)].groupby("model")["lifecycle_status"].apply(
        lambda x: x.mode()[0] if len(x) > 0 else "Unknown"
    )
    model_colors = [model_lifecycle.get(m, "Unknown") for m in model_counts.index]
    fig_models = go.Figure()
    for status in LIFECYCLE_COLORS:
        mask = [c == status for c in model_colors]
        if any(mask):
            vals = [v for v, m in zip(model_counts.values, mask) if m]
            names = [n for n, m in zip(model_counts.index, mask) if m]
            fig_models.add_trace(go.Bar(
                x=vals, y=names, orientation="h",
                name=status, marker_color=LIFECYCLE_COLORS[status], marker_line=dict(color="#003087", width=1),
                hovertemplate="<b>%{y}</b><br>Count: %{x:,}<br>Status: " + status + "<extra></extra>",
            ))
    fig_models.update_layout(margin=dict(t=10, b=30, l=10, r=10), height=400,
                             yaxis=dict(autorange="reversed"), barmode="stack")

    return kpi_row, fig_dt, fig_lc, fig_risk, fig_state, fig_aff, fig_models


# ---------------------------------------------------------------------------
# Callbacks: Geographic Map
# ---------------------------------------------------------------------------
@app.callback(
    [Output("geo-map", "figure"),
     Output("state-choropleth", "figure"),
     Output("county-risk-bar", "figure")],
    [Input("current-page", "data"),
     Input("filter-state", "value"),
     Input("filter-affiliate", "value"),
     Input("filter-device-type", "value"),
     Input("filter-lifecycle", "value"),
     Input("filtered-data-signal", "data"),
     Input("exceptions-signal", "data")],
)
def update_map(page, states, affiliates, dtypes, lifecycle, _signal, _exc_signal):
    if page != "story":
        return [no_update] * 3

    _, df = get_filtered_frames(states, affiliates, dtypes, lifecycle)
    geo = df[df["latitude"].notna() & df["longitude"].notna()].copy()

    # Aggregate by site
    site_agg = geo.groupby(["site_code", "latitude", "longitude", "site_name", "state", "county"]).agg(
        device_count=("device_id", "count"),
        avg_risk=("risk_score", "mean"),
        past_eol=("lifecycle_status", lambda x: (x == "Past EoL").sum()),
        past_eos=("lifecycle_status", lambda x: (x == "Past EoS").sum()),
        total_cost=("total_refresh_cost", "sum"),
    ).reset_index()
    site_agg = site_agg[site_agg["device_count"] >= 3].copy()

    if site_agg.empty:
        return (
            blank_figure("No geocoded at-risk sites for the current filters."),
            blank_figure("No state risk data for the current filters."),
            blank_figure("No county risk data for the current filters."),
        )

    # Build clean hover text for map
    site_agg["hover_text"] = site_agg.apply(
        lambda r: (
            f"<b>{r['site_name']}</b><br>"
            f"State: {r['state']} | County: {r['county']}<br>"
            f"Devices: {int(r['device_count']):,}<br>"
            f"Avg Risk Score: {r['avg_risk']:.1f}<br>"
            f"Past EoL: {int(r['past_eol'])} | Past EoS: {int(r['past_eos'])}<br>"
            f"Refresh Cost: ${r['total_cost']:,.0f}"
        ), axis=1
    )

    fig_map = px.scatter_map(
        site_agg,
        lat="latitude",
        lon="longitude",
        size="device_count",
        color="avg_risk",
        color_continuous_scale=BRAND_CONTINUOUS_SCALE,
        size_max=25,
        zoom=5,
        center={"lat": 33.0, "lon": -85.0},
    )
    fig_map.update_traces(
        hovertemplate="%{customdata[0]}<extra></extra>",
        customdata=site_agg[["hover_text"]].values,
    )
    fig_map.update_layout(
        margin=dict(t=0, b=0, l=0, r=0),
        map_style="carto-positron",
    )

    # State choropleth
    state_risk = df.groupby("state").agg(
        avg_risk=("risk_score", "mean"),
        count=("device_id", "count"),
    ).reset_index()
    fig_choro = px.choropleth(
        state_risk, locations="state", locationmode="USA-states",
        color="avg_risk", color_continuous_scale=BRAND_CONTINUOUS_SCALE,
        scope="usa",
        labels={"avg_risk": "Avg Risk Score"},
    )
    fig_choro.update_traces(
        hovertemplate="<b>%{location}</b><br>Devices: %{customdata[0]:,}<br>Avg Risk: %{z:.1f}<extra></extra>",
        customdata=state_risk[["count"]].values,
    )
    fig_choro.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=400)

    # County risk bar
    county_risk = df[df["county"].notna()].groupby("county").agg(
        avg_risk=("risk_score", "mean"),
        count=("device_id", "count"),
    ).reset_index()
    county_risk = county_risk[county_risk["count"] >= 5].sort_values("avg_risk", ascending=False).head(20)
    fig_county = go.Figure(go.Bar(
        x=county_risk["avg_risk"],
        y=county_risk["county"],
        orientation="h",
        marker=dict(color=county_risk["avg_risk"], colorscale=BRAND_CONTINUOUS_SCALE),
        customdata=county_risk[["count"]].values,
        hovertemplate="<b>%{y}</b><br>Avg Risk: %{x:.1f}<br>Devices: %{customdata[0]:,}<extra></extra>",
    ))
    fig_county.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=400,
                             xaxis_title="Avg Risk Score", yaxis_title="County",
                             yaxis=dict(autorange="reversed"))

    return fig_map, fig_choro, fig_county


# ---------------------------------------------------------------------------
# Callbacks: Timeline
# ---------------------------------------------------------------------------
@app.callback(
    [Output("eos-timeline", "figure"),
     Output("eol-timeline", "figure"),
     Output("overdue-table-container", "children")],
    [Input("current-page", "data"),
     Input("filter-state", "value"),
     Input("filter-affiliate", "value"),
     Input("filter-device-type", "value"),
     Input("filter-lifecycle", "value"),
     Input("filtered-data-signal", "data"),
     Input("exceptions-signal", "data")],
)
def update_timeline(page, states, affiliates, dtypes, lifecycle, _signal, _exc_signal):
    if page != "timeline":
        return [no_update] * 3

    _, df = get_filtered_frames(states, affiliates, dtypes, lifecycle)

    today = pd.Timestamp.now().normalize()

    # EoS timeline
    eos_df = df[df["eos_date"].notna()].copy()
    eos_by_date = eos_df.groupby([pd.Grouper(key="eos_date", freq="QE"), "device_type"]).size().reset_index(name="count")
    fig_eos = px.bar(eos_by_date, x="eos_date", y="count", color="device_type",
                     color_discrete_map=DEVICE_TYPE_COLORS,
                     labels={
                         "eos_date": "End-of-Sale Date",
                         "count": "Devices",
                         "device_type": "Device Type",
                     })
    fig_eos.update_traces(
        marker_line_color="#003087",
        marker_line_width=1,
        hovertemplate="<b>%{fullData.name}</b><br>Quarter: %{x|%b %Y}<br>Devices: %{y:,}<extra></extra>"
    )
    today_str = today.strftime("%Y-%m-%d")
    fig_eos.add_vline(x=today_str, line_dash="dash", line_color="red")
    fig_eos.add_annotation(
        x=today_str,
        y=0.98,
        yref="paper",
        text="Today",
        showarrow=False,
        xanchor="left",
        xshift=12,
        yshift=0,
        font=dict(color="red"),
    )
    fig_eos.update_layout(
        margin=dict(t=44, b=72, l=16, r=16),
        height=350,
        barmode="stack",
        legend=dict(
            title_text="Device Type",
            orientation="h",
            yanchor="top",
            y=-0.22,
            xanchor="left",
            x=0,
        ),
    )

    # EoL timeline
    eol_df = df[df["eol_date"].notna()].copy()
    eol_by_date = eol_df.groupby([pd.Grouper(key="eol_date", freq="QE"), "device_type"]).size().reset_index(name="count")
    fig_eol = px.bar(eol_by_date, x="eol_date", y="count", color="device_type",
                     color_discrete_map=DEVICE_TYPE_COLORS,
                     labels={
                         "eol_date": "End-of-Life Date",
                         "count": "Devices",
                         "device_type": "Device Type",
                     })
    fig_eol.update_traces(
        marker_line_color="#003087",
        marker_line_width=1,
        hovertemplate="<b>%{fullData.name}</b><br>Quarter: %{x|%b %Y}<br>Devices: %{y:,}<extra></extra>"
    )
    fig_eol.add_vline(x=today_str, line_dash="dash", line_color="red")
    fig_eol.add_annotation(
        x=today_str,
        y=0.98,
        yref="paper",
        text="Today",
        showarrow=False,
        xanchor="left",
        xshift=12,
        yshift=0,
        font=dict(color="red"),
    )
    fig_eol.update_layout(
        margin=dict(t=44, b=72, l=16, r=16),
        height=350,
        barmode="stack",
        legend=dict(
            title_text="Device Type",
            orientation="h",
            yanchor="top",
            y=-0.22,
            xanchor="left",
            x=0,
        ),
    )

    # Overdue devices table
    overdue = df[(df["lifecycle_status"] == "Past EoL")].copy()
    overdue = overdue.sort_values("risk_score", ascending=False)
    overdue["eol_date_str"] = overdue["eol_date"].dt.strftime("%Y-%m-%d")
    overdue["days_overdue"] = (today - overdue["eol_date"]).dt.days

    table_data = overdue[["hostname", "model", "device_type", "state", "site_code",
                          "eol_date_str", "days_overdue", "risk_score"]].head(100)
    table = dash_table.DataTable(
        data=table_data.to_dict("records"),
        columns=[
            {"name": "Hostname", "id": "hostname"},
            {"name": "Model", "id": "model"},
            {"name": "Type", "id": "device_type"},
            {"name": "State", "id": "state"},
            {"name": "Site", "id": "site_code"},
            {"name": "EoL Date", "id": "eol_date_str"},
            {"name": "Days Overdue", "id": "days_overdue"},
            {"name": "Risk Score", "id": "risk_score"},
        ],
        page_size=15,
        style_table={"overflowX": "auto"},
        style_cell={"textAlign": "left", "padding": "8px", "fontSize": "13px"},
        style_header={"fontWeight": "bold", "backgroundColor": "#f8f9fa"},
        style_data_conditional=[
            {"if": {"filter_query": "{risk_score} >= 75"}, "backgroundColor": "#f8d7da"},
            {"if": {"filter_query": "{risk_score} >= 50 && {risk_score} < 75"}, "backgroundColor": "#fff3cd"},
        ],
        sort_action="native",
        filter_action="native",
    )

    return fig_eos, fig_eol, html.Div([
        html.P(f"{len(overdue):,} devices past End-of-Life still in production", className="text-danger fw-bold"),
        table,
    ])


# ---------------------------------------------------------------------------
# Callbacks: Proximity Analysis
# ---------------------------------------------------------------------------
@app.callback(
    [Output("proximity-map", "figure"),
     Output("cluster-details", "children")],
    [Input("current-page", "data"),
     Input("proximity-radius", "value"),
     Input("proximity-min-sites", "value"),
     Input("filter-state", "value"),
     Input("filter-affiliate", "value"),
     Input("filter-device-type", "value"),
     Input("filter-lifecycle", "value"),
     Input("filtered-data-signal", "data"),
     Input("exceptions-signal", "data")],
)
def update_proximity(page, radius, min_sites, states, affiliates, dtypes, lifecycle, _signal, _exc_signal):
    if page != "proximity":
        return [no_update] * 2

    _, df = get_filtered_frames(states, affiliates, dtypes, lifecycle)
    geo = df[df["latitude"].notna() & df["longitude"].notna()].copy()

    # Aggregate to site level
    sites = geo.groupby(["site_code", "latitude", "longitude", "site_name", "state"]).agg(
        device_count=("device_id", "count"),
        avg_risk=("risk_score", "mean"),
        total_cost=("total_refresh_cost", "sum"),
        past_eol=("lifecycle_status", lambda x: (x == "Past EoL").sum()),
    ).reset_index()

    if len(sites) < 2:
        return go.Figure(), html.P("Not enough sites for clustering.")

    # DBSCAN clustering using haversine
    coords = np.radians(sites[["latitude", "longitude"]].values)
    earth_radius_miles = 3958.8
    eps_radians = radius / earth_radius_miles
    db = DBSCAN(eps=eps_radians, min_samples=min_sites, metric="haversine")
    sites["cluster"] = db.fit_predict(coords)

    # Filter to actual clusters (not noise = -1)
    clustered = sites[sites["cluster"] >= 0].copy()
    noise = sites[sites["cluster"] < 0].copy()

    # Create map
    fig = go.Figure()

    # Add noise points (unclustered)
    if len(noise) > 0:
        fig.add_trace(go.Scattermap(
            lat=noise["latitude"],
            lon=noise["longitude"],
            mode="markers",
            marker=dict(size=6, color="#6c757d", opacity=0.4),
            name="Unclustered",
            hovertext=noise["site_name"],
        ))

    # Add clustered points with color by cluster
    if len(clustered) > 0:
        n_clusters = clustered["cluster"].nunique()
        colors = BRAND_CHART_COLORS[:max(n_clusters, 1)]
        if n_clusters > len(colors):
            colors = [BRAND_CHART_COLORS[i % len(BRAND_CHART_COLORS)] for i in range(n_clusters)]
        for i, cluster_id in enumerate(sorted(clustered["cluster"].unique())):
            c = clustered[clustered["cluster"] == cluster_id]
            color = colors[i % len(colors)]
            fig.add_trace(go.Scattermap(
                lat=c["latitude"],
                lon=c["longitude"],
                mode="markers",
                marker=dict(size=12, color=color, opacity=0.8),
                name=f"Cluster {cluster_id + 1}",
                hovertext=[
                    f"{row.site_name}<br>Devices: {row.device_count}<br>Risk: {row.avg_risk:.0f}<br>Cost: ${row.total_cost:,.0f}"
                    for _, row in c.iterrows()
                ],
            ))

    fig.update_layout(
        map=dict(style="carto-positron", center=dict(lat=33.0, lon=-85.0), zoom=5),
        margin=dict(t=0, b=0, l=0, r=0),
        showlegend=False,
    )

    # Cluster summary table
    if len(clustered) > 0:
        cluster_summary = clustered.groupby("cluster").agg(
            sites=("site_code", "count"),
            total_devices=("device_count", "sum"),
            avg_risk=("avg_risk", "mean"),
            total_cost=("total_cost", "sum"),
            past_eol=("past_eol", "sum"),
            states=("state", lambda x: ", ".join(sorted(x.unique()))),
        ).reset_index()
        cluster_summary["cluster"] = cluster_summary["cluster"] + 1
        cluster_summary = cluster_summary.sort_values("avg_risk", ascending=False)

        table = dash_table.DataTable(
            data=cluster_summary.to_dict("records"),
            columns=[
                {"name": "Cluster", "id": "cluster"},
                {"name": "Sites", "id": "sites"},
                {"name": "Total Devices", "id": "total_devices"},
                {"name": "Avg Risk", "id": "avg_risk", "type": "numeric", "format": {"specifier": ".1f"}},
                {"name": "Total Cost", "id": "total_cost", "type": "numeric",
                 "format": {"locale": {"symbol": ["$", ""]}, "specifier": "$,.0f"}},
                {"name": "Past EoL", "id": "past_eol"},
                {"name": "States", "id": "states"},
            ],
            page_size=10,
            style_cell={"textAlign": "left", "padding": "8px", "fontSize": "13px"},
            style_header={"fontWeight": "bold", "backgroundColor": "#f8f9fa"},
            sort_action="native",
        )
        detail = html.Div([
            html.P(f"Found {n_clusters} clusters with {len(clustered)} sites within {radius} mi radius.",
                   className="fw-bold text-primary"),
            table,
        ])
    else:
        detail = html.P(f"No clusters found at {radius} mi radius with minimum {min_sites} sites.",
                        className="text-muted")

    return fig, detail


# ---------------------------------------------------------------------------
# Callbacks: Cost & Risk
# ---------------------------------------------------------------------------
@app.callback(
    [Output("cost-by-lifecycle", "figure"),
     Output("cost-by-type", "figure"),
     Output("risk-vs-cost-scatter", "figure"),
     Output("cost-by-affiliate", "figure"),
     Output("cumulative-cost", "figure")],
    [Input("current-page", "data"),
     Input("filter-state", "value"),
     Input("filter-affiliate", "value"),
     Input("filter-device-type", "value"),
     Input("filter-lifecycle", "value"),
     Input("filtered-data-signal", "data"),
     Input("exceptions-signal", "data")],
)
def update_cost(page, states, affiliates, dtypes, lifecycle, _signal, _exc_signal):
    if page not in {"story", "cost"}:
        return [no_update] * 5

    _, df = get_filtered_frames(states, affiliates, dtypes, lifecycle)

    # Cost by lifecycle
    cost_lc = df.groupby("lifecycle_status")["total_refresh_cost"].sum().reset_index()
    fig_cost_lc = px.bar(cost_lc, x="lifecycle_status", y="total_refresh_cost",
                         color="lifecycle_status", color_discrete_map=LIFECYCLE_COLORS,
                         labels={"total_refresh_cost": "Total Cost ($)", "lifecycle_status": ""})
    fig_cost_lc.update_traces(
        marker_line_color="#003087",
        marker_line_width=1,
        hovertemplate="<b>%{x}</b><br>Total Cost: $%{y:,.0f}<extra></extra>"
    )
    fig_cost_lc.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=350, showlegend=False)

    # Cost by device type
    cost_dt = df.groupby("device_type").agg(
        avg_cost=("total_refresh_cost", "mean"),
        total_cost=("total_refresh_cost", "sum"),
    ).reset_index()
    fig_cost_dt = px.bar(cost_dt, x="device_type", y="avg_cost",
                         color="device_type", color_discrete_map=DEVICE_TYPE_COLORS,
                         labels={
                             "avg_cost": "Avg Cost per Device ($)",
                             "device_type": "Device Type",
                         })
    fig_cost_dt.update_traces(
        marker_line_color="#003087",
        marker_line_width=1,
        hovertemplate="<b>%{x}</b><br>Avg Cost: $%{y:,.0f}<extra></extra>"
    )
    fig_cost_dt.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=350, showlegend=False)

    # Risk vs cost scatter (by site)
    site_data = df.groupby("site_code").agg(
        avg_risk=("risk_score", "mean"),
        total_cost=("total_refresh_cost", "sum"),
        count=("device_id", "count"),
        state=("state", "first"),
    ).reset_index()
    site_data = site_data[site_data["total_cost"] > 0]
    fig_scatter = px.scatter(site_data, x="avg_risk", y="total_cost",
                             size="count", color="state",
                             color_discrete_sequence=BRAND_CHART_COLORS,
                             labels={"avg_risk": "Average Risk Score",
                                     "total_cost": "Total Refresh Cost ($)"},
                             opacity=0.6)
    fig_scatter.update_traces(
        marker=dict(line=dict(color="#003087", width=1)),
        hovertemplate="<b>%{hovertext}</b><br>Avg Risk: %{x:.1f}<br>Refresh Cost: $%{y:,.0f}<br>Devices: %{marker.size:,}<extra></extra>",
        hovertext=site_data["site_code"],
    )
    fig_scatter.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=350, showlegend=False)

    # Cost by affiliate
    cost_aff = df.groupby("affiliate")["total_refresh_cost"].sum().sort_values(ascending=True).reset_index()
    fig_cost_aff = go.Figure(go.Bar(
        x=cost_aff["total_refresh_cost"],
        y=cost_aff["affiliate"],
        orientation="h",
        marker=dict(color="#1479BD"),
        hovertemplate="<b>%{y}</b><br>Total Cost: $%{x:,.0f}<extra></extra>",
    ))
    fig_cost_aff.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=350,
                               xaxis_title="Total Cost ($)", yaxis_title="")

    # Cumulative cost by EoL date (investment timeline)
    eol_cost = df[df["eol_date"].notna() & (df["total_refresh_cost"] > 0)].copy()
    eol_cost = eol_cost.sort_values("eol_date")
    eol_cost["cumulative_cost"] = eol_cost["total_refresh_cost"].cumsum()
    fig_cum = px.area(eol_cost, x="eol_date", y="cumulative_cost",
                      color_discrete_sequence=["#1479BD"],
                      labels={"eol_date": "End-of-Life Date", "cumulative_cost": "Cumulative Cost ($)"})
    fig_cum.update_traces(
        line=dict(color="#1479BD"),
        fillcolor="rgba(20, 121, 189, 0.28)",
        hovertemplate="EoL Date: %{x|%b %d, %Y}<br>Cumulative Cost: $%{y:,.0f}<extra></extra>"
    )
    today_str = pd.Timestamp.now().strftime("%Y-%m-%d")
    fig_cum.add_vline(x=today_str, line_dash="dash", line_color="red")
    fig_cum.add_annotation(
        x=today_str,
        y=0.98,
        yref="paper",
        text="Today",
        showarrow=False,
        xanchor="left",
        xshift=12,
        yshift=0,
        font=dict(color="red"),
    )
    fig_cum.update_layout(margin=dict(t=44, b=10, l=16, r=16), height=350)

    return fig_cost_lc, fig_cost_dt, fig_scatter, fig_cost_aff, fig_cum


# ---------------------------------------------------------------------------
# Callbacks: Port Utilization & Capacity
# ---------------------------------------------------------------------------
@app.callback(
    [Output("capacity-kpi-row", "children"),
     Output("capacity-histogram", "figure"),
     Output("capacity-by-type", "figure"),
     Output("capacity-by-site", "figure"),
     Output("capacity-table", "children")],
    [Input("current-page", "data"),
     Input("filter-state", "value"),
     Input("filter-affiliate", "value"),
     Input("filter-device-type", "value"),
     Input("filter-lifecycle", "value"),
     Input("filtered-data-signal", "data"),
     Input("exceptions-signal", "data")],
)
def update_capacity(page, states, affiliates, dtypes, lifecycle, _signal, _exc_signal):
    if page != "capacity":
        return [no_update] * 5

    df = apply_filters(DF, states, affiliates, dtypes, lifecycle)
    df = df[df["exception_flagged"] != True]

    # Only devices with port data (from NA source — switches, routers, voice gateways)
    port_df = df[df["total_ports"].notna() & (df["total_ports"] > 0)].copy()
    port_df["utilization_pct"] = port_df["ports_in_use"] / port_df["total_ports"] * 100

    total_ports = port_df["total_ports"].sum()
    in_use = port_df["ports_in_use"].sum()
    available = port_df["free_ports"].sum()
    avg_util = port_df["utilization_pct"].mean() if len(port_df) > 0 else 0
    over_90 = (port_df["utilization_pct"] >= 90).sum()

    # ---- KPI row ----
    kpi_row = dbc.Row([
        dbc.Col(make_kpi_card("Total Ports", f"{total_ports:,.0f}", "sitemap", "primary"), md=2),
        dbc.Col(make_kpi_card("Ports In Use", f"{in_use:,.0f}", "plug", "warning"), md=2),
        dbc.Col(make_kpi_card("Ports Available", f"{available:,.0f}", "check-circle", "success"), md=2),
        dbc.Col(make_kpi_card("Avg Utilization", f"{avg_util:.1f}%", "tachometer-alt", "info"), md=2),
        dbc.Col(make_kpi_card(">90% Utilized", f"{over_90:,}", "exclamation-triangle", "danger"), md=2),
        dbc.Col(make_kpi_card("Devices w/ Data", f"{len(port_df):,}", "server", "secondary"), md=2),
    ], className="mb-4")

    # ---- Chart 1: Utilization Distribution Histogram ----
    bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100.01]
    labels = ["0-10%", "10-20%", "20-30%", "30-40%", "40-50%",
              "50-60%", "60-70%", "70-80%", "80-90%", "90-100%"]
    port_df["util_band"] = pd.cut(port_df["utilization_pct"], bins=bins, labels=labels, right=False)
    band_counts = port_df["util_band"].value_counts().reindex(labels, fill_value=0)

    # Color each bar by its utilization level (green→red)
    band_colors = [
        "#9CC987", "#9CC987",  # 0-20% green
        "#B2D235", "#B2D235",  # 20-40% lime
        "#00BDF2", "#00BDF2",  # 40-60% blue
        "#007DBA", "#007DBA",  # 60-80% dark blue
        "#ED1D24", "#ED1D24",  # 80-100% red
    ]
    fig_hist = go.Figure(go.Bar(
        x=band_counts.index.astype(str),
        y=band_counts.values,
        marker=dict(color=band_colors, line=dict(color="#003087", width=1)),
        hovertemplate="<b>%{x}</b><br>Devices: %{y:,}<extra></extra>",
    ))
    fig_hist.update_layout(
        margin=dict(t=10, b=10, l=10, r=10), height=350,
        xaxis_title="Utilization Band", yaxis_title="Device Count",
    )

    # ---- Chart 2: Utilization by Device Type ----
    type_util = port_df.groupby("device_type").agg(
        avg_util=("utilization_pct", "mean"),
        device_count=("device_id", "count"),
        total_ports=("total_ports", "sum"),
        in_use=("ports_in_use", "sum"),
    ).reset_index()

    fig_type = go.Figure()
    for _, row in type_util.iterrows():
        dt = row["device_type"]
        fig_type.add_trace(go.Bar(
            x=[dt],
            y=[row["avg_util"]],
            name=dt,
            marker=dict(
                color=DEVICE_TYPE_COLORS.get(dt, "#6c757d"),
                line=dict(color="#003087", width=1),
            ),
            customdata=[[int(row["device_count"]), int(row["total_ports"]), int(row["in_use"])]],
            hovertemplate=(
                "<b>%{x}</b><br>"
                "Avg Utilization: %{y:.1f}%<br>"
                "Devices: %{customdata[0]:,}<br>"
                "Total Ports: %{customdata[1]:,}<br>"
                "In Use: %{customdata[2]:,}"
                "<extra></extra>"
            ),
        ))
    fig_type.update_layout(
        margin=dict(t=10, b=10, l=10, r=10), height=350,
        yaxis_title="Avg Utilization %", showlegend=False,
    )

    # ---- Chart 3: Top 20 Sites by Avg Utilization ----
    site_util = port_df.groupby("site_code").agg(
        avg_util=("utilization_pct", "mean"),
        device_count=("device_id", "count"),
        total_ports=("total_ports", "sum"),
        free_ports=("free_ports", "sum"),
    ).reset_index()
    # Only sites with at least 3 devices so the avg is meaningful
    site_util = site_util[site_util["device_count"] >= 3]
    top_sites = site_util.sort_values("avg_util", ascending=True).tail(20)

    fig_site = go.Figure(go.Bar(
        x=top_sites["avg_util"],
        y=top_sites["site_code"],
        orientation="h",
        marker=dict(
            color=top_sites["avg_util"],
            colorscale=BRAND_CONTINUOUS_SCALE,
            colorbar=dict(title="Util %"),
            line=dict(color="#003087", width=1),
        ),
        customdata=top_sites[["device_count", "total_ports", "free_ports"]].values,
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Avg Utilization: %{x:.1f}%<br>"
            "Devices: %{customdata[0]:,}<br>"
            "Total Ports: %{customdata[1]:,}<br>"
            "Free Ports: %{customdata[2]:,}"
            "<extra></extra>"
        ),
    ))
    fig_site.update_layout(
        margin=dict(t=10, b=10, l=10, r=10), height=500,
        xaxis_title="Avg Utilization %", yaxis_title="Site",
    )

    # ---- Table: High-Utilization Devices (>90%) ----
    high_util = port_df[port_df["utilization_pct"] >= 90].sort_values(
        "utilization_pct", ascending=False
    ).copy()
    high_util["utilization_pct"] = high_util["utilization_pct"].round(1)
    high_util["total_ports"] = high_util["total_ports"].astype(int)
    high_util["ports_in_use"] = high_util["ports_in_use"].astype(int)
    high_util["free_ports"] = high_util["free_ports"].astype(int)

    table_data = high_util[[
        "hostname", "model", "device_type", "site_code", "state",
        "total_ports", "ports_in_use", "free_ports", "utilization_pct",
        "lifecycle_status", "risk_score",
    ]].head(200)

    cap_table = dash_table.DataTable(
        data=table_data.to_dict("records"),
        columns=[
            {"name": "Hostname", "id": "hostname"},
            {"name": "Model", "id": "model"},
            {"name": "Type", "id": "device_type"},
            {"name": "Site", "id": "site_code"},
            {"name": "State", "id": "state"},
            {"name": "Total Ports", "id": "total_ports"},
            {"name": "In Use", "id": "ports_in_use"},
            {"name": "Free", "id": "free_ports"},
            {"name": "Util %", "id": "utilization_pct"},
            {"name": "Lifecycle", "id": "lifecycle_status"},
            {"name": "Risk", "id": "risk_score"},
        ],
        page_size=15,
        sort_action="native",
        filter_action="native",
        style_table={"overflowX": "auto"},
        style_cell={"textAlign": "left", "padding": "8px", "fontSize": "13px"},
        style_header={"fontWeight": "bold", "backgroundColor": "#f8f9fa"},
        style_data_conditional=[
            {"if": {"filter_query": "{utilization_pct} >= 95"}, "backgroundColor": "#f8d7da"},
            {"if": {"filter_query": "{utilization_pct} >= 90 && {utilization_pct} < 95"},
             "backgroundColor": "#fff3cd"},
        ],
    )

    table_content = html.Div([
        html.P(
            f"Showing {len(high_util):,} devices above 90% port utilization. "
            f"Port data available for {len(port_df):,} devices from Network Automation "
            f"source (Switches, Routers, Voice Gateways).",
            className="text-muted small mb-2",
        ),
        cap_table,
    ])

    return kpi_row, fig_hist, fig_type, fig_site, table_content


# ---------------------------------------------------------------------------
# Callbacks: Exceptions
# ---------------------------------------------------------------------------
@app.callback(
    Output("exception-search-results", "children"),
    Input("exception-search-btn", "n_clicks"),
    State("exception-search", "value"),
    prevent_initial_call=True,
)
def search_for_exceptions(n, query):
    if not query or len(query) < 2:
        return html.P("Enter at least 2 characters to search.", className="text-muted")

    q = query.upper()
    matches = DF[
        (DF["hostname"].str.upper().str.contains(q, na=False)) |
        (DF["serial_number"].str.upper().str.contains(q, na=False))
    ].head(50)

    if len(matches) == 0:
        return html.P("No devices found.", className="text-muted")

    rows = []
    exceptions = load_exceptions()
    for _, row in matches.iterrows():
        is_flagged = str(row["serial_number"]) in exceptions
        rows.append(
            dbc.Card(dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.Strong(row["hostname"]),
                        html.Span(f" — {row['model']} ({row['device_type']})", className="text-muted ms-2"),
                        html.Br(),
                        html.Small(f"SN: {row['serial_number']} | Site: {row['site_code']} | State: {row['state']}",
                                   className="text-muted"),
                    ], md=8),
                    dbc.Col([
                        dbc.Badge(row["lifecycle_status"],
                                  color={"Past EoL": "danger", "Past EoS": "warning",
                                         "Current": "success"}.get(row["lifecycle_status"], "secondary"),
                                  className="me-2"),
                        dbc.Button(
                            "Unflag" if is_flagged else "Flag Exception",
                            id={"type": "exception-toggle", "index": str(row["serial_number"])},
                            color="outline-danger" if is_flagged else "outline-warning",
                            size="sm",
                        ),
                    ], md=4, className="text-end"),
                ]),
                dbc.Collapse(
                    dbc.Input(
                        id={"type": "exception-reason", "index": str(row["serial_number"])},
                        placeholder="Reason for exception...",
                        className="mt-2",
                        value=exceptions.get(str(row["serial_number"]), {}).get("reason", ""),
                    ),
                    is_open=not is_flagged,
                ),
            ]), className="mb-2 shadow-sm"),
        )

    return html.Div(rows)


@app.callback(
    [Output("exception-list", "children"),
     Output("exceptions-signal", "data")],
    [Input("current-page", "data"),
     Input({"type": "exception-toggle", "index": ALL}, "n_clicks")],
    [State({"type": "exception-reason", "index": ALL}, "value"),
     State({"type": "exception-toggle", "index": ALL}, "id"),
     State("exceptions-signal", "data")],
)
def manage_exceptions(page, clicks, reasons, ids, prev_signal):
    ctx = callback_context
    exceptions = load_exceptions()
    signal = prev_signal or 0

    # Handle toggle clicks
    if ctx.triggered and ctx.triggered[0]["prop_id"] != "current-page.data":
        for i, click in enumerate(clicks or []):
            if click and i < len(ids):
                serial = ids[i]["index"]
                if serial in exceptions:
                    del exceptions[serial]
                else:
                    reason = reasons[i] if i < len(reasons) and reasons[i] else ""
                    exceptions[serial] = {
                        "reason": reason,
                        "flagged_at": pd.Timestamp.now().isoformat(),
                    }
        save_exceptions(exceptions)
        signal += 1  # bump signal so other pages re-render

    # If not on exceptions page, don't try to update exception-list
    if page != "exceptions":
        return no_update, signal

    # Display current exceptions
    if not exceptions:
        return html.P("No exceptions flagged.", className="text-muted"), signal

    rows = []
    for serial, info in exceptions.items():
        device = DF[DF["serial_number"] == serial]
        if len(device) > 0:
            d = device.iloc[0]
            rows.append(html.Li([
                html.Strong(d["hostname"]),
                f" ({d['model']}) — SN: {serial}",
                html.Br(),
                html.Small(f"Reason: {info.get('reason', 'No reason given')} | "
                           f"Flagged: {info.get('flagged_at', 'Unknown')}", className="text-muted"),
            ], className="mb-2"))
        else:
            rows.append(html.Li(f"SN: {serial} — {info.get('reason', '')}", className="mb-2"))

    return html.Ul(rows, className="list-unstyled"), signal


@app.callback(
    Output("exception-count-badge", "children"),
    [Input("current-page", "data"),
     Input({"type": "exception-toggle", "index": ALL}, "n_clicks")],
)
def update_exception_count(page, clicks):
    if page != "exceptions":
        return no_update
    exceptions = load_exceptions()
    return dbc.Badge(f"{len(exceptions)} exceptions flagged",
                     color="warning" if exceptions else "secondary",
                     className="fs-6")


# ---------------------------------------------------------------------------
# Callbacks: Priorities
# ---------------------------------------------------------------------------
@app.callback(
    [Output("priority-sites", "figure"),
     Output("priority-models", "figure"),
     Output("refresh-schedule-table", "children"),
     Output("staffing-kpis", "children"),
     Output("staffing-by-phase", "figure"),
     Output("staffing-by-site", "figure"),
     Output("staffing-fte-table", "children")],
    [Input("current-page", "data"),
     Input("filter-state", "value"),
     Input("filter-affiliate", "value"),
     Input("filter-device-type", "value"),
     Input("filter-lifecycle", "value"),
     Input("filtered-data-signal", "data"),
     Input("exceptions-signal", "data")],
)
def update_priorities(page, states, affiliates, dtypes, lifecycle, _signal, _exc_signal):
    if page != "story":
        return [no_update] * 7

    _, df = get_filtered_frames(states, affiliates, dtypes, lifecycle)

    # Top risk sites
    site_risk = df.groupby(["site_code", "site_name", "state", "affiliate"]).agg(
        avg_risk=("risk_score", "mean"),
        device_count=("device_id", "count"),
        past_eol=("lifecycle_status", lambda x: (x == "Past EoL").sum()),
        total_cost=("total_refresh_cost", "sum"),
    ).reset_index()
    top_sites = site_risk.sort_values("avg_risk", ascending=False).head(20)
    fig_sites = go.Figure(go.Bar(
        x=top_sites["avg_risk"],
        y=top_sites["site_code"],
        orientation="h",
        marker=dict(color=top_sites["avg_risk"], colorscale=BRAND_CONTINUOUS_SCALE,
                    colorbar=dict(title="Risk")),
        customdata=top_sites[["site_name", "device_count", "past_eol", "total_cost", "affiliate"]].values,
        hovertemplate=(
            "<b>%{y}</b> — %{customdata[0]}<br>"
            "Affiliate: %{customdata[4]}<br>"
            "Avg Risk: %{x:.1f}<br>"
            "Devices: %{customdata[1]:,}<br>"
            "Past EoL: %{customdata[2]:,}<br>"
            "Refresh Cost: $%{customdata[3]:,.0f}"
            "<extra></extra>"
        ),
    ))
    fig_sites.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=500,
                            xaxis_title="Avg Risk Score", yaxis_title="Site",
                            yaxis=dict(autorange="reversed"))

    # Priority models
    model_risk = df.groupby(["model", "model_parent", "device_type"]).agg(
        avg_risk=("risk_score", "mean"),
        count=("device_id", "count"),
        total_cost=("total_refresh_cost", "sum"),
    ).reset_index()
    model_risk["priority_score"] = model_risk["avg_risk"] * np.log1p(model_risk["count"])
    top_models = model_risk.sort_values("priority_score", ascending=False).head(20)

    fig_models = go.Figure()
    for dtype in DEVICE_TYPE_COLORS:
        subset = top_models[top_models["device_type"] == dtype]
        if len(subset) > 0:
            fig_models.add_trace(go.Bar(
                x=subset["priority_score"],
                y=subset["model"],
                orientation="h",
                name=dtype,
                marker_color=DEVICE_TYPE_COLORS[dtype],
                marker_line=dict(color="#003087", width=1),
                customdata=subset[["count", "avg_risk", "total_cost"]].values,
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    "Type: " + dtype + "<br>"
                    "Priority Score: %{x:.1f}<br>"
                    "Devices: %{customdata[0]:,}<br>"
                    "Avg Risk: %{customdata[1]:.1f}<br>"
                    "Refresh Cost: $%{customdata[2]:,.0f}"
                    "<extra></extra>"
                ),
            ))
    fig_models.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=500,
                             xaxis_title="Priority Score", yaxis_title="Model",
                             yaxis=dict(autorange="reversed"), barmode="stack")

    # Refresh schedule recommendation
    today = pd.Timestamp.now()
    schedule_data = []

    # Build phase filters for reuse in staffing section
    phase_filters = {
        "1 — Immediate": df["lifecycle_status"] == "Past EoL",
        "2 — Near-Term": df["lifecycle_status"] == "Past EoS",
        "3 — Planning": df["lifecycle_status"] == "EoS within 1yr",
        "4 — Strategic": (df["in_scope"] == "Yes") & (df["lifecycle_status"] == "Current"),
    }
    phase_timelines = {
        "1 — Immediate": "0-6 months",
        "2 — Near-Term": "6-18 months",
        "3 — Planning": "12-24 months",
        "4 — Strategic": "18-36 months",
    }
    phase_risks = {
        "1 — Immediate": "Critical — No vendor support, security vulnerabilities",
        "2 — Near-Term": "High — Limited procurement, approaching EoL",
        "3 — Planning": "Medium — Plan procurement before EoS",
        "4 — Strategic": "Low — Proactive lifecycle management",
    }
    phase_criteria = {
        "1 — Immediate": "Past End-of-Life",
        "2 — Near-Term": "Past End-of-Sale",
        "3 — Planning": "EoS within 1 year",
        "4 — Strategic": "In-Scope, still Current",
    }

    for phase, mask in phase_filters.items():
        p = df[mask]
        schedule_data.append({
            "Phase": phase,
            "Criteria": phase_criteria[phase],
            "Devices": len(p),
            "Sites": p["site_code"].nunique(),
            "Est. Cost": f"${p['total_refresh_cost'].sum():,.0f}",
            "Timeline": phase_timelines[phase],
            "Risk": phase_risks[phase],
        })

    table = dash_table.DataTable(
        data=schedule_data,
        columns=[{"name": k, "id": k} for k in schedule_data[0].keys()],
        style_cell={"textAlign": "left", "padding": "10px", "fontSize": "13px"},
        style_header={"fontWeight": "bold", "backgroundColor": "#f8f9fa"},
        style_data_conditional=[
            {"if": {"filter_query": '{Phase} = "1 — Immediate"'}, "backgroundColor": "#f8d7da"},
            {"if": {"filter_query": '{Phase} = "2 — Near-Term"'}, "backgroundColor": "#fff3cd"},
            {"if": {"filter_query": '{Phase} = "3 — Planning"'}, "backgroundColor": "#d1ecf1"},
            {"if": {"filter_query": '{Phase} = "4 — Strategic"'}, "backgroundColor": "#d4edda"},
        ],
    )

    # ==================================================================
    # Staffing & Resource Planning
    # ==================================================================
    HOURS_PER_FTE_YEAR = 1_800  # standard billable hours per FTE per year

    # Filter to devices that have staffing data
    staffed = df[df["de_hours"].notna()].copy()
    total_de = staffed["de_hours"].sum()
    total_se = staffed["se_hours"].sum()
    total_fot = staffed["fot_hours"].sum()
    total_hours = total_de + total_se + total_fot
    total_ftes = total_hours / HOURS_PER_FTE_YEAR

    # Staffing KPI row
    staffing_kpis = dbc.Row([
        dbc.Col(make_kpi_card(
            "Devices with Staffing Data",
            f"{len(staffed):,} of {len(df):,}",
            "users-cog", "primary",
        ), md=3),
        dbc.Col(make_kpi_card(
            "Total Engineering Hours",
            f"{total_hours:,.0f}",
            "clock", "info",
        ), md=3),
        dbc.Col(make_kpi_card(
            "Estimated FTEs Required",
            f"{total_ftes:,.1f}",
            "hard-hat", "warning",
        ), md=3),
        dbc.Col(make_kpi_card(
            "Avg Hours per Device",
            f"{(total_hours / max(len(staffed), 1)):.1f}",
            "tachometer-alt", "success",
        ), md=3),
    ])

    # ---- Chart 1: Engineering Hours by Refresh Phase (grouped bar) ----
    phase_hours = []
    phase_colors = {
        "1 — Immediate": "#dc3545",
        "2 — Near-Term": "#fd7e14",
        "3 — Planning": "#0dcaf0",
        "4 — Strategic": "#28a745",
    }
    for phase, mask in phase_filters.items():
        p = staffed[mask]
        phase_hours.append({
            "Phase": phase,
            "Design Engineer": p["de_hours"].sum(),
            "Systems Engineer": p["se_hours"].sum(),
            "Field Ops Tech": p["fot_hours"].sum(),
        })
    phase_hours_df = pd.DataFrame(phase_hours)

    role_colors = {
        "Design Engineer": "#0d6efd",
        "Systems Engineer": "#6610f2",
        "Field Ops Tech": "#20c997",
    }
    fig_phase = go.Figure()
    for role, color in role_colors.items():
        fig_phase.add_trace(go.Bar(
            x=phase_hours_df["Phase"],
            y=phase_hours_df[role],
            name=role,
            marker_color=color,
            hovertemplate=(
                "<b>%{x}</b><br>"
                + role + ": %{y:,.0f} hrs"
                "<extra></extra>"
            ),
        ))
    fig_phase.update_layout(
        barmode="group",
        margin=dict(t=10, b=10, l=10, r=10),
        height=400,
        yaxis_title="Hours",
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
    )

    # ---- Chart 2: Top 15 Sites by Total Engineering Hours ----
    site_hours = staffed.groupby("site_code").agg(
        de=("de_hours", "sum"),
        se=("se_hours", "sum"),
        fot=("fot_hours", "sum"),
        device_count=("device_id", "count"),
    ).reset_index()
    site_hours["total"] = site_hours["de"] + site_hours["se"] + site_hours["fot"]
    top_hour_sites = site_hours.sort_values("total", ascending=True).tail(15)

    fig_site_hours = go.Figure()
    for role, col, color in [
        ("Design Engineer", "de", role_colors["Design Engineer"]),
        ("Systems Engineer", "se", role_colors["Systems Engineer"]),
        ("Field Ops Tech", "fot", role_colors["Field Ops Tech"]),
    ]:
        fig_site_hours.add_trace(go.Bar(
            x=top_hour_sites[col],
            y=top_hour_sites["site_code"],
            name=role,
            orientation="h",
            marker_color=color,
            hovertemplate=(
                "<b>%{y}</b><br>"
                + role + ": %{x:,.0f} hrs"
                "<extra></extra>"
            ),
        ))
    fig_site_hours.update_layout(
        barmode="stack",
        margin=dict(t=10, b=10, l=10, r=10),
        height=450,
        xaxis_title="Total Hours",
        legend=dict(orientation="h", yanchor="bottom", y=-0.22, xanchor="center", x=0.5),
    )

    # ---- Table: FTE Projection by Phase ----
    fte_data = []
    for phase, mask in phase_filters.items():
        p = staffed[mask]
        de_h = p["de_hours"].sum()
        se_h = p["se_hours"].sum()
        fot_h = p["fot_hours"].sum()
        tot_h = de_h + se_h + fot_h
        fte_data.append({
            "Phase": phase,
            "Timeline": phase_timelines[phase],
            "Devices": len(p),
            "DE Hours": f"{de_h:,.0f}",
            "SE Hours": f"{se_h:,.0f}",
            "FOT Hours": f"{fot_h:,.0f}",
            "Total Hours": f"{tot_h:,.0f}",
            "Est. FTEs": f"{tot_h / HOURS_PER_FTE_YEAR:.1f}",
        })

    fte_table = dash_table.DataTable(
        data=fte_data,
        columns=[{"name": k, "id": k} for k in fte_data[0].keys()],
        style_cell={"textAlign": "left", "padding": "10px", "fontSize": "13px"},
        style_header={"fontWeight": "bold", "backgroundColor": "#f8f9fa"},
        style_data_conditional=[
            {"if": {"filter_query": '{Phase} = "1 — Immediate"'}, "backgroundColor": "#f8d7da"},
            {"if": {"filter_query": '{Phase} = "2 — Near-Term"'}, "backgroundColor": "#fff3cd"},
            {"if": {"filter_query": '{Phase} = "3 — Planning"'}, "backgroundColor": "#d1ecf1"},
            {"if": {"filter_query": '{Phase} = "4 — Strategic"'}, "backgroundColor": "#d4edda"},
        ],
    )

    staffing_fte_content = html.Div([
        html.P(
            f"Based on {len(staffed):,} devices with ModelData staffing estimates "
            f"({len(staffed) / max(len(df), 1) * 100:.0f}% of filtered devices). "
            f"FTE assumes {HOURS_PER_FTE_YEAR:,} billable hours per year.",
            className="text-muted small mb-2",
        ),
        fte_table,
    ])

    return (fig_sites, fig_models, table,
            staffing_kpis, fig_phase, fig_site_hours, staffing_fte_content)


# ---------------------------------------------------------------------------
# Callbacks: Story Page Panels
# ---------------------------------------------------------------------------
@app.callback(
    Output("collapse-cost-breakdown", "is_open"),
    Input("btn-toggle-cost-breakdown", "n_clicks"),
    State("collapse-cost-breakdown", "is_open"),
    prevent_initial_call=True,
)
def toggle_cost_breakdown(n, is_open):
    if not n:
        return is_open
    return not is_open


@app.callback(
    [Output("story-cost-breakdown", "children"),
     Output("support-story", "children"),
     Output("data-confidence", "children")],
    [Input("current-page", "data"),
     Input("filter-state", "value"),
     Input("filter-affiliate", "value"),
     Input("filter-device-type", "value"),
     Input("filter-lifecycle", "value"),
     Input("filtered-data-signal", "data"),
     Input("exceptions-signal", "data")],
)
def update_story_panels(page, states, affiliates, dtypes, lifecycle, _signal, _exc_signal):
    if page != "story":
        return [no_update] * 3

    full_df, risk_df = get_filtered_frames(states, affiliates, dtypes, lifecycle)

    device_cost = risk_df["device_cost"].sum()
    dna_cost = risk_df["dna_cost"].sum()
    staging_cost = risk_df["staging_cost"].sum()
    tax_overhead = risk_df["tax_overhead"].sum()
    labor_cost = risk_df["labor_cost"].sum()
    all_in_total = risk_df["total_refresh_cost"].sum()

    cost_table = dash_table.DataTable(
        data=[
            {"Component": "Device Cost (hardware)", "Fleet Total": fmt_money(device_cost), "What It Covers": "Physical Cisco device purchase price"},
            {"Component": "DNA Licensing", "Fleet Total": fmt_money(dna_cost), "What It Covers": "Cisco software subscription licenses"},
            {"Component": "Staging Cost", "Fleet Total": fmt_money(staging_cost), "What It Covers": "Pre-deployment preparation and testing"},
            {"Component": "Tax & Overhead", "Fleet Total": fmt_money(tax_overhead), "What It Covers": "Taxes and administrative overhead"},
            {"Component": "Labor Cost", "Fleet Total": fmt_money(labor_cost), "What It Covers": "Engineering and technician labor"},
            {"Component": "All-In Total", "Fleet Total": fmt_money(all_in_total), "What It Covers": "Combined planning estimate"},
        ],
        columns=[
            {"name": "Component", "id": "Component"},
            {"name": "Fleet Total", "id": "Fleet Total"},
            {"name": "What It Covers", "id": "What It Covers"},
        ],
        style_table={"overflowX": "auto"},
        style_cell={"textAlign": "left", "padding": "8px", "fontSize": "13px"},
        style_header={"fontWeight": "bold", "backgroundColor": "#f8f9fa"},
    )

    unknown_count = int((full_df["lifecycle_status"] == "Unknown").sum())
    unknown_pct = (unknown_count / max(len(full_df), 1)) * 100
    support_story = html.Div([
        html.P(
            f"{int((risk_df['lifecycle_status'] == 'Past EoL').sum()):,} devices are already Past EoL and should be treated as the highest support and security concern.",
            className="mb-2",
        ),
        html.P(
            f"{int((risk_df['lifecycle_status'] == 'Past EoS').sum()):,} additional devices are Past EoS, meaning the procurement window has closed and refresh flexibility is shrinking.",
            className="mb-2",
        ),
        html.P(
            f"{unknown_count:,} devices ({unknown_pct:.1f}% of the visible fleet) remain Unknown and are excluded from risk prioritization per Southern Company guidance.",
            className="mb-0",
        ),
    ])

    geocoded_pct = (full_df["latitude"].notna() & full_df["longitude"].notna()).mean() * 100 if len(full_df) else 0
    replacement_pct = (risk_df["replacement_device"].fillna("").astype(str).str.len() > 0).mean() * 100 if len(risk_df) else 0
    staffing_mask = risk_df[["de_hours", "se_hours", "fot_hours"]].notna().all(axis=1) if len(risk_df) else pd.Series(dtype=bool)
    staffing_pct = staffing_mask.mean() * 100 if len(risk_df) else 0
    spend_pct = (risk_df["total_refresh_cost"].sum() / max(full_df["total_refresh_cost"].sum(), 1)) * 100 if len(full_df) else 0

    confidence = dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H5(f"{geocoded_pct:.1f}%", className="mb-1 text-primary"),
            html.Small("Geocoded Coverage", className="text-muted"),
        ]), className="border-0 bg-light"), md=6, className="mb-2"),
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H5(f"{spend_pct:.1f}%", className="mb-1 text-primary"),
            html.Small("Modeled Spend in Risk-Focused Fleet", className="text-muted"),
        ]), className="border-0 bg-light"), md=6, className="mb-2"),
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H5(f"{replacement_pct:.1f}%", className="mb-1 text-primary"),
            html.Small("Replacement Mapping Coverage", className="text-muted"),
        ]), className="border-0 bg-light"), md=6, className="mb-2"),
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H5(f"{staffing_pct:.1f}%", className="mb-1 text-primary"),
            html.Small("Staffing-Hour Coverage", className="text-muted"),
        ]), className="border-0 bg-light"), md=6, className="mb-2"),
        dbc.Col(html.Small(
            "Unknown lifecycle rows stay in inventory counts but are intentionally removed from the risk narrative.",
            className="text-muted",
        ), md=12),
    ], className="g-2")

    return cost_table, support_story, confidence


@app.callback(
    [Output("optimizer-kpis", "children"),
     Output("optimizer-build-chart", "figure"),
     Output("optimizer-first-wave", "children"),
     Output("optimizer-full-portfolio", "children"),
     Output("optimizer-exec-brief", "children"),
     Output("final-recommendation", "children")],
    [Input("current-page", "data"),
     Input("optimizer-budget", "value"),
     Input("optimizer-fte", "value"),
     Input("optimizer-objective", "value"),
     Input("filter-state", "value"),
     Input("filter-affiliate", "value"),
     Input("filter-device-type", "value"),
     Input("filter-lifecycle", "value"),
     Input("filtered-data-signal", "data"),
     Input("exceptions-signal", "data")],
)
def update_optimizer(page, budget_cap, fte_cap, objective_mode, states, affiliates, dtypes, lifecycle, _signal, _exc_signal):
    if page != "story":
        return [no_update] * 6

    _, risk_df = get_filtered_frames(states, affiliates, dtypes, lifecycle)
    selected_df, first_wave_df, scored_df = build_optimizer_plan(
        risk_df,
        budget_cap or 12,
        fte_cap or 10,
        objective_mode or "Balanced",
    )

    if selected_df.empty:
        empty_message = html.P("No program-scale sites fit the current budget and FTE limits.", className="text-muted")
        return (
            dbc.Alert("No recommendation available for the current constraints.", color="warning"),
            blank_figure("No recommendation available."),
            empty_message,
            html.Div(),
            html.P("Relax the optimizer constraints or broaden the filters to build a portfolio recommendation.", className="mb-0"),
            html.P("No final recommendation can be generated under the current constraints.", className="mb-0"),
        )

    kpis = dbc.Row([
        dbc.Col(make_kpi_card("Sites Selected", f"{len(selected_df):,}", "map-marker-alt", "primary"), md=3),
        dbc.Col(make_kpi_card("Devices Covered", f"{int(selected_df['device_count'].sum()):,}", "server", "info"), md=3),
        dbc.Col(make_kpi_card("Planning Budget Used", fmt_money(selected_df["planning_budget"].sum()), "dollar-sign", "warning"), md=3),
        dbc.Col(make_kpi_card("FTE Used", f"{selected_df['fte'].sum():.1f}", "users-cog", "success"), md=3),
    ], className="g-3")

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=selected_df["selection_rank"],
        y=selected_df["planning_budget_used"] / 1_000_000,
        name="Cumulative Budget ($M)",
        marker_color="#00BDF2",
        hovertemplate="Site #%{x}<br>Budget Used: $%{y:.2f}M<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=selected_df["selection_rank"],
        y=selected_df["risk_covered_pct"],
        name="Risk Covered (%)",
        yaxis="y2",
        mode="lines+markers",
        line=dict(color="#ED1D24", width=3),
        hovertemplate="Site #%{x}<br>Risk Covered: %{y:.1f}%<extra></extra>",
    ))
    wave_size = min(8, len(selected_df))
    fig.add_vrect(x0=0.5, x1=wave_size + 0.5, fillcolor="rgba(178,210,53,0.18)", line_width=0)
    fig.update_layout(
        margin=dict(t=10, b=10, l=10, r=10),
        height=360,
        xaxis=dict(
            title="Recommendation Order",
            tickmode="array",
            tickvals=selected_df["selection_rank"],
            ticktext=selected_df["site_code"],
        ),
        yaxis=dict(title="Planning Budget ($M)"),
        yaxis2=dict(title="Risk Covered (%)", overlaying="y", side="right", range=[0, 100]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )

    first_wave_table = dash_table.DataTable(
        data=first_wave_df[["site_code", "site_name", "dominant_phase", "device_count", "planning_budget", "fte", "why_selected"]]
        .rename(columns={
            "site_code": "Site",
            "site_name": "Name",
            "dominant_phase": "Phase",
            "device_count": "Devices",
            "planning_budget": "Planning Budget",
            "fte": "FTE",
            "why_selected": "Why Selected",
        })
        .assign(**{
            "Planning Budget": lambda x: x["Planning Budget"].map(fmt_money),
            "FTE": lambda x: x["FTE"].map(lambda v: f"{v:.1f}"),
        })
        .to_dict("records"),
        columns=[
            {"name": "Site", "id": "Site"},
            {"name": "Name", "id": "Name"},
            {"name": "Phase", "id": "Phase"},
            {"name": "Devices", "id": "Devices"},
            {"name": "Planning Budget", "id": "Planning Budget"},
            {"name": "FTE", "id": "FTE"},
            {"name": "Why Selected", "id": "Why Selected"},
        ],
        style_table={"overflowX": "auto"},
        style_cell={"textAlign": "left", "padding": "8px", "fontSize": "13px"},
        style_header={"fontWeight": "bold", "backgroundColor": "#f8f9fa"},
        page_size=min(8, len(first_wave_df)),
    )

    portfolio_table = dash_table.DataTable(
        data=selected_df[["selection_rank", "site_code", "site_name", "state", "affiliate", "device_count", "planning_budget", "fte", "optimizer_score"]]
        .rename(columns={
            "selection_rank": "Rank",
            "site_code": "Site",
            "site_name": "Name",
            "state": "State",
            "affiliate": "Affiliate",
            "device_count": "Devices",
            "planning_budget": "Planning Budget",
            "fte": "FTE",
            "optimizer_score": "Score",
        })
        .assign(**{
            "Planning Budget": lambda x: x["Planning Budget"].map(fmt_money),
            "FTE": lambda x: x["FTE"].map(lambda v: f"{v:.1f}"),
            "Score": lambda x: x["Score"].map(lambda v: f"{v:.2f}"),
        })
        .to_dict("records"),
        columns=[{"name": c, "id": c} for c in ["Rank", "Site", "Name", "State", "Affiliate", "Devices", "Planning Budget", "FTE", "Score"]],
        style_table={"overflowX": "auto"},
        style_cell={"textAlign": "left", "padding": "8px", "fontSize": "13px"},
        style_header={"fontWeight": "bold", "backgroundColor": "#f8f9fa"},
        page_size=min(12, len(selected_df)),
    )
    full_portfolio = html.Details([
        html.Summary("Show full recommended portfolio", className="fw-semibold"),
        html.Div(portfolio_table, className="pt-3"),
    ])

    first_wave_budget = first_wave_df["planning_budget"].sum()
    first_wave_actual = first_wave_df["total_cost"].sum()
    first_wave_fte = first_wave_df["fte"].sum()
    first_wave_risk = first_wave_df["risk_covered_pct"].iloc[-1]

    exec_brief = html.Div([
        html.P(
            f"Under the current {objective_mode} settings, the optimizer selects {len(selected_df)} sites before hitting the planning caps.",
            className="mb-2",
        ),
        html.P(
            f"The suggested first wave contains {len(first_wave_df)} sites, covers {int(first_wave_df['device_count'].sum()):,} devices, uses {fmt_money(first_wave_budget)} in planning budget, and consumes {first_wave_fte:.1f} FTE.",
            className="mb-2",
        ),
        html.P(
            f"That first wave captures about {first_wave_risk:.1f}% of the modeled risk in the visible risk-focused fleet.",
            className="mb-0",
        ),
    ])

    pilot_df = selected_df.head(min(3, len(selected_df)))
    pilot_budget = pilot_df["planning_budget"].sum()
    pilot_fte = pilot_df["fte"].sum()
    final_recommendation = html.Div([
        html.P(
            f"Primary recommendation: approve the first wave of {len(first_wave_df)} sites now, with about {fmt_money(first_wave_actual)} in modeled refresh spend and {fmt_money(first_wave_budget)} in planning budget.",
            className="mb-3",
        ),
        html.P(
            f"Smaller pilot option: start with the first {len(pilot_df)} sites for about {fmt_money(pilot_budget)} in planning budget and {pilot_fte:.1f} FTE, then expand once delivery assumptions are validated.",
            className="mb-0",
        ),
    ])

    return kpis, fig, first_wave_table, full_portfolio, exec_brief, final_recommendation


# ---------------------------------------------------------------------------
# Callbacks: Fleet Assistant
# ---------------------------------------------------------------------------
@app.callback(
    Output("chat-collapse", "is_open"),
    [Input("chat-toggle-btn", "n_clicks"),
     Input("chat-close-btn", "n_clicks")],
    State("chat-collapse", "is_open"),
    prevent_initial_call=True,
)
def toggle_chat(toggle_clicks, close_clicks, is_open):
    ctx = callback_context
    if not ctx.triggered:
        return is_open
    triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]
    if triggered_id == "chat-close-btn":
        return False
    return not is_open


@app.callback(
    Output("chat-messages", "children"),
    Input("chat-history", "data"),
)
def render_chat_messages(history):
    if not history:
        return html.P("No messages yet.", className="text-muted")

    blocks = []
    for msg in history:
        is_user = msg.get("role") == "user"
        blocks.append(
            html.Div(
                msg.get("content", ""),
                className="mb-2 p-2 rounded",
                style={
                    "backgroundColor": "#EAF1FB" if is_user else "#F4F6F8",
                    "marginLeft": "24px" if is_user else "0",
                    "marginRight": "0" if is_user else "24px",
                    "whiteSpace": "pre-wrap",
                },
            )
        )
    return blocks


@app.callback(
    [Output("chat-history", "data"),
     Output("chat-input", "value")],
    [Input("chat-send-btn", "n_clicks"),
     Input("chat-input", "n_submit")],
    [State("chat-input", "value"),
     State("chat-history", "data"),
     State("filter-state", "value"),
     State("filter-affiliate", "value"),
     State("filter-device-type", "value"),
     State("filter-lifecycle", "value")],
    prevent_initial_call=True,
)
def send_chat_message(send_clicks, submit_clicks, user_text, history, states, affiliates, dtypes, lifecycle):
    if not user_text or not user_text.strip():
        return no_update, no_update

    history = history or []
    history = history + [{"role": "user", "content": user_text.strip()}]

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key or OpenAI is None:
        history.append({
            "role": "assistant",
            "content": "Fleet Assistant is unavailable because no OpenAI API key is configured.",
        })
        return history, ""

    full_df, risk_df = get_filtered_frames(states, affiliates, dtypes, lifecycle)
    summary = build_chat_summary(full_df, risk_df)

    client = OpenAI(api_key=api_key)
    try:
        response = client.responses.create(
            model="gpt-4o-mini",
            input=[
                {
                    "role": "system",
                    "content": (
                        "You answer questions about a Southern Company network lifecycle dashboard. "
                        "Only use the provided aggregated summary statistics. "
                        "Do not invent hostnames, serial numbers, IP addresses, or raw device records. "
                        "Keep answers concise and data-grounded."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Dashboard summary: {json.dumps(summary, default=float)}\n\nQuestion: {user_text.strip()}",
                },
            ],
        )
        answer = response.output_text.strip()
    except Exception as exc:  # pragma: no cover - runtime integration
        answer = f"Fleet Assistant could not answer right now: {exc}"

    history.append({"role": "assistant", "content": answer})
    return history, ""


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Southern Company Network Lifecycle Dashboard")
    print("=" * 60)
    print(f"Loaded {len(DF):,} devices")
    port = int(os.environ.get("PORT", 8050))
    print(f"Open http://localhost:{port} in your browser")
    print("=" * 60 + "\n")
    app.run(debug=False, host="0.0.0.0", port=port)
