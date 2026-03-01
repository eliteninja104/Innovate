"""
Southern Company Network Equipment Lifecycle Dashboard
======================================================
Interactive web dashboard for network equipment lifecycle management.
Run with: python app.py
"""

import base64
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, html, dcc, callback_context, dash_table, Input, Output, State, ALL, MATCH, no_update
import dash_bootstrap_components as dbc
from sklearn.cluster import DBSCAN

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

BASE_DIR = Path(__file__).resolve().parent

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
    df = pd.read_csv(csv_path, low_memory=False)
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
            html.Div(
                html.Img(
                    src=logo_src,
                    alt="Southern Company logo",
                    style={
                        "width": "88px",
                        "height": "36px",
                        "objectFit": "cover",
                        "objectPosition": "left center",
                        "display": "block",
                    },
                ),
                className="mb-3",
            ) if logo_src else None,
            html.H5("Southern Company", className="fw-bold mb-0", style={"color": "#FFFFFF"}),
            html.Small("Network Lifecycle Dashboard", style={"color": "#FFFFFF", "opacity": "0.9"}),
        ], className="p-3 bg-dark"),

        html.Hr(className="my-0"),

        html.Div([
            html.Label("Navigation", className="fw-bold text-muted small mb-2"),
            dbc.Nav([
                dbc.NavLink([html.I(className="fas fa-tachometer-alt me-2"), "Executive Summary"],
                            href="#", id="nav-overview", active=True, className="nav-link-custom"),
                dbc.NavLink([html.I(className="fas fa-map-marked-alt me-2"), "Geographic Risk Map"],
                            href="#", id="nav-map", className="nav-link-custom"),
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
                dbc.NavLink([html.I(className="fas fa-sort-amount-up me-2"), "Refresh Planning"],
                            href="#", id="nav-priorities", className="nav-link-custom"),
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
def overview_page():
    return html.Div([
        html.H4("Executive Summary", className="mb-3"),
        html.Div(id="kpi-row"),
        dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody([
                html.H6("Device Distribution by Type", className="text-muted"),
                dcc.Graph(id="chart-device-types", config={"displayModeBar": False}),
            ]), className="shadow-sm"), md=4),
            dbc.Col(dbc.Card(dbc.CardBody([
                html.H6("Lifecycle Status Overview", className="text-muted"),
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
        dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody([
                html.H6("Top 15 Models by Count", className="text-muted"),
                dcc.Graph(id="chart-top-models", config={"displayModeBar": False}),
            ]), className="shadow-sm"), md=12),
        ]),
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


# ---------------------------------------------------------------------------
# Main Layout
# ---------------------------------------------------------------------------
app.layout = html.Div([
    dcc.Store(id="current-page", data="overview"),
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
])


# ---------------------------------------------------------------------------
# Callbacks: Navigation
# ---------------------------------------------------------------------------
NAV_IDS = ["nav-overview", "nav-map", "nav-timeline", "nav-proximity",
           "nav-cost", "nav-capacity", "nav-exceptions", "nav-priorities"]
PAGE_MAP = {
    "nav-overview": "overview",
    "nav-map": "map",
    "nav-timeline": "timeline",
    "nav-proximity": "proximity",
    "nav-cost": "cost",
    "nav-capacity": "capacity",
    "nav-exceptions": "exceptions",
    "nav-priorities": "priorities",
}


@app.callback(
    Output("current-page", "data"),
    [Input(nid, "n_clicks") for nid in NAV_IDS],
    prevent_initial_call=True,
)
def switch_page(*args):
    ctx = callback_context
    if not ctx.triggered:
        return "overview"
    triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]
    return PAGE_MAP.get(triggered_id, "overview")


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
        "overview": overview_page,
        "map": map_page,
        "timeline": timeline_page,
        "proximity": proximity_page,
        "cost": cost_page,
        "capacity": capacity_page,
        "exceptions": exceptions_page,
        "priorities": priorities_page,
    }
    return pages.get(page, overview_page)()


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
    parts = [f"{len(active):,} devices"]
    if excepted > 0:
        parts.append(f"{excepted} excepted")
    if states:
        parts.append(f"{len(states)} state(s)")
    if affiliates:
        parts.append(f"{len(affiliates)} affiliate(s)")
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
    if page != "overview":
        return [no_update] * 7

    df = apply_filters(DF, states, affiliates, dtypes, lifecycle)
    # Exclude exceptions
    df = df[df["exception_flagged"] != True]

    # KPIs
    total = len(df)
    past_eol = (df["lifecycle_status"] == "Past EoL").sum()
    past_eos = (df["lifecycle_status"] == "Past EoS").sum()
    total_cost = df["total_refresh_cost"].sum()
    critical = (df["risk_tier"] == "Critical").sum()
    sites = df["site_code"].nunique()

    kpi_row = dbc.Row([
        dbc.Col(make_kpi_card("Total Active Devices", f"{total:,}", "server", "primary"), md=2),
        dbc.Col(make_kpi_card("Past End-of-Life", f"{past_eol:,}", "exclamation-triangle", "danger"), md=2),
        dbc.Col(make_kpi_card("Past End-of-Sale", f"{past_eos:,}", "clock", "warning"), md=2),
        dbc.Col(make_kpi_card("Critical Risk", f"{critical:,}", "fire", "danger"), md=2),
        dbc.Col(make_kpi_card("Est. Refresh Cost", f"${total_cost:,.0f}", "dollar-sign", "info"), md=2),
        dbc.Col(make_kpi_card("Active Sites", f"{sites:,}", "building", "success"), md=2),
    ], className="mb-4")

    # Device types pie
    dt_counts = df["device_type"].value_counts()
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

    # Lifecycle pie
    lc_counts = df["lifecycle_status"].value_counts()
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
    risk_counts = df["risk_tier"].value_counts().reindex(["Low", "Medium", "High", "Critical"], fill_value=0)
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
    state_counts = df.groupby("state").agg(
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
    aff_counts = df.groupby("affiliate").agg(
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
    model_counts = df["model"].value_counts().head(15)
    model_lifecycle = df[df["model"].isin(model_counts.index)].groupby("model")["lifecycle_status"].apply(
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
    if page != "map":
        return [no_update] * 3

    df = apply_filters(DF, states, affiliates, dtypes, lifecycle)
    df = df[df["exception_flagged"] != True]
    geo = df[df["latitude"].notna() & df["longitude"].notna()].copy()

    # Aggregate by site
    site_agg = geo.groupby(["site_code", "latitude", "longitude", "site_name", "state", "county"]).agg(
        device_count=("device_id", "count"),
        avg_risk=("risk_score", "mean"),
        past_eol=("lifecycle_status", lambda x: (x == "Past EoL").sum()),
        past_eos=("lifecycle_status", lambda x: (x == "Past EoS").sum()),
        total_cost=("total_refresh_cost", "sum"),
    ).reset_index()

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
    ).sort_values("avg_risk", ascending=False).head(20).reset_index()
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

    df = apply_filters(DF, states, affiliates, dtypes, lifecycle)
    df = df[df["exception_flagged"] != True]

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

    df = apply_filters(DF, states, affiliates, dtypes, lifecycle)
    df = df[df["exception_flagged"] != True]
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
                marker=dict(size=12, color=color, opacity=0.8, line=dict(color="#003087", width=1)),
                name=f"Cluster {cluster_id + 1}",
                hovertext=[
                    f"{row.site_name}<br>Devices: {row.device_count}<br>Risk: {row.avg_risk:.0f}<br>Cost: ${row.total_cost:,.0f}"
                    for _, row in c.iterrows()
                ],
            ))

    fig.update_layout(
        map=dict(style="carto-positron", center=dict(lat=33.0, lon=-85.0), zoom=5),
        margin=dict(t=0, b=0, l=0, r=0),
        legend=dict(orientation="h"),
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
    if page != "cost":
        return [no_update] * 5

    df = apply_filters(DF, states, affiliates, dtypes, lifecycle)
    df = df[df["exception_flagged"] != True]

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
    fig_scatter.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=350)

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
    if page != "priorities":
        return [no_update] * 7

    df = apply_filters(DF, states, affiliates, dtypes, lifecycle)
    df = df[df["exception_flagged"] != True]

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
# Run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Southern Company Network Lifecycle Dashboard")
    print("=" * 60)
    print(f"Loaded {len(DF):,} devices")
    print(f"Open http://localhost:8050 in your browser")
    print("=" * 60 + "\n")
    app.run(debug=False, port=8050)
