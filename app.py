# ============================================================
# app.py — AI Solutions Dashboard (With Project Deliverables)
# ============================================================

import json, base64, io, os, csv
from datetime import datetime

import pandas as pd
import numpy as np
import dash
from dash import dcc, html, Input, Output, State, no_update, ctx, ALL
import dash_bootstrap_components as dbc
from dash.dash_table import DataTable
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

from data_manager import (
    generate_sample_data, validate_data,
    COUNTRIES_CONTINENTS, SERVICE_PAGES, AGE_GROUPS, GENDERS,
)

# ============================================================
# 1.  APP INITIALISATION
# ============================================================

app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.DARKLY,
        "https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap"
    ],
    suppress_callback_exceptions=True,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)
app.title = "AI Solutions Dashboard"
server = app.server

# ============================================================
# 2.  CONSTANTS & COLOURS (Corporate Muted Palette)
# ============================================================

USERS = {
    "analyst": {"password": "analyst123", "role": "analyst", "name": "Sales Analyst"},
    "manager": {"password": "manager123", "role": "manager", "name": "Sales Manager"},
}

KPI_COLORS = {
    "total_visits":     "#4e79a7", "unique_visitors":  "#76b7b2",
    "demo_requests":    "#59a14f", "ai_assistant":     "#b07aa1",
    "promotion_events": "#f28e2b", "conversion_rate":  "#e15759",
}

CHART_COLORS = [
    '#4e79a7', '#f28e2b', '#e15759', '#76b7b2', '#59a14f', 
    '#edc948', '#b07aa1', '#ff9da7', '#9c755f', '#bab0ac'
]

_template_layout = {
    "paper_bgcolor": "rgba(0,0,0,0)", "plot_bgcolor":  "rgba(0,0,0,0)",
    "font": {"color": "#f8fafc", "family": "Inter, Segoe UI, Arial", "size": 11},
    "title": {"font": {"size": 14, "color": "#f8fafc"}},
    "xaxis": {"gridcolor": "#334155", "zerolinecolor": "#475569", "tickfont": {"color": "#94a3b8"}},
    "yaxis": {"gridcolor": "#334155", "zerolinecolor": "#475569", "tickfont": {"color": "#94a3b8"}},
    "legend": {"font": {"color": "#94a3b8"}, "bgcolor": "rgba(0,0,0,0)"},
    "margin": {"l": 48, "r": 20, "t": 65, "b": 42},
}
pio.templates["dashboard"] = go.layout.Template(layout=_template_layout)
pio.templates.default = "dashboard"

initial_df = generate_sample_data(5000)
initial_json_data = initial_df.to_json(date_format="iso", orient="split")

# ============================================================
# 3.  HELPER FUNCTIONS
# ============================================================

def load_data_from_store(data):
    if data is None: return None
    if isinstance(data, dict): data = json.dumps(data)
    df = pd.read_json(io.StringIO(data), orient="split")
    bool_map = {True: True, False: False, 1: True, 0: False, "True": True, "False": False, "true": True, "false": False, "1": True, "0": False}
    for col in ("request_demo", "ai_assistant", "promotion_event"):
        if col in df.columns: df[col] = df[col].map(bool_map).fillna(False).astype(bool)
    if "timestamp" in df.columns: df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df

def filter_dataframe(df, country, continent, gender, age_group, service_page, date_range):
    if df is None: return None
    f = df.copy()
    if country: f = f[f["country"].isin([country] if isinstance(country, str) else country)]
    if continent: f = f[f["continent"].isin([continent] if isinstance(continent, str) else continent)]
    if gender: f = f[f["gender"].isin([gender] if isinstance(gender, str) else gender)]
    if age_group: f = f[f["age_group"].isin([age_group] if isinstance(age_group, str) else age_group)]
    if service_page: f = f[f["service_page"].isin([service_page] if isinstance(service_page, str) else service_page)]
    if date_range and date_range[0] and date_range[1]:
        f = f[(f["timestamp"] >= pd.to_datetime(date_range[0])) & (f["timestamp"] <= pd.to_datetime(date_range[1]))]
    return f

def empty_fig(msg="No data loaded"):
    fig = go.Figure()
    fig.add_annotation(text=msg, xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False, font=dict(size=16, color="#94a3b8"))
    fig.update_layout(xaxis={"visible": False}, yaxis={"visible": False}, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    return fig

def simulate_geolocation(ip_list):
    countries = list(COUNTRIES_CONTINENTS.keys())
    return [np.random.choice(countries) for _ in ip_list]

# ============================================================
# 4.  CHART CREATION FUNCTIONS
# ============================================================

def chart_service_ranking(df):
    if df is None or df.empty: return empty_fig()
    counts = df.groupby("service_page").size().sort_values(ascending=True)
    fig = go.Figure(go.Bar(x=counts.values, y=counts.index, orientation="h", marker_color=CHART_COLORS[:len(counts)], text=counts.values, textposition="auto", textfont={"color": "#f8fafc"}, hovertemplate='<b>%{y}</b><br>Requests: %{x:,.0f}<extra></extra>'))
    fig.update_layout(title="Service Page Requests by Volume", xaxis_title="Requests", yaxis_title="", margin=dict(l=140, r=20, t=65, b=42))
    return fig

def chart_service_kpi_donut(df):
    if df is None or df.empty: return empty_fig()
    kpi = df.groupby("service_page").agg({"request_demo": "sum", "ai_assistant": "sum", "promotion_event": "sum"}).reset_index()
    fig = go.Figure()
    labels_map = {"request_demo": "Demo Requests", "ai_assistant": "AI Assistant", "promotion_event": "Promotions"}
    for i, (col, lbl) in enumerate(labels_map.items()):
        fig.add_trace(go.Pie(labels=kpi["service_page"], values=kpi[col], hole=0.55, name=lbl, visible=True if i == 0 else "legendonly", marker_colors=CHART_COLORS, hovertemplate='<b>%{label}</b><br>Count: %{value:,.0f}<br>Share: %{percent}<extra></extra>'))
    fig.update_layout(title="KPI Distribution by Service Page", showlegend=True, annotations=[dict(text="KPIs", x=0.5, y=0.5, font_size=14, showarrow=False, font_color="#f8fafc")]); return fig

def chart_country_kpi(df):
    if df is None or df.empty: return empty_fig()
    g = df.groupby("country").agg({"request_demo": "sum", "ai_assistant": "sum", "promotion_event": "sum"}).reset_index(); g["total"] = g[["request_demo","ai_assistant","promotion_event"]].sum(axis=1); g = g.sort_values("total", ascending=False).head(10)
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Demo Requests", x=g["country"], y=g["request_demo"], marker_color="#4e79a7", hovertemplate='Demo Requests: %{y:,.0f}<extra></extra>'))
    fig.add_trace(go.Bar(name="AI Assistant", x=g["country"], y=g["ai_assistant"], marker_color="#76b7b2", hovertemplate='AI Assistant: %{y:,.0f}<extra></extra>'))
    fig.add_trace(go.Bar(name="Promotions", x=g["country"], y=g["promotion_event"], marker_color="#f28e2b", hovertemplate='Promotions: %{y:,.0f}<extra></extra>'))
    fig.update_layout(title="Top 10 Countries by KPI", barmode="group", legend={"orientation":"h","y":1.12}); return fig

def chart_continent_kpi(df):
    if df is None or df.empty: return empty_fig()
    g = df.groupby("continent").agg({"request_demo": "sum", "ai_assistant": "sum", "promotion_event": "sum"}).reset_index()
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Demo Requests", x=g["continent"], y=g["request_demo"], marker_color="#4e79a7", hovertemplate='Demo Requests: %{y:,.0f}<extra></extra>'))
    fig.add_trace(go.Bar(name="AI Assistant", x=g["continent"], y=g["ai_assistant"], marker_color="#76b7b2", hovertemplate='AI Assistant: %{y:,.0f}<extra></extra>'))
    fig.add_trace(go.Bar(name="Promotions", x=g["continent"], y=g["promotion_event"], marker_color="#f28e2b", hovertemplate='Promotions: %{y:,.0f}<extra></extra>'))
    fig.update_layout(title="KPI Distribution by Continent", barmode="group", legend={"orientation":"h","y":1.12}); return fig

def chart_demo_trend(df):
    if df is None or df.empty: return empty_fig()
    c = df.copy(); c["month"] = c["timestamp"].dt.to_period("M").astype(str); m = c[c["request_demo"]].groupby("month").size().reset_index(name="demo_count")
    fig = go.Figure(go.Scatter(x=m["month"], y=m["demo_count"], mode="lines+markers", line={"color":"#4e79a7","width":2}, marker={"size":5}, fill="tozeroy", fillcolor="rgba(78,121,167,0.15)", hovertemplate='Demo Requests: %{y:,.0f}<extra></extra>'))
    fig.update_layout(title="Demo Requests Trend Over Time", xaxis_title="Month", yaxis_title="Demo Requests"); return fig

def chart_gender_kpi(df):
    if df is None or df.empty: return empty_fig()
    g = df.groupby("gender").agg({"request_demo": "sum", "ai_assistant": "sum", "promotion_event": "sum"}).reset_index()
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Demo Requests", x=g["gender"], y=g["request_demo"], marker_color="#4e79a7", hovertemplate='Demo Requests: %{y:,.0f}<extra></extra>'))
    fig.add_trace(go.Bar(name="AI Assistant", x=g["gender"], y=g["ai_assistant"], marker_color="#76b7b2", hovertemplate='AI Assistant: %{y:,.0f}<extra></extra>'))
    fig.add_trace(go.Bar(name="Promotions", x=g["gender"], y=g["promotion_event"], marker_color="#f28e2b", hovertemplate='Promotions: %{y:,.0f}<extra></extra>'))
    fig.update_layout(title="KPI Distribution by Gender", barmode="group", legend={"orientation":"h","y":1.12}); return fig

def chart_age_service(df):
    if df is None or df.empty: return empty_fig()
    g = df.groupby(["age_group","service_page"]).size().reset_index(name="count"); age_order = ["18-24","25-34","35-44","45-54","55+"]; fig = go.Figure()
    for i, page in enumerate(SERVICE_PAGES):
        p = g[g["service_page"] == page]; d = dict(zip(p["age_group"], p["count"])); fig.add_trace(go.Bar(name=page, x=age_order, y=[d.get(a, 0) for a in age_order], marker_color=CHART_COLORS[i % len(CHART_COLORS)], hovertemplate=f'<b>{page}</b><br>Count: '+'%{y:,.0f}<extra></extra>'))
    fig.update_layout(title="Age Group Distribution by Service Page", barmode="stack", legend={"orientation":"h","y":1.12}); return fig

def chart_peak_hour(df):
    if df is None or df.empty: return empty_fig()
    c = df.copy(); c["hour"] = c["timestamp"].dt.hour; h = c.groupby("hour").agg({"request_demo": "sum", "ai_assistant": "sum"}).reset_index(); fig = go.Figure()
    fig.add_trace(go.Bar(name="Demo Requests", x=h["hour"], y=h["request_demo"], marker_color="#4e79a7", hovertemplate='Hour: %{x}:00<br>Demos: %{y:,.0f}<extra></extra>'))
    fig.add_trace(go.Bar(name="AI Assistant", x=h["hour"], y=h["ai_assistant"], marker_color="#76b7b2", hovertemplate='Hour: %{x}:00<br>AI: %{y:,.0f}<extra></extra>'))
    fig.update_layout(title="Peak Engagement by Hour of Day", xaxis=dict(tickmode="linear",tick0=0,dtick=2), barmode="group", legend={"orientation":"h","y":1.12}); return fig

def chart_peak_day(df):
    if df is None or df.empty: return empty_fig()
    c = df.copy(); c["day"] = c["timestamp"].dt.day_name(); days = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]; h = c.groupby("day").agg({"request_demo": "sum", "ai_assistant": "sum"}).reindex(days, fill_value=0).reset_index(); fig = go.Figure()
    fig.add_trace(go.Bar(name="Demo Requests", x=h["day"], y=h["request_demo"], marker_color="#4e79a7", hovertemplate='Demos: %{y:,.0f}<extra></extra>'))
    fig.add_trace(go.Bar(name="AI Assistant", x=h["day"], y=h["ai_assistant"], marker_color="#76b7b2", hovertemplate='AI: %{y:,.0f}<extra></extra>'))
    fig.update_layout(title="Peak Engagement by Day of Week", barmode="group", legend={"orientation":"h","y":1.12}); return fig

def chart_peak_month(df):
    if df is None or df.empty: return empty_fig()
    c = df.copy(); c["month_name"] = c["timestamp"].dt.month_name(); months = ["January","February","March","April","May","June","July","August","September","October","November","December"]; h = c.groupby("month_name").agg({"request_demo": "sum", "ai_assistant": "sum"}).reindex(months, fill_value=0).reset_index(); fig = go.Figure()
    fig.add_trace(go.Scatter(name="Demo Requests", x=h["month_name"], y=h["request_demo"], mode="lines+markers", line={"color":"#4e79a7","width":2}, fill="tozeroy", fillcolor="rgba(78,121,167,0.15)", hovertemplate='Demos: %{y:,.0f}<extra></extra>'))
    fig.add_trace(go.Scatter(name="AI Assistant", x=h["month_name"], y=h["ai_assistant"], mode="lines+markers", line={"color":"#76b7b2","width":2}, fill="tozeroy", fillcolor="rgba(118,183,178,0.15)", hovertemplate='AI: %{y:,.0f}<extra></extra>'))
    fig.update_layout(title="Peak Engagement by Month", legend={"orientation":"h","y":1.12}); return fig

# ============================================================
# 5.  REUSABLE UI COMPONENTS
# ============================================================

def make_kpi_card(icon, value_id, label, color):
    return html.Div([html.Div(icon, className="kpi-icon"), html.Div(id=value_id, className="kpi-value", children="—"), html.Div(label, className="kpi-label")], className="kpi-card", style={"borderLeft": f"5px solid {color}", "backgroundColor": "#1e293b"})

def make_chart_card(chart_id):
    return dbc.Card([dbc.CardBody([dcc.Graph(id=chart_id, config={"displayModeBar": "hover", "responsive": True}, style={"height": "340px"})], style={"padding": "6px"})], className="chart-card")

# ============================================================
# 6.  LAYOUT — LOGIN
# ============================================================

login_overlay = html.Div(id="login-overlay", className="login-overlay", children=[
    html.Div([
        html.Img(src="/assets/logo.jpg", style={"height": "80px", "marginBottom": "20px", "objectFit": "contain"}),
        html.H4("AI Solutions Dashboard", className="text-center", style={"color": "#f8fafc", "fontWeight": "600", "marginTop": "0"}),
        html.Hr(style={"borderColor": "#4e79a7"}), 
        dbc.Label("Username", style={"color": "#94a3b8"}),
        dbc.Input(id="login-username", type="text", placeholder="Enter username", style={"backgroundColor": "#334155", "color": "#f8fafc", "borderColor": "#475569"}), html.Br(),
        dbc.Label("Password", style={"color": "#94a3b8"}), 
        dbc.Input(id="login-password", type="password", placeholder="Enter password", style={"backgroundColor": "#334155", "color": "#f8fafc", "borderColor": "#475569"}), html.Br(),
        html.Div(id="login-error", style={"color": "#e15759"}, className="mb-2"),
        dbc.Button("Login", id="login-btn", color="primary", className="w-100", style={"backgroundColor": "#4e79a7", "borderColor": "#4e79a7", "fontWeight": "600"}),
    ], className="login-card", style={"textAlign": "center"}),
])

# ============================================================
# 7.  LAYOUT — MAIN DASHBOARD
# ============================================================

header = html.Div([
    dbc.Row([
        dbc.Col([html.Img(src="/assets/logo.jpg", style={"height": "40px", "marginRight": "15px", "objectFit": "contain"}), html.H3("AI Solutions Dashboard", style={"color": "#f8fafc", "fontWeight": "700", "margin": "0"})], width="auto", className="d-flex align-items-center"),
        dbc.Col(width=True), 
        dbc.Col([
            dbc.Button(html.I(className="fas fa-bell", style={"fontSize": "16px"}), id="open-audit-btn", color="secondary", size="sm", className="me-2", style={"backgroundColor": "#334155", "borderColor": "#475569"}),
            html.Div(id="user-info", style={"color": "#94a3b8", "marginRight": "12px", "display": "inline-block"}),
            dbc.Button("Logout", id="logout-btn", size="sm", color="danger", style={"display": "none"}, className="me-2"),
        ], width="auto", className="d-flex align-items-center"),
    ], align="center"),
], className="header-bar mb-3")

upload_generate_row = dbc.Row([
    dbc.Col([dcc.Upload(id="upload-data", children=html.Div([html.I(className="fas fa-upload me-2"), "Upload CSV / Drag & Drop"]), style={"width": "100%", "height": "42px", "lineHeight": "42px", "borderWidth": "2px", "borderStyle": "dashed", "borderRadius": "8px", "textAlign": "center", "backgroundColor": "#334155", "borderColor": "#4e79a7", "color": "#94a3b8", "cursor": "pointer", "fontSize": "13px"}, multiple=False)], width=3, className="me-2"),
    dbc.Col([dbc.Button([html.I(className="fas fa-database me-2"), "Generate Logs"], id="generate-btn", color="info", size="md", style={"width": "100%", "backgroundColor": "#4e79a7", "borderColor": "#4e79a7", "fontWeight": "600"})], width=2, className="me-2"),
    dbc.Col([dbc.Button([html.I(className="fas fa-download me-2"), "Download Raw Logs"], id="download-raw-btn", color="success", size="md", style={"width": "100%", "backgroundColor": "#59a14f", "borderColor": "#59a14f", "fontWeight": "600"})], width=2, className="me-2"),
    dbc.Col([html.Div(id="data-status", className="data-status", style={"color": "#76b7b2"})], width=3, className="d-flex align-items-center"),
    dcc.Download(id="download-raw-logs"),
], className="mb-3", align="center")

# ============================================================
# FIXED KPI STRIP
# ============================================================

# FIXED FUNCTION
def make_kpi_card(value_id, label, color):
    return html.Div(
        [
            html.Div(
                id=value_id,
                className="kpi-value",
                children="0"
            ),

            html.Div(
                label,
                className="kpi-label"
            ),
        ],

        className="kpi-card",

        style={
            "borderTop": f"4px solid {color}",
            "backgroundColor": "#1e293b",
            "borderRadius": "12px",
            "padding": "18px",
            "height": "110px",
            "display": "flex",
            "flexDirection": "column",
            "justifyContent": "center",
            "boxShadow": "0 4px 12px rgba(0,0,0,0.25)",
        }
    )


# ============================================================
# FIXED KPI STRIP
# ============================================================

kpi_strip = dbc.Row(
    [

        dbc.Col(
            make_kpi_card(
                "kpi-total-visits",
                "Total Visits",
                KPI_COLORS["total_visits"]
            ),
            width=2
        ),

        dbc.Col(
            make_kpi_card(
                "kpi-unique-visitors",
                "Unique Visitors",
                KPI_COLORS["unique_visitors"]
            ),
            width=2
        ),

        dbc.Col(
            make_kpi_card(
                "kpi-demo-requests",
                "Demo Requests",
                KPI_COLORS["demo_requests"]
            ),
            width=2
        ),

        dbc.Col(
            make_kpi_card(
                "kpi-ai-assistant",
                "AI Assistant",
                KPI_COLORS["ai_assistant"]
            ),
            width=2
        ),

        dbc.Col(
            make_kpi_card(
                "kpi-promotion-events",
                "Promotion Events",
                KPI_COLORS["promotion_events"]
            ),
            width=2
        ),

        dbc.Col(
            make_kpi_card(
                "kpi-conversion-rate",
                "Conversion Rate",
                KPI_COLORS["conversion_rate"]
            ),
            width=2
        ),

    ],

    className="mb-3 g-2"
)

filter_sidebar = html.Div([
    html.H6("🔍  Filters", style={"color": "#f8fafc", "fontWeight": "700", "marginBottom": "12px"}),
    html.Label("Country"), dcc.Dropdown(id="filter-country", multi=True, clearable=True, placeholder="All Countries", style={"backgroundColor": "#334155", "color": "#0f172a"}),
    html.Label("Continent"), dcc.Dropdown(id="filter-continent", multi=True, clearable=True, placeholder="All Continents", style={"backgroundColor": "#334155", "color": "#0f172a"}),
    html.Label("Gender"), dcc.Dropdown(id="filter-gender", multi=True, clearable=True, placeholder="All Genders", options=[{"label": g, "value": g} for g in GENDERS], style={"backgroundColor": "#334155", "color": "#0f172a"}),
    html.Label("Age Group"), dcc.Dropdown(id="filter-age", multi=True, clearable=True, placeholder="All Age Groups", options=[{"label": a, "value": a} for a in AGE_GROUPS], style={"backgroundColor": "#334155", "color": "#0f172a"}),
    html.Label("Service Page"), dcc.Dropdown(id="filter-service", multi=True, clearable=True, placeholder="All Services", options=[{"label": s, "value": s} for s in SERVICE_PAGES], style={"backgroundColor": "#334155", "color": "#0f172a"}),
    html.Label("Date Range"), dcc.DatePickerRange(id="filter-date-range", start_date=None, end_date=None, display_format="YYYY-MM-DD", style={"width": "100%"}),
    html.Br(), dbc.Button("🔄  Reset Filters", id="reset-filters-btn", color="secondary", size="sm", style={"width": "100%", "marginTop": "8px", "backgroundColor": "#475569", "borderColor": "#475569"}),
], className="filter-sidebar")

visible_charts = dbc.Row([
    dbc.Col(make_chart_card("chart-service-ranking"),  width=6), dbc.Col(make_chart_card("chart-service-kpi-donut"),width=6),
    dbc.Col(make_chart_card("chart-country-kpi"),      width=6), dbc.Col(make_chart_card("chart-continent-kpi"),    width=6),
    dbc.Col(make_chart_card("chart-demo-trend"),       width=6), dbc.Col(make_chart_card("chart-gender-kpi"),       width=6),
], className="g-2")

collapsible_charts = dbc.Collapse([
    dbc.Row([dbc.Col(make_chart_card("chart-age-service"), width=6), dbc.Col(make_chart_card("chart-peak-hour"), width=6)], className="g-2 mt-2"),
    dbc.Row([dbc.Col(make_chart_card("chart-peak-day"), width=6), dbc.Col(make_chart_card("chart-peak-month"), width=6)], className="g-2 mt-1"),
], id="collapse-more-charts", is_open=False)

more_charts_toggle = html.Div([dbc.Button([html.Span("Show More Charts  "), html.I(id="collapse-icon", className="fas fa-chevron-down", style={"transition": "transform 0.3s"})], id="toggle-more-charts-btn", color="dark", size="sm", style={"width": "100%", "borderColor": "#334155", "color": "#94a3b8", "fontWeight": "600"}, className="mt-1 mb-2")], className="text-center")

dashboard_tab = dbc.Tab(label="📊 Dashboard", tab_id="tab-dashboard", children=[dcc.Loading(type="circle", color="#4e79a7", children=[visible_charts]), more_charts_toggle, dcc.Loading(type="circle", color="#4e79a7", children=[collapsible_charts])])

logs_tab = dbc.Tab(label="📋 Data Logs", tab_id="tab-logs", children=[
    html.Div([html.Hr(style={"borderColor": "#334155", "marginTop": "5px"}), html.P("Review the generated or uploaded log data below to verify data structure, format, and completeness.", style={"color": "#94a3b8", "fontSize": "13px"}),
        DataTable(id='data-table', page_size=20, style_table={'overflowX': 'auto', 'minWidth': '100%'}, style_header={'backgroundColor': '#1e293b', 'color': '#f8fafc', 'fontWeight': 'bold', 'border': '1px solid #334155'}, style_cell={'backgroundColor': '#0f172a', 'color': '#f8fafc', 'border': '1px solid #334155', 'textAlign': 'left', 'padding': '8px', 'fontFamily': 'Inter, Segoe UI, Arial', 'fontSize': '12px'}, style_data_conditional=[{'if': {'row_index': 'odd'}, 'backgroundColor': '#1e293b'}])], style={"marginTop": "10px"})
])

dict_tab = dbc.Tab(label="📖 Data Dictionary", tab_id="tab-dict", children=[
    html.Div([html.Hr(style={"borderColor": "#334155", "marginTop": "5px"}), html.P("Field mapping document explaining the log data structure, types, and valid values.", style={"color": "#94a3b8", "fontSize": "13px"}),
        DataTable(
            data=[
                {'Field': 'timestamp', 'Type': 'Datetime', 'Description': 'Date and time of the user request (YYYY-MM-DD HH:MM:SS)', 'Required': 'Yes', 'Valid Values': 'Valid ISO datetime'},
                {'Field': 'ip_address', 'Type': 'String', 'Description': 'IPv4 address of the visitor', 'Required': 'Yes', 'Valid Values': 'e.g., 192.168.1.1'},
                {'Field': 'visitor_id', 'Type': 'String', 'Description': 'Unique identifier for the visitor session', 'Required': 'Yes', 'Valid Values': 'e.g., V-1A2B3C4D'},
                {'Field': 'country', 'Type': 'String', 'Description': 'Country of origin (Resolved via IP Geolocation if missing)', 'Required': 'Yes*', 'Valid Values': 'e.g., United Kingdom'},
                {'Field': 'continent', 'Type': 'String', 'Description': 'Continent of origin', 'Required': 'Yes*', 'Valid Values': 'e.g., Europe'},
                {'Field': 'gender', 'Type': 'String', 'Description': 'Gender demographic of the visitor', 'Required': 'Yes', 'Valid Values': 'Male, Female, Other'},
                {'Field': 'age_group', 'Type': 'String', 'Description': 'Age demographic of the visitor', 'Required': 'Yes', 'Valid Values': '18-24, 25-34, 35-44, 45-54, 55+'},
                {'Field': 'service_page', 'Type': 'String', 'Description': 'The web service page requested', 'Required': 'Yes', 'Valid Values': 'AI Solutions, Cloud Services, Data Analytics, Cybersecurity, IoT Platform, Digital Marketing'},
                {'Field': 'request_demo', 'Type': 'Boolean', 'Description': 'Did the user request a product demonstration?', 'Required': 'Yes', 'Valid Values': 'True, False'},
                {'Field': 'ai_assistant', 'Type': 'Boolean', 'Description': 'Did the user interact with the AI Assistant?', 'Required': 'Yes', 'Valid Values': 'True, False'},
                {'Field': 'promotion_event', 'Type': 'Boolean', 'Description': 'Did the user trigger a promotional event?', 'Required': 'Yes', 'Valid Values': 'True, False'},
            ],
            style_table={'overflowX': 'auto', 'minWidth': '100%'}, style_header={'backgroundColor': '#1e293b', 'color': '#f8fafc', 'fontWeight': 'bold', 'border': '1px solid #334155'}, style_cell={'backgroundColor': '#0f172a', 'color': '#f8fafc', 'border': '1px solid #334155', 'textAlign': 'left', 'padding': '8px', 'fontFamily': 'Inter, Segoe UI, Arial', 'fontSize': '12px'}, style_data_conditional=[{'if': {'row_index': 'odd'}, 'backgroundColor': '#1e293b'}],
            style_cell_conditional=[{'if': {'column_id': 'Description'}, 'width': '40%'}]
        )
    ], style={"marginTop": "10px"})
])

main_tabs = dbc.Tabs([dashboard_tab, logs_tab, dict_tab], id="main-tabs", active_tab="tab-dashboard", style={"marginBottom": "20px"})

footer_bar = html.Div([
    dbc.Row([
        dbc.Col([html.Span("Actions & Export", style={"color": "#f8fafc", "fontWeight": "600", "marginRight": "20px"})], width="auto"),
        dbc.Col([
            dbc.Button([html.I(className="fas fa-image me-1"), "Export Dashboard (Print / PDF)"], id="export-charts-btn", color="info", size="sm", className="me-2", style={"backgroundColor": "#4e79a7", "borderColor": "#4e79a7"}), 
            dbc.Button([html.I(className="fas fa-filter me-1"), "Download Filtered Data"], id="download-csv-btn", color="success", size="sm", className="me-2", style={"backgroundColor": "#59a14f", "borderColor": "#59a14f"}), 
            dbc.Button([html.I(className="fas fa-chart-bar me-1"), "Stats Report"], id="download-stats-btn", color="warning", size="sm", style={"backgroundColor": "#f28e2b", "borderColor": "#f28e2b"}),
            dcc.Download(id="download-csv"), dcc.Download(id="download-stats")
        ], width="auto"), dbc.Col(width=True),
        dbc.Col([html.Span(id="last-updated", style={"color": "#64748b", "fontSize": "12px"})], width="auto", className="d-flex align-items-center"),
    ], align="center"),
], className="footer-bar")

html.Div(id="print-trigger", style={"display": "none"})

audit_modal = dbc.Modal([
    dbc.ModalHeader(dbc.ModalTitle("🔔 System Activity & Test Log", style={"color": "#f8fafc"})),
    dbc.ModalBody(html.Div(id="audit-log-display", style={"maxHeight": "400px", "overflowY": "auto"})),
    dbc.ModalFooter(dbc.Button("Close", id="close-audit-btn", className="ms-auto", color="secondary")),
], id="audit-modal", size="lg", style={"color": "#f8fafc"})

# ============================================================
# 8.  COMPLETE APP LAYOUT
# ============================================================

app.layout = html.Div([
    login_overlay, 
    dcc.Store(id="data-store", storage_type="session", data=initial_json_data), 
    dcc.Store(id="filtered-store", storage_type="memory"), 
    dcc.Store(id="auth-store", storage_type="session", data={"logged_in": False, "role": None, "name": None}),
    dcc.Store(id="audit-store", storage_type="session", data=[]),
    html.Div(id="main-app", style={"display": "none"}, children=[header, dbc.Container([upload_generate_row, dcc.Loading(type="circle", color="#4e79a7", children=[kpi_strip]), dbc.Row([dbc.Col(filter_sidebar, width=2, className="pe-1"), dbc.Col(main_tabs, width=10)], className="mt-2"), footer_bar], fluid=True)]),
    dbc.Alert(id="validation-alert", is_open=False, duration=6000, style={"position": "fixed", "bottom": "20px", "right": "20px", "zIndex": "9999", "width": "380px"}),
    audit_modal,
])

# ============================================================
# 9.  CALLBACKS
# ============================================================

def add_audit_entry(current_log, action, details=""):
    entry = {"timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'), "action": action, "details": details}
    current_log.insert(0, entry)
    return current_log[:50]

@app.callback([Output("login-overlay", "style"), Output("main-app", "style"), Output("auth-store", "data"), Output("login-error", "children"), Output("user-info", "children"), Output("logout-btn", "style"), Output("audit-store", "data", allow_duplicate=True)], [Input("login-btn", "n_clicks"), Input("logout-btn", "n_clicks")], [State("login-username", "value"), State("login-password", "value"), State("auth-store", "data"), State("audit-store", "data")], prevent_initial_call=True)
def handle_login(l_clicks, lo_clicks, username, password, auth, audit):
    triggered = ctx.triggered_id
    if triggered == "logout-btn": return {"display": "flex"}, {"display": "none"}, {"logged_in": False, "role": None, "name": None}, "", "", {"display": "none"}, add_audit_entry(audit, "LOGOUT", f"User logged out")
    if not username or not password: return no_update, no_update, no_update, "Please enter both username and password.", no_update, no_update, no_update
    user = USERS.get(username)
    if user and user["password"] == password: 
        role_label = " (View Only)" if user["role"] == "analyst" else " (Full Control)"
        return {"display": "none"}, {"display": "block"}, {"logged_in": True, "role": user["role"], "name": user["name"]}, "", f"👤 {user['name']}{role_label}", {"display": "inline-block"}, add_audit_entry(audit, "LOGIN", f"{user['name']} logged in")
    return no_update, no_update, no_update, "Invalid credentials. Try again.", no_update, no_update, add_audit_entry(audit, "LOGIN_FAIL", f"Failed login attempt for '{username}'")

@app.callback([Output("generate-btn", "disabled"), Output("export-charts-btn", "disabled"), Output("download-csv-btn", "disabled"), Output("download-raw-btn", "disabled"), Output("download-stats-btn", "disabled")], [Input("auth-store", "data")])
def enforce_roles(auth):
    if auth is None or not auth.get("logged_in"): return True, True, True, True, True
    if auth.get("role") == "analyst": return True, True, True, True, True
    return False, False, False, False, False

@app.callback([Output("data-store", "data", allow_duplicate=True), Output("data-status", "children", allow_duplicate=True), Output("audit-store", "data", allow_duplicate=True)], [Input("generate-btn", "n_clicks")], [State("auth-store", "data"), State("audit-store", "data")], prevent_initial_call=True)
def generate_data(n, auth, audit):
    if not auth or auth.get("role") != "manager": return no_update, html.Span("⛔ Only managers can generate data.", style={"color": "#e15759"}), no_update
    if n is None: return no_update, no_update, no_update
    df = generate_sample_data(5000)
    return df.to_json(date_format="iso", orient="split"), html.Span([html.I(className="fas fa-check-circle me-1"), f"Generated {len(df):,} unique logs successfully"], style={"color": "#59a14f"}), add_audit_entry(audit, "GENERATE", f"Generated {len(df):,} records")

@app.callback(
    [Output("data-store", "data", allow_duplicate=True), Output("data-status", "children", allow_duplicate=True), Output("validation-alert", "children"), Output("validation-alert", "color"), Output("validation-alert", "is_open"), Output("main-tabs", "active_tab"), Output("audit-store", "data", allow_duplicate=True)], 
    [Input("upload-data", "contents")], [State("upload-data", "filename"), State("auth-store", "data"), State("audit-store", "data")], prevent_initial_call=True
)
def upload_data(contents, filename, auth, audit):
    if contents is None: return no_update, no_update, no_update, no_update, no_update, no_update, no_update
    if not auth or auth.get("role") != "manager": return no_update, html.Span("⛔ Only managers can upload data.", style={"color": "#e15759"}), no_update, no_update, no_update, no_update, add_audit_entry(audit, "UPLOAD_FAIL", "Permission denied")
    try:
        decoded = base64.b64decode(contents.split(",")[1])
        df = pd.read_csv(io.StringIO(decoded.decode("utf-8")))
    except Exception as e: 
        return no_update, no_update, f"Error reading file: {str(e)}", "danger", True, "tab-logs", add_audit_entry(audit, "UPLOAD_ERROR", f"File read error: {filename}")
    
    if len(df) > 50000: return no_update, html.Span("⚠️ File too large.", style={"color": "#e15759"}), "Max 50,000 rows.", "warning", True, "tab-logs", add_audit_entry(audit, "UPLOAD_FAIL", f"File too large: {filename}")

    if 'country' not in df.columns and 'ip_address' in df.columns:
        df['country'] = simulate_geolocation(df['ip_address'])
        df['continent'] = df['country'].map(COUNTRIES_CONTINENTS)

    is_valid, errors, warnings = validate_data(df)
    if not is_valid: 
        return (no_update, html.Span("❌ Invalid data", style={"color": "#e15759"}), "❌ Validation failed:\n" + "\n".join(f"• {e}" for e in errors), "danger", True, "tab-logs", add_audit_entry(audit, "VALIDATION_FAIL", f"Invalid data in {filename}"))
        
    return (df.to_json(date_format="iso", orient="split"), html.Span([html.I(className="fas fa-check-circle me-1"), f"Loaded {len(df):,} records from {filename}"], style={"color": "#59a14f"}), f"✅ Data loaded: {len(df):,} records from {filename}", "success", True, "tab-dashboard", add_audit_entry(audit, "UPLOAD_SUCCESS", f"Loaded {filename} ({len(df):,} rows)"))

@app.callback([Output("filter-country", "options"), Output("filter-continent", "options"), Output("filter-country", "value"), Output("filter-continent", "value"), Output("filter-gender", "value"), Output("filter-age", "value"), Output("filter-service", "value"), Output("filter-date-range", "start_date"), Output("filter-date-range", "end_date")], [Input("data-store", "data"), Input("reset-filters-btn", "n_clicks")])
def update_filters_and_reset(data, reset_clicks):
    if data is None: return [], [], None, None, None, None, None, None, None
    df = load_data_from_store(data)
    if df is None: return [], [], None, None, None, None, None, None, None
    return ([{"label": c, "value": c} for c in sorted(df["country"].dropna().unique())], [{"label": c, "value": c} for c in sorted(df["continent"].dropna().unique())], None, None, None, None, None, None, None)

@app.callback(Output("filtered-store", "data"), [Input("data-store", "data"), Input("filter-country", "value"), Input("filter-continent", "value"), Input("filter-gender", "value"), Input("filter-age", "value"), Input("filter-service", "value"), Input("filter-date-range", "start_date"), Input("filter-date-range", "end_date")])
def update_filtered_store(data, country, continent, gender, age, service, sd, ed):
    if data is None: return None
    df = load_data_from_store(data); df = filter_dataframe(df, country, continent, gender, age, service, [sd, ed])
    if df is None or df.empty: return None
    return df.to_json(date_format="iso", orient="split")

@app.callback([Output("kpi-total-visits", "children"), Output("kpi-unique-visitors", "children"), Output("kpi-demo-requests", "children"), Output("kpi-ai-assistant", "children"), Output("kpi-promotion-events", "children"), Output("kpi-conversion-rate", "children"), Output("last-updated", "children")], [Input("filtered-store", "data")])
def update_kpis(filtered_data):
    if filtered_data is None: return "0", "0", "0", "0", "0", "0%", ""
    df = load_data_from_store(filtered_data)
    if df is None or df.empty: return "0", "0", "0", "0", "0", "0%", ""
    total = len(df); unique = df["visitor_id"].nunique() if "visitor_id" in df else total
    demos = int(df.get("request_demo", 0).sum()); ai = int(df.get("ai_assistant", 0).sum()); promo = int(df.get("promotion_event", 0).sum())
    conv = (demos / total * 100) if total else 0
    fmt = lambda n: f"{n/1_000_000:.1f}M" if n >= 1_000_000 else (f"{n/1_000:.1f}K" if n >= 1_000 else str(n))
    return fmt(total), fmt(unique), fmt(demos), fmt(ai), fmt(promo), f"{conv:.1f}%", f"Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

@app.callback([Output("chart-service-ranking", "figure"), Output("chart-service-kpi-donut", "figure"), Output("chart-country-kpi", "figure"), Output("chart-continent-kpi", "figure"), Output("chart-demo-trend", "figure"), Output("chart-gender-kpi", "figure"), Output("chart-age-service", "figure"), Output("chart-peak-hour", "figure"), Output("chart-peak-day", "figure"), Output("chart-peak-month", "figure")], [Input("filtered-store", "data")])
def update_all_charts(filtered_data):
    if filtered_data is None: return [empty_fig()] * 10
    df = load_data_from_store(filtered_data)
    return [chart_service_ranking(df), chart_service_kpi_donut(df), chart_country_kpi(df), chart_continent_kpi(df), chart_demo_trend(df), chart_gender_kpi(df), chart_age_service(df), chart_peak_hour(df), chart_peak_day(df), chart_peak_month(df)]

@app.callback([Output("data-table", "data"), Output("data-table", "columns")], [Input("filtered-store", "data")])
def update_data_table(filtered_data):
    if filtered_data is None: return [], []
    df = load_data_from_store(filtered_data)
    if df is None or df.empty: return [], []
    df_display = df.copy(); df_display['timestamp'] = df_display['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
    for col in ['request_demo', 'ai_assistant', 'promotion_event']:
        if col in df_display.columns: df_display[col] = df_display[col].map({True: '✅ True', False: '❌ False'})
    return df_display.to_dict('records'), [{"name": col.replace('_', ' ').title(), "id": col} for col in df_display.columns]

@app.callback([Output("collapse-more-charts", "is_open"), Output("collapse-icon", "className")], [Input("toggle-more-charts-btn", "n_clicks")], [State("collapse-more-charts", "is_open")], prevent_initial_call=True)
def toggle_more_charts(n, is_open): return (not is_open, "fas fa-chevron-up") if n else (is_open, "fas fa-chevron-down")

app.clientside_callback("function(n) { if(n>0) window.print(); return ''; }", Output("print-trigger", "children"), [Input("export-charts-btn", "n_clicks")], prevent_initial_call=True)

# FIX: Removed invalid allow_duplicate from State
@app.callback(Output("download-raw-logs", "data"), [Input("download-raw-btn", "n_clicks")], [State("data-store", "data"), State("auth-store", "data")], prevent_initial_call=True)
def download_raw_logs(n, data, auth):
    if not auth or auth.get("role") != "manager" or n is None or data is None: return no_update
    df = load_data_from_store(data)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return dcc.send_data_frame(df.to_csv, f"raw_logs_{ts}.csv", index=False)

@app.callback(Output("download-csv", "data"), [Input("download-csv-btn", "n_clicks")], [State("filtered-store", "data"), State("auth-store", "data")], prevent_initial_call=True)
def download_csv(n, filtered_data, auth):
    if not auth or auth.get("role") != "manager" or n is None or filtered_data is None: return no_update
    df = load_data_from_store(filtered_data)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return dcc.send_data_frame(df.to_csv, f"filtered_data_{ts}.csv", index=False)

@app.callback(Output("download-stats", "data"), [Input("download-stats-btn", "n_clicks")], [State("filtered-store", "data"), State("auth-store", "data")], prevent_initial_call=True)
def download_stats_report(n, filtered_data, auth):
    if not auth or auth.get("role") != "manager" or n is None or filtered_data is None: return no_update
    df = load_data_from_store(filtered_data)
    if df.empty: return no_update
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["=== DESCRIPTIVE STATISTICS (Task 5.3) ==="])
    desc = df.describe(include='all').fillna('').reset_index()
    writer.writerow(desc.columns.tolist())
    for _, row in desc.iterrows(): writer.writerow(row.tolist())
    writer.writerow([])
    writer.writerow(["=== SERVICE USAGE RANKING (Task 5.4) ==="])
    service_rank = df.groupby('service_page').size().reset_index(name='request_count').sort_values('request_count', ascending=False)
    writer.writerow(service_rank.columns.tolist())
    for _, row in service_rank.iterrows(): writer.writerow(row.tolist())
    writer.writerow([])
    writer.writerow(["=== PEAK ENGAGEMENT - TEMPORAL (Task 5.5) ==="])
    c = df.copy(); c['hour'] = c['timestamp'].dt.hour
    peak_hour = c.groupby('hour').agg(demos=('request_demo','sum'), ai=('ai_assistant','sum')).reset_index()
    writer.writerow(['Hour', 'Demo Requests', 'AI Assistant'])
    for _, row in peak_hour.iterrows(): writer.writerow(row.tolist())
    writer.writerow([])
    writer.writerow(["=== DEMOGRAPHICS SEGMENTATION (Task 5.6) ==="])
    demo_seg = df.groupby(['age_group','gender']).size().reset_index(name='count')
    writer.writerow(demo_seg.columns.tolist())
    for _, row in demo_seg.iterrows(): writer.writerow(row.tolist())
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output.seek(0)
    return dcc.send_string(output.getvalue(), f"stats_report_{ts}.csv", type="text/csv")

@app.callback(Output("audit-modal", "is_open"), [Input("open-audit-btn", "n_clicks"), Input("close-audit-btn", "n_clicks")], [State("audit-modal", "is_open")], prevent_initial_call=True)
def toggle_audit_modal(o1, o2, is_open): return not is_open

@app.callback(Output("audit-log-display", "children"), [Input("audit-modal", "is_open")], [State("audit-store", "data")])
def render_audit_log(is_open, data):
    if not is_open or not data: return html.P("No activity recorded yet.", style={"color":"#94a3b8"})
    items = []
    for entry in data:
        color = "#59a14f" if "SUCCESS" in entry['action'] or "LOGIN" in entry['action'] else ("#e15759" if "FAIL" in entry['action'] or "ERROR" in entry['action'] else "#f8fafc")
        items.append(html.Li([html.Span(entry['timestamp'], style={"color":"#64748b","marginRight":"10px"}), html.B(entry['action'], style={"color":color,"marginRight":"10px"}), html.Span(entry['details'], style={"color":"#a0b4c8"})], style={"marginBottom":"8px", "listStyleType":"none"}))
    return html.Ul(items)

if __name__ == "__main__":
    app.run(debug=True, port=8050)