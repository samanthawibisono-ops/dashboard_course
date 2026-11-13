"""Dashboard churn untuk latihan callback Dash."""

from pathlib import Path

import pandas as pd
import plotly.express as px
from dash import Dash, Input, Output, State, ctx, dash_table, dcc, html, no_update
from dash.exceptions import PreventUpdate

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / "data" / "customer_churn.csv", parse_dates=["signup_date"])
df["status"] = df["churn"].map({0: "Bertahan", 1: "Churn"})

WARNA = {"Bertahan": "#4C72B0", "Churn": "#C44E52"}
KARTU = {
    "flex": 1,
    "padding": "14px",
    "borderRadius": "10px",
    "background": "#F4F6FA",
    "textAlign": "center",
}
TABEL_COLS = [
    "customer_id",
    "region",
    "segment",
    "contract_type",
    "tenure_months",
    "monthly_charges",
    "num_complaints",
    "status",
]
TENURE_MIN = int(df["tenure_months"].min())
TENURE_MAX = int(df["tenure_months"].max())
TENURE_MARKS = {n: str(n) for n in (TENURE_MIN, 12, 24, 36, 48, 60, TENURE_MAX)}


def filter_df(region, segment, kontrak, tenure, min_keluhan):
    filtered = df
    if region:
        filtered = filtered[filtered["region"].isin(region)]
    if segment:
        filtered = filtered[filtered["segment"].isin(segment)]
    if kontrak and kontrak != "Semua":
        filtered = filtered[filtered["contract_type"] == kontrak]
    filtered = filtered[filtered["tenure_months"].between(*tenure)]
    return filtered[filtered["num_complaints"] >= (min_keluhan or 0)]


def kartu(judul, nilai):
    return html.Div(
        [html.Small(judul), html.H2(nilai, style={"margin": "4px 0"})],
        style=KARTU,
    )


app = Dash(__name__, title="Dash Callback")
app.layout = html.Div(
    style={"maxWidth": "1200px", "margin": "auto", "fontFamily": "Arial"},
    children=[
        html.H1("Dash Callback System — Dashboard Churn"),
        html.Div(
            style={"display": "grid", "gridTemplateColumns": "1fr 1fr 1fr", "gap": "16px"},
            children=[
                html.Div(
                    [
                        html.Label("Region"),
                        dcc.Dropdown(
                            id="dd-region",
                            options=sorted(df["region"].unique()),
                            multi=True,
                            placeholder="Semua region",
                        ),
                    ]
                ),
                html.Div(
                    [
                        html.Label("Segmen (opsi bergantung region)"),
                        dcc.Dropdown(
                            id="dd-segment", multi=True, placeholder="Semua segmen"
                        ),
                    ]
                ),
                html.Div(
                    [
                        html.Label("Jenis kontrak"),
                        dcc.RadioItems(
                            id="rd-kontrak",
                            inline=True,
                            value="Semua",
                            options=["Semua", "Bulanan", "Tahunan", "Dua Tahun"],
                        ),
                    ]
                ),
            ],
        ),
        html.Div(
            style={"marginTop": "12px"},
            children=[
                html.Label("Tenure (bulan)"),
                dcc.RangeSlider(
                    id="rs-tenure",
                    min=TENURE_MIN,
                    max=TENURE_MAX,
                    step=1,
                    value=[TENURE_MIN, TENURE_MAX],
                    marks=TENURE_MARKS,
                ),
            ],
        ),
        html.Div(
            style={"display": "flex", "gap": "8px", "alignItems": "center"},
            children=[
                html.Label("Minimal jumlah keluhan:"),
                dcc.Input(
                    id="in-keluhan",
                    type="number",
                    min=0,
                    max=10,
                    value=0,
                    style={"width": "70px"},
                ),
                html.Button("Terapkan", id="btn-terapkan", n_clicks=0),
                html.Button("Reset filter", id="btn-reset", n_clicks=0),
                html.Span(id="info-keluhan", style={"color": "#666"}),
            ],
        ),
        dcc.Store(id="store-keluhan", data=0),
        html.Div(id="kpi", style={"display": "flex", "gap": "12px", "margin": "16px 0"}),
        html.Div(
            style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "12px"},
            children=[dcc.Graph(id="g-bar"), dcc.Graph(id="g-scatter")],
        ),
        html.P(
            "Tip: klik salah satu batang region untuk memfilter tabel di bawah.",
            style={"color": "#666"},
        ),
        html.H3(id="judul-tabel"),
        dash_table.DataTable(
            id="tbl",
            page_size=10,
            sort_action="native",
            columns=[{"name": column, "id": column} for column in TABEL_COLS],
            style_cell={"padding": "6px", "textAlign": "left"},
            style_header={"fontWeight": "bold", "background": "#E9EDF5"},
            style_data_conditional=[
                {
                    "if": {"filter_query": "{status} = 'Churn'"},
                    "backgroundColor": "#FBE9EA",
                }
            ],
        ),
    ],
)


# Opsi segmen mengikuti region yang dipilih.
@app.callback(Output("dd-segment", "options"), Input("dd-region", "value"))
def update_segment_options(region):
    filtered = df[df["region"].isin(region)] if region else df
    return sorted(filtered["segment"].unique())


# Nilai keluhan diterapkan saat tombol diklik; Reset mengembalikan semua filter.
@app.callback(
    Output("store-keluhan", "data"),
    Output("in-keluhan", "value"),
    Output("dd-region", "value"),
    Output("dd-segment", "value"),
    Output("rd-kontrak", "value"),
    Output("rs-tenure", "value"),
    Input("btn-terapkan", "n_clicks"),
    Input("btn-reset", "n_clicks"),
    State("in-keluhan", "value"),
    prevent_initial_call=True,
)
def terapkan_atau_reset(_n_apply, _n_reset, nilai):
    if ctx.triggered_id == "btn-reset":
        return 0, 0, None, None, "Semua", [TENURE_MIN, TENURE_MAX]
    if nilai is None:
        raise PreventUpdate
    return nilai, no_update, no_update, no_update, no_update, no_update


# Filter utama memperbarui KPI dan kedua chart.
@app.callback(
    Output("kpi", "children"),
    Output("g-bar", "figure"),
    Output("g-scatter", "figure"),
    Output("info-keluhan", "children"),
    Input("dd-region", "value"),
    Input("dd-segment", "value"),
    Input("rd-kontrak", "value"),
    Input("rs-tenure", "value"),
    Input("store-keluhan", "data"),
)
def update_dashboard(region, segment, kontrak, tenure, min_keluhan):
    filtered = filter_df(region, segment, kontrak, tenure, min_keluhan)
    info = f"Filter keluhan aktif: ≥ {min_keluhan or 0}"
    if filtered.empty:
        kosong = px.scatter(title="Tidak ada data untuk filter ini")
        return [html.Div("Data kosong", style=KARTU)], kosong, kosong, info

    kpis = [
        kartu("Pelanggan", f"{len(filtered):,}"),
        kartu("Churn rate", f"{filtered['churn'].mean():.1%}"),
        kartu(
            "Rata-rata tagihan",
            f"Rp{filtered['monthly_charges'].mean():,.0f}".replace(",", "."),
        ),
        kartu("Rata-rata keluhan", f"{filtered['num_complaints'].mean():.2f}"),
    ]

    regional = (
        filtered.groupby("region", as_index=False)["churn"]
        .mean()
        .sort_values("churn", ascending=False)
    )
    bar = px.bar(
        regional,
        x="region",
        y="churn",
        template="plotly_white",
        title="Churn rate per region",
        labels={"churn": "Churn rate", "region": ""},
        color_discrete_sequence=["#C44E52"],
    )
    bar.update_yaxes(tickformat=".0%")
    scatter = px.scatter(
        filtered,
        x="tenure_months",
        y="monthly_charges",
        color="status",
        opacity=0.6,
        color_discrete_map=WARNA,
        template="plotly_white",
        hover_name="customer_id",
        title="Tenure vs tagihan",
        labels={"tenure_months": "Tenure (bulan)", "monthly_charges": "Tagihan (Rp)"},
    )
    return kpis, bar, scatter, info


# Klik bar region menyaring tabel detail.
@app.callback(
    Output("tbl", "data"),
    Output("judul-tabel", "children"),
    Input("g-bar", "clickData"),
    Input("dd-region", "value"),
    Input("dd-segment", "value"),
    Input("rd-kontrak", "value"),
    Input("rs-tenure", "value"),
    Input("store-keluhan", "data"),
)
def update_tabel(click, region, segment, kontrak, tenure, min_keluhan):
    filtered = filter_df(region, segment, kontrak, tenure, min_keluhan)
    judul = "Detail pelanggan (semua region terpilih)"
    if ctx.triggered_id == "g-bar" and click:
        selected_region = click["points"][0]["x"]
        filtered = filtered[filtered["region"] == selected_region]
        judul = f"Detail pelanggan — region {selected_region}"
    return filtered[TABEL_COLS].to_dict("records"), judul


if __name__ == "__main__":
    app.run(debug=False, port=8052)
