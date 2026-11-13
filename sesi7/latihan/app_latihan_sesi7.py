"""Latihan callback Dash: filter, grafik, dan seleksi pelanggan."""

from pathlib import Path

import pandas as pd
import plotly.express as px
from dash import Dash, Input, Output, State, ctx, dash_table, dcc, html, no_update
from dash.exceptions import PreventUpdate

ROOT = Path(__file__).resolve().parents[2]
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
INTERNET_SERVICES = sorted(df["internet_service"].unique())
STORE_COLS = [*TABEL_COLS, "churn", "internet_service"]
TENURE_MIN = int(df["tenure_months"].min())
TENURE_MAX = int(df["tenure_months"].max())
TENURE_MARKS = {n: str(n) for n in (TENURE_MIN, 12, 24, 36, 48, 60, TENURE_MAX)}


def filter_df(region, segment, kontrak, tenure, min_keluhan, internet_service):
    filtered = df
    if region:
        filtered = filtered[filtered["region"].isin(region)]
    if segment:
        filtered = filtered[filtered["segment"].isin(segment)]
    if kontrak and kontrak != "Semua":
        filtered = filtered[filtered["contract_type"] == kontrak]
    services = INTERNET_SERVICES if internet_service is None else internet_service
    filtered = filtered[filtered["internet_service"].isin(services)]
    filtered = filtered[filtered["tenure_months"].between(*tenure)]
    return filtered[filtered["num_complaints"] >= (min_keluhan or 0)]


def kartu(judul, nilai):
    return html.Div(
        [html.Small(judul), html.H2(nilai, style={"margin": "4px 0"})],
        style=KARTU,
    )


app = Dash(__name__, title="Latihan Callback Sesi 7")
app.layout = html.Div(
    style={"maxWidth": "1200px", "margin": "auto", "fontFamily": "Arial"},
    children=[
        html.H1("Latihan Callback — Dashboard Churn"),
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
                html.Label("Layanan internet"),
                dcc.Checklist(
                    id="ck-internet",
                    options=INTERNET_SERVICES,
                    value=INTERNET_SERVICES,
                    inline=True,
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
        dcc.Store(id="store-filtered"),
        html.Div(id="kpi", style={"display": "flex", "gap": "12px", "margin": "16px 0"}),
        html.Div(
            style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "12px"},
            children=[
                dcc.Graph(id="g-bar"),
                dcc.Graph(id="g-scatter"),
                dcc.Graph(id="g-hist"),
            ],
        ),
        html.Div(
            id="scatter-detail",
            children="Klik satu titik scatter untuk melihat detail pelanggan.",
            style={"padding": "10px", "background": "#F4F6FA"},
        ),
        html.P(
            "Tip: klik batang region atau gunakan box/lasso pada scatter untuk memfilter tabel.",
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
    Output("ck-internet", "value"),
    Input("btn-terapkan", "n_clicks"),
    Input("btn-reset", "n_clicks"),
    State("in-keluhan", "value"),
    prevent_initial_call=True,
)
def terapkan_atau_reset(_n_apply, _n_reset, nilai):
    if ctx.triggered_id == "btn-reset":
        return 0, 0, None, None, "Semua", [TENURE_MIN, TENURE_MAX], INTERNET_SERVICES
    if nilai is None:
        raise PreventUpdate
    return nilai, no_update, no_update, no_update, no_update, no_update, no_update


# Hitung hasil filter sekali dan bagikan ke grafik, detail, serta tabel.
@app.callback(
    Output("store-filtered", "data"),
    Input("dd-region", "value"),
    Input("dd-segment", "value"),
    Input("rd-kontrak", "value"),
    Input("rs-tenure", "value"),
    Input("store-keluhan", "data"),
    Input("ck-internet", "value"),
)
def store_filtered(region, segment, kontrak, tenure, min_keluhan, internet_service):
    filtered = filter_df(region, segment, kontrak, tenure, min_keluhan, internet_service)
    return filtered[STORE_COLS].to_dict("records")


# Callback utama memperbarui KPI, bar, scatter, dan histogram dari hasil filter.
@app.callback(
    Output("kpi", "children"),
    Output("g-bar", "figure"),
    Output("g-scatter", "figure"),
    Output("info-keluhan", "children"),
    Output("g-hist", "figure"),
    Input("store-filtered", "data"),
    State("store-keluhan", "data"),
)
def update_dashboard(records, min_keluhan):
    filtered = pd.DataFrame.from_records(records or [], columns=STORE_COLS)
    info = f"Filter keluhan aktif: ≥ {min_keluhan or 0}"
    if filtered.empty:
        kosong = px.scatter(title="Tidak ada data untuk filter ini")
        return [html.Div("Data kosong", style=KARTU)], kosong, kosong, info, kosong

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
        custom_data=["customer_id"],
        title="Tenure vs tagihan",
        labels={"tenure_months": "Tenure (bulan)", "monthly_charges": "Tagihan (Rp)"},
    )
    scatter.update_layout(dragmode="select")
    histogram = px.histogram(
        filtered,
        x="tenure_months",
        color="status",
        nbins=30,
        barmode="overlay",
        opacity=0.7,
        color_discrete_map=WARNA,
        template="plotly_white",
        title="Distribusi tenure pelanggan terfilter",
        labels={"tenure_months": "Tenure (bulan)", "status": "Status"},
    )
    return kpis, bar, scatter, info, histogram


# Klik titik scatter menampilkan detail dari hasil filter yang sama.
@app.callback(
    Output("scatter-detail", "children"),
    Input("g-scatter", "clickData"),
    Input("store-filtered", "data"),
)
def show_customer_detail(click, records):
    prompt = "Klik satu titik scatter untuk melihat detail pelanggan."
    if not click or not click.get("points") or not records:
        return prompt
    custom_data = click["points"][0].get("customdata")
    if not custom_data:
        return prompt
    customer_id = custom_data[0]
    customer = next(
        (row for row in records if row["customer_id"] == customer_id), None
    )
    if customer is None:
        return prompt
    return html.Div(
        [
            html.Strong(f"Pelanggan {customer['customer_id']}"),
            html.Div(f"Region: {customer['region']}"),
            html.Div(f"Kontrak: {customer['contract_type']}"),
            html.Div(f"Jumlah keluhan: {customer['num_complaints']}"),
        ]
    )


def table_rows(records, bar_click, selected_data, trigger):
    """Terapkan interaksi terakhir pada bar atau scatter ke tabel."""
    filtered = pd.DataFrame.from_records(records or [], columns=STORE_COLS)
    title = "Detail pelanggan (semua hasil filter)"
    if trigger == "g-bar" and bar_click and bar_click.get("points"):
        region = bar_click["points"][0]["x"]
        filtered = filtered[filtered["region"] == region]
        title = f"Detail pelanggan — region {region}"
    elif trigger == "g-scatter" and selected_data is not None:
        selected_ids = {
            point["customdata"][0]
            for point in selected_data.get("points", [])
            if point.get("customdata")
        }
        filtered = filtered[filtered["customer_id"].isin(selected_ids)]
        title = f"Detail pelanggan — {len(filtered)} titik dipilih di scatter"
    return filtered[TABEL_COLS].to_dict("records"), title


# Klik bar atau seleksi box/lasso pada scatter menyaring tabel.
@app.callback(
    Output("tbl", "data"),
    Output("judul-tabel", "children"),
    Input("g-bar", "clickData"),
    Input("g-scatter", "selectedData"),
    Input("store-filtered", "data"),
)
def update_tabel(bar_click, selected_data, records):
    return table_rows(records, bar_click, selected_data, ctx.triggered_id)


if __name__ == "__main__":
    app.run(debug=False, port=8055)
