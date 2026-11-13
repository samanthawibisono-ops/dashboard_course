"""Dashboard Plotly dengan filter yang memperbarui seluruh grafik."""

from pathlib import Path

import pandas as pd
import plotly.express as px
from dash import Dash, Input, Output, dcc, html

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / "data" / "customer_churn.csv", parse_dates=["signup_date"])
df["status"] = df["churn"].map({0: "Bertahan", 1: "Churn"})

WARNA = {"Bertahan": "#4C72B0", "Churn": "#C44E52"}
REGIONS = sorted(df["region"].unique())
CONTRACTS = sorted(df["contract_type"].unique())
TENURE_MIN = int(df["tenure_months"].min())
TENURE_MAX = int(df["tenure_months"].max())


def make_figures(data):
    hist = px.histogram(
        data,
        x="tenure_months",
        color="status",
        nbins=36,
        barmode="overlay",
        opacity=0.7,
        color_discrete_map=WARNA,
        template="plotly_white",
        labels={"tenure_months": "Tenure (bulan)", "status": "Status"},
        title="Churn terkonsentrasi pada tenure pendek",
    )
    scatter = px.scatter(
        data,
        x="tenure_months",
        y="monthly_charges",
        color="status",
        color_discrete_map=WARNA,
        opacity=0.55,
        template="plotly_white",
        hover_name="customer_id",
        hover_data=["region", "contract_type", "num_complaints"],
        labels={"tenure_months": "Tenure (bulan)", "monthly_charges": "Tagihan (Rp)"},
        title="Tenure vs tagihan pelanggan",
    )

    bar_df = data.groupby(["region", "contract_type"], as_index=False).agg(
        pelanggan=("customer_id", "count"),
        pelanggan_churn=("churn", "sum"),
        churn_rate=("churn", "mean"),
    )
    bar = px.bar(
        bar_df,
        x="region",
        y="pelanggan_churn",
        color="contract_type",
        barmode="group",
        template="plotly_white",
        hover_data={"pelanggan": True, "churn_rate": ":.1%"},
        labels={
            "pelanggan_churn": "Pelanggan churn",
            "region": "Region",
            "contract_type": "Kontrak",
            "pelanggan": "Total pelanggan",
            "churn_rate": "Churn rate",
        },
        title="Pelanggan churn per region dan kontrak",
    )
    grouped_offsets = [trace.name for trace in bar.data]
    stacked_offsets = ["stack"] * len(bar.data)
    bar.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                direction="right",
                x=1,
                y=1.15,
                buttons=[
                    dict(
                        label="Grouped",
                        method="update",
                        args=[{"offsetgroup": grouped_offsets}, {"barmode": "group"}],
                    ),
                    dict(
                        label="Stacked",
                        method="update",
                        args=[{"offsetgroup": stacked_offsets}, {"barmode": "stack"}],
                    ),
                ],
            )
        ]
    )
    return hist, scatter, bar


initial_hist, initial_scatter, initial_bar = make_figures(df)
app = Dash(__name__, title="Plotly di Dash")
app.layout = html.Div(
    style={"maxWidth": "1150px", "margin": "auto", "fontFamily": "Arial"},
    children=[
        html.H1("Interactive Visualization with Plotly"),
        html.P("Pilih filter untuk memperbarui semua grafik. Klik legenda untuk menyembunyikan kelompok."),
        html.Div(
            style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "16px"},
            children=[
                html.Div([
                    html.Label("Region"),
                    dcc.Dropdown(
                        id="region-filter",
                        options=[{"label": value, "value": value} for value in REGIONS],
                        value=REGIONS,
                        multi=True,
                    ),
                ]),
                html.Div([
                    html.Label("Jenis kontrak"),
                    dcc.Dropdown(
                        id="contract-filter",
                        options=[{"label": value, "value": value} for value in CONTRACTS],
                        value=CONTRACTS,
                        multi=True,
                    ),
                ]),
            ],
        ),
        html.Div(
            style={"marginTop": "20px", "marginBottom": "16px"},
            children=[
                html.Label("Tenure (bulan)"),
                dcc.RangeSlider(
                    id="tenure-filter",
                    min=TENURE_MIN,
                    max=TENURE_MAX,
                    step=1,
                    value=[TENURE_MIN, TENURE_MAX],
                    marks={TENURE_MIN: str(TENURE_MIN), TENURE_MAX: str(TENURE_MAX)},
                ),
            ],
        ),
        html.P(id="filter-count", children=f"{len(df):,} pelanggan ditampilkan"),
        html.Div(
            style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "12px"},
            children=[
                dcc.Graph(id="histogram-graph", figure=initial_hist),
                dcc.Graph(id="scatter-graph", figure=initial_scatter),
            ],
        ),
        dcc.Graph(
            id="bar-graph",
            figure=initial_bar,
            config={"displaylogo": False, "toImageButtonOptions": {"format": "png", "scale": 2}},
        ),
        html.P("Pilih setidaknya dua jenis kontrak untuk melihat perbedaan Grouped dan Stacked."),
    ],
)


@app.callback(
    Output("histogram-graph", "figure"),
    Output("scatter-graph", "figure"),
    Output("bar-graph", "figure"),
    Output("filter-count", "children"),
    Input("region-filter", "value"),
    Input("contract-filter", "value"),
    Input("tenure-filter", "value"),
)
def update_charts(regions, contracts, tenure):
    filtered = df[
        df["region"].isin(regions or [])
        & df["contract_type"].isin(contracts or [])
        & df["tenure_months"].between(*tenure)
    ]
    return (*make_figures(filtered), f"{len(filtered):,} dari {len(df):,} pelanggan ditampilkan")


if __name__ == "__main__":
    app.run(debug=False, port=8051)
