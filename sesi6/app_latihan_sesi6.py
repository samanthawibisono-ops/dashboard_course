"""Latihan sesi 6: grafik Plotly interaktif dalam aplikasi Dash."""

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, Input, Output, dcc, html

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / "data" / "customer_churn.csv")
df["status"] = df["churn"].map({0: "Bertahan", 1: "Churn"})

STATUS_COLOR = {"Bertahan": "#4C72B0", "Churn": "#C44E52"}
REGIONS = sorted(df["region"].unique())
CONTRACTS = sorted(df["contract_type"].unique())
TENURE_MIN = int(df["tenure_months"].min())
TENURE_MAX = int(df["tenure_months"].max())
NATIONAL_CHURN_RATE = float(df["churn"].mean())


def empty_figure():
    figure = go.Figure()
    figure.add_annotation(
        text="Tidak ada data untuk kombinasi filter ini.",
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
    )
    figure.update_layout(template="plotly_white", xaxis_visible=False, yaxis_visible=False)
    return figure


def make_figures(data):
    """Buat empat grafik yang mengikuti filter pengguna."""
    if data.empty:
        return tuple(empty_figure() for _ in range(4))

    histogram = px.histogram(
        data,
        x="tenure_months",
        color="status",
        nbins=36,
        barmode="overlay",
        opacity=0.7,
        color_discrete_map=STATUS_COLOR,
        template="plotly_white",
        labels={"tenure_months": "Tenure (bulan)", "status": "Status"},
        title="Distribusi tenure menurut status churn",
    )

    bubble = px.scatter(
        data,
        x="tenure_months",
        y="monthly_charges",
        color="status",
        size="support_calls",
        size_max=28,
        opacity=0.6,
        color_discrete_map=STATUS_COLOR,
        template="plotly_white",
        hover_name="customer_id",
        hover_data=["region", "contract_type", "num_complaints", "support_calls"],
        labels={
            "tenure_months": "Tenure (bulan)",
            "monthly_charges": "Tagihan bulanan (Rp)",
            "support_calls": "Panggilan CS",
        },
        title="Tenure vs tagihan (ukuran bubble = panggilan CS)",
    )

    payment_data = (
        data.groupby("payment_method", as_index=False)
        .agg(pelanggan=("customer_id", "count"), churn_rate=("churn", "mean"))
        .sort_values("churn_rate")
    )
    payment_bar = px.bar(
        payment_data,
        x="churn_rate",
        y="payment_method",
        orientation="h",
        text_auto=".1%",
        hover_data={"pelanggan": True},
        color_discrete_sequence=["#C44E52"],
        template="plotly_white",
        labels={"churn_rate": "Churn rate", "payment_method": "Metode pembayaran"},
        title="Churn rate per metode pembayaran",
    )
    payment_bar.add_vline(
        x=NATIONAL_CHURN_RATE,
        line_dash="dash",
        line_color="#2F5DA8",
        annotation_text=f"Rata-rata nasional {NATIONAL_CHURN_RATE:.1%}",
    )
    payment_bar.update_xaxes(tickformat=".0%", rangemode="tozero")
    payment_bar.update_yaxes(
        categoryorder="array",
        categoryarray=payment_data["payment_method"].tolist(),
    )

    hierarchy = data.groupby(
        ["region", "segment", "contract_type"], as_index=False
    ).agg(pelanggan=("customer_id", "count"))
    sunburst = px.sunburst(
        hierarchy,
        path=["region", "segment", "contract_type"],
        values="pelanggan",
        color="region",
        template="plotly_white",
        title="Jumlah pelanggan menurut region, segmen, dan kontrak",
    )
    return histogram, bubble, payment_bar, sunburst


def make_gauge():
    """Tampilkan churn rate seluruh data sebagai acuan tetap."""
    gauge = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=NATIONAL_CHURN_RATE * 100,
            number={"suffix": "%", "valueformat": ".1f"},
            title={"text": "Churn rate keseluruhan"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#C44E52"},
            },
        )
    )
    gauge.update_layout(template="plotly_white", height=350)
    return gauge


initial_histogram, initial_bubble, initial_bar, initial_sunburst = make_figures(df)
app = Dash(__name__, title="Latihan Sesi 6")
app.layout = html.Div(
    style={"maxWidth": "1150px", "margin": "auto", "fontFamily": "Arial"},
    children=[
        html.H1("Latihan Visualisasi Plotly — Customer Churn"),
        html.P("Filter region, kontrak, dan tenure akan memperbarui empat grafik di bawah."),
        html.Div(
            style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "16px"},
            children=[
                html.Div(
                    [
                        html.Label("Region"),
                        dcc.Dropdown(
                            id="region-filter",
                            options=REGIONS,
                            value=REGIONS,
                            multi=True,
                        ),
                    ]
                ),
                html.Div(
                    [
                        html.Label("Jenis kontrak"),
                        dcc.Dropdown(
                            id="contract-filter",
                            options=CONTRACTS,
                            value=CONTRACTS,
                            multi=True,
                        ),
                    ]
                ),
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
                dcc.Graph(id="histogram-graph", figure=initial_histogram),
                dcc.Graph(id="bubble-graph", figure=initial_bubble),
                dcc.Graph(id="payment-bar-graph", figure=initial_bar),
                dcc.Graph(id="sunburst-graph", figure=initial_sunburst),
            ],
        ),
        html.P(
            "Cara membaca sunburst: lingkaran dalam menunjukkan region, cincin tengah "
            "menunjukkan segmen, dan cincin luar menunjukkan jenis kontrak. "
            "Luas setiap irisan sebanding dengan jumlah pelanggan; klik irisan untuk "
            "memperbesar kelompok tersebut."
        ),
        dcc.Graph(id="gauge-graph", figure=make_gauge()),
        html.P("Gauge memakai seluruh data sebagai acuan nasional, sehingga tidak berubah oleh filter."),
    ],
)


@app.callback(
    Output("histogram-graph", "figure"),
    Output("bubble-graph", "figure"),
    Output("payment-bar-graph", "figure"),
    Output("sunburst-graph", "figure"),
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
    return (
        *make_figures(filtered),
        f"{len(filtered):,} dari {len(df):,} pelanggan ditampilkan",
    )


if __name__ == "__main__":
    app.run(debug=False, port=8054)
