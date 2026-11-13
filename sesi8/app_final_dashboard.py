"""Final customer churn dashboard with EDA, filters, and model inference."""

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, dash_table, dcc, html
from dash.exceptions import PreventUpdate

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from utils.model_utils import load_metadata, load_model, predict, validate_input


# Data and model are loaded once when the app starts.
META = load_metadata()
MODEL = load_model()
raw = pd.read_csv(ROOT / "data" / "customer_churn.csv", parse_dates=["signup_date"])
raw["total_charges"] = raw["total_charges"].fillna(
    raw["monthly_charges"] * raw["tenure_months"]
)
DF = predict(MODEL, raw, META)
DF["status"] = DF["churn"].map({0: "Bertahan", 1: "Churn"})
DF["periode"] = DF["signup_date"].dt.to_period("Q").astype(str)

CATS = META["categories"]
RISK_ORDER = ["Rendah", "Sedang", "Tinggi"]
RISK_COLOR = {"Rendah": "#55A868", "Sedang": "#DD8452", "Tinggi": "#C44E52"}
TENURE_MIN = int(DF["tenure_months"].min())
TENURE_MAX = int(DF["tenure_months"].max())

COLOR = {
    "bg": "#F5F7FB",
    "card": "white",
    "ink": "#1F2A44",
    "muted": "#6B7280",
    "accent": "#2F5DA8",
}
CARD = {
    "background": COLOR["card"],
    "borderRadius": "12px",
    "padding": "16px",
    "boxShadow": "0 1px 3px rgba(0,0,0,.08)",
}


def label(text):
    return html.Label(text, style={"fontWeight": 600, "fontSize": "13px"})


def rupiah(value):
    return f"Rp{value:,.0f}".replace(",", ".")


def kpi(judul, nilai, sub=""):
    return html.Div(
        style={**CARD, "flex": 1},
        children=[
            html.Div(judul, style={"color": COLOR["muted"], "fontSize": "13px"}),
            html.Div(
                nilai,
                style={"fontSize": "28px", "fontWeight": 700, "color": COLOR["ink"]},
            ),
            html.Div(sub, style={"color": COLOR["muted"], "fontSize": "12px"}),
        ],
    )


def num_input(component_id, text, value, minimum, maximum, step=1):
    return html.Div(
        [
            label(text),
            dcc.Input(
                id=component_id,
                type="number",
                value=value,
                min=minimum,
                max=maximum,
                step=step,
                style={"width": "100%"},
            ),
        ]
    )


def dd_input(component_id, text, options):
    return html.Div(
        [
            label(text),
            dcc.Dropdown(
                id=component_id, options=options, value=options[0], clearable=False
            ),
        ]
    )


def chart_card(component_id):
    return html.Div(dcc.Graph(id=component_id), style=CARD)


sidebar = html.Div(
    style={**CARD, "width": "250px", "flexShrink": 0},
    children=[
        html.H4("Filter", style={"marginTop": 0}),
        label("Region"),
        dcc.Dropdown(id="f-region", options=CATS["region"], multi=True, placeholder="Semua"),
        html.Br(),
        label("Segmen"),
        dcc.Dropdown(id="f-segment", options=CATS["segment"], multi=True, placeholder="Semua"),
        html.Br(),
        label("Jenis kontrak"),
        dcc.Checklist(
            id="f-kontrak", options=CATS["contract_type"], value=CATS["contract_type"]
        ),
        html.Br(),
        label("Kategori risiko (prediksi)"),
        dcc.Checklist(id="f-risk", options=RISK_ORDER, value=RISK_ORDER, inline=True),
        html.Br(),
        label("Tenure (bulan)"),
        dcc.RangeSlider(
            id="f-tenure",
            min=TENURE_MIN,
            max=TENURE_MAX,
            value=[TENURE_MIN, TENURE_MAX],
            step=1,
            marks={n: str(n) for n in (TENURE_MIN, 24, 48, TENURE_MAX)},
        ),
        html.Hr(),
        html.Small(
            f"Model: {META['model_name']} v{META['model_version']}",
            style={"color": COLOR["muted"]},
        ),
    ],
)

TABLE_FIELDS = [
    ("ID", "customer_id"),
    ("Region", "region"),
    ("Segmen", "segment"),
    ("Kontrak", "contract_type"),
    ("Tenure", "tenure_months"),
    ("Keluhan", "num_complaints"),
    ("Prob. churn", "churn_probability"),
    ("Risiko", "risk_category"),
]
TABLE_COLUMNS = [
    {
        "name": name,
        "id": field,
        **(
            {"type": "numeric", "format": {"specifier": ".1%"}}
            if field == "churn_probability"
            else {}
        ),
    }
    for name, field in TABLE_FIELDS
]

tab_overview = html.Div(
    [
        html.Div(id="kpi-row", style={"display": "flex", "gap": "12px", "marginBottom": "12px"}),
        html.Div(
            style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "12px"},
            children=[
                chart_card("g-donut"),
                chart_card("g-region-risk"),
                chart_card("g-trend"),
                chart_card("g-contract"),
                chart_card("g-risk"),
                chart_card("g-actual-pred"),
            ],
        ),
        html.Div(
            style={**CARD, "marginTop": "12px"},
            children=[
                html.H4(id="tbl-title", style={"marginTop": 0}),
                dash_table.DataTable(
                    id="tbl-risk",
                    page_size=10,
                    sort_action="native",
                    filter_action="native",
                    columns=TABLE_COLUMNS,
                    style_cell={"padding": "6px", "fontSize": "13px", "textAlign": "left"},
                    style_header={"fontWeight": "bold", "background": "#E9EDF5"},
                    style_data_conditional=[
                        {
                            "if": {
                                "filter_query": "{risk_category} = 'Tinggi'",
                                "column_id": "risk_category",
                            },
                            "color": "white",
                            "backgroundColor": RISK_COLOR["Tinggi"],
                        },
                        {
                            "if": {
                                "filter_query": "{risk_category} = 'Sedang'",
                                "column_id": "risk_category",
                            },
                            "backgroundColor": "#FBE3D3",
                        },
                    ],
                ),
            ],
        ),
    ]
)

tab_predict = html.Div(
    style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "12px"},
    children=[
        html.Div(
            style=CARD,
            children=[
                html.H4("Simulasi prediksi pelanggan", style={"marginTop": 0}),
                html.Div(
                    style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "10px"},
                    children=[
                        dd_input("p-region", "Region", CATS["region"]),
                        dd_input("p-segment", "Segmen", CATS["segment"]),
                        dd_input("p-contract", "Kontrak", CATS["contract_type"]),
                        dd_input("p-payment", "Pembayaran", CATS["payment_method"]),
                        dd_input("p-service", "Layanan", CATS["internet_service"]),
                        num_input("p-tenure", "Tenure (bulan)", 6, 1, 72),
                        num_input("p-monthly", "Tagihan bulanan (Rp)", 450000, 100000, 2000000, 10000),
                        num_input("p-complaints", "Jumlah keluhan", 2, 0, 10),
                        num_input("p-calls", "Panggilan CS", 3, 0, 15),
                    ],
                ),
                html.Br(),
                html.Button(
                    "Jalankan prediksi",
                    id="btn-predict",
                    n_clicks=0,
                    style={
                        "background": COLOR["accent"],
                        "color": "white",
                        "border": 0,
                        "padding": "10px 18px",
                        "borderRadius": "8px",
                        "cursor": "pointer",
                    },
                ),
            ],
        ),
        html.Div(
            style=CARD,
            children=[
                html.H4("Hasil prediksi", style={"marginTop": 0}),
                html.Div(id="pred-output", children=html.P("Isi form lalu klik tombol.")),
                dcc.Graph(id="g-gauge", style={"height": "260px"}),
            ],
        ),
    ],
)

app = Dash(__name__, title="Final Dashboard — Customer Churn")
server = app.server  # Entry point for deployment.
app.layout = html.Div(
    style={
        "background": COLOR["bg"],
        "minHeight": "100vh",
        "padding": "16px",
        "fontFamily": "Inter, Arial, sans-serif",
        "color": COLOR["ink"],
    },
    children=[
        html.Div(
            style={"marginBottom": "12px"},
            children=[
                html.H2("Customer Churn Intelligence Dashboard", style={"margin": 0}),
                html.Div(
                    "Pertanyaan bisnis: pelanggan mana yang berisiko churn dan apa yang harus "
                    "dilakukan? · Data latihan 2023–2025 · Prediksi dari final trained model",
                    style={"color": COLOR["muted"]},
                ),
            ],
        ),
        html.Div(
            style={"display": "flex", "gap": "12px", "alignItems": "flex-start"},
            children=[
                sidebar,
                html.Div(
                    style={"flex": 1, "minWidth": 0},
                    children=[
                        dcc.Tabs(
                            id="tabs",
                            value="overview",
                            children=[
                                dcc.Tab(
                                    label="Overview & Risiko",
                                    value="overview",
                                    children=[tab_overview],
                                ),
                                dcc.Tab(
                                    label="Simulasi Prediksi",
                                    value="predict",
                                    children=[tab_predict],
                                ),
                            ],
                        )
                    ],
                ),
            ],
        ),
    ],
)


def apply_filters(region, segment, kontrak, risk, tenure):
    filtered = DF
    if region:
        filtered = filtered[filtered["region"].isin(region)]
    if segment:
        filtered = filtered[filtered["segment"].isin(segment)]
    filtered = filtered[
        filtered["contract_type"].isin(kontrak or [])
        & filtered["risk_category"].isin(risk or [])
    ]
    return filtered[filtered["tenure_months"].between(*tenure)]


FILTERS = [
    Input("f-region", "value"),
    Input("f-segment", "value"),
    Input("f-kontrak", "value"),
    Input("f-risk", "value"),
    Input("f-tenure", "value"),
]


@app.callback(
    Output("kpi-row", "children"),
    Output("g-donut", "figure"),
    Output("g-region-risk", "figure"),
    Output("g-trend", "figure"),
    Output("g-contract", "figure"),
    Output("g-risk", "figure"),
    Output("g-actual-pred", "figure"),
    Output("tbl-risk", "data"),
    Output("tbl-title", "children"),
    *FILTERS,
)
def update_overview(region, segment, kontrak, risk, tenure):
    filtered = apply_filters(region, segment, kontrak, risk, tenure)
    if filtered.empty:
        empty = px.scatter(title="Tidak ada data — longgarkan filter", template="plotly_white")
        return [kpi("Pelanggan", "0")], empty, empty, empty, empty, empty, empty, [], "Tidak ada data"

    high_risk = filtered["risk_category"].eq("Tinggi")
    kpis = [
        kpi("Pelanggan", f"{len(filtered):,}", "sesuai filter"),
        kpi("Churn rate aktual", f"{filtered['churn'].mean():.1%}", f"total: {DF['churn'].mean():.1%}"),
        kpi("Rata-rata prob. churn", f"{filtered['churn_probability'].mean():.1%}", "prediksi model"),
        kpi(
            "Risiko tinggi",
            f"{high_risk.sum():,}",
            rupiah(filtered.loc[high_risk, "monthly_charges"].sum()) + "/bln berisiko",
        ),
    ]

    donut = px.pie(
        filtered,
        names="risk_category",
        hole=0.55,
        color="risk_category",
        color_discrete_map=RISK_COLOR,
        category_orders={"risk_category": RISK_ORDER},
        title="Komposisi kategori risiko",
    )
    donut.update_traces(textinfo="percent+label", sort=False)
    donut.add_annotation(text=f"<b>{len(filtered):,}</b><br>pelanggan", showarrow=False, font_size=14)
    donut.update_layout(showlegend=False)

    regional_risk = (
        filtered.assign(tinggi=high_risk)
        .groupby("region", as_index=False)["tinggi"]
        .mean()
        .sort_values("tinggi")
    )
    region_risk = px.bar(
        regional_risk,
        x="tinggi",
        y="region",
        orientation="h",
        text_auto=".1%",
        template="plotly_white",
        color_discrete_sequence=[RISK_COLOR["Tinggi"]],
        title="Porsi pelanggan risiko tinggi per region",
        labels={"tinggi": "Porsi risiko tinggi", "region": ""},
    )
    region_risk.update_xaxes(tickformat=".0%")

    trend_data = (
        filtered.groupby("periode")
        .agg(aktual=("churn", "mean"), prediksi=("churn_probability", "mean"))
        .reset_index()
    )
    trend = px.line(
        trend_data.melt("periode", var_name="seri", value_name="rate"),
        x="periode",
        y="rate",
        color="seri",
        markers=True,
        template="plotly_white",
        title="Churn aktual vs rata-rata prediksi per kuartal pendaftaran",
        labels={"periode": "Kuartal pendaftaran", "rate": "Rate", "seri": ""},
    )
    trend.update_yaxes(tickformat=".0%")

    contract_data = (
        filtered.groupby(["contract_type", "risk_category"])
        .size()
        .reset_index(name="n")
    )
    contract = px.bar(
        contract_data,
        x="contract_type",
        y="n",
        color="risk_category",
        color_discrete_map=RISK_COLOR,
        category_orders={
            "risk_category": RISK_ORDER,
            "contract_type": ["Bulanan", "Tahunan", "Dua Tahun"],
        },
        template="plotly_white",
        title="Komposisi risiko per jenis kontrak",
        labels={"contract_type": "", "n": "Pelanggan", "risk_category": "Risiko"},
    )

    risk_distribution = px.histogram(
        filtered,
        x="churn_probability",
        nbins=30,
        color="risk_category",
        color_discrete_map=RISK_COLOR,
        category_orders={"risk_category": RISK_ORDER},
        template="plotly_white",
        title="Distribusi probabilitas churn (prediksi)",
        labels={"churn_probability": "Probabilitas churn", "risk_category": "Risiko"},
    )
    risk_distribution.add_vline(
        x=META["decision_threshold"], line_dash="dash", annotation_text="threshold"
    )

    actual_pred_data = (
        filtered.groupby("region")
        .agg(
            aktual=("churn", "mean"),
            prediksi=("churn_probability", "mean"),
            n=("customer_id", "count"),
        )
        .reset_index()
    )
    actual_pred = px.scatter(
        actual_pred_data,
        x="aktual",
        y="prediksi",
        size="n",
        text="region",
        template="plotly_white",
        title="Region: churn aktual vs prediksi (ukuran = jumlah pelanggan)",
        labels={"aktual": "Churn rate aktual", "prediksi": "Rata-rata prediksi"},
    )
    actual_pred.update_traces(textposition="top center", cliponaxis=False)
    padding = 0.03
    actual_pred.update_xaxes(
        tickformat=".0%",
        range=[actual_pred_data["aktual"].min() - padding, actual_pred_data["aktual"].max() + padding],
    )
    actual_pred.update_yaxes(
        tickformat=".0%",
        range=[actual_pred_data["prediksi"].min() - padding, actual_pred_data["prediksi"].max() + padding],
    )

    top = filtered.sort_values("churn_probability", ascending=False).head(200)
    title = f"Prioritas retensi — {len(top)} pelanggan dengan probabilitas tertinggi"
    return (
        kpis,
        donut,
        region_risk,
        trend,
        contract,
        risk_distribution,
        actual_pred,
        top.to_dict("records"),
        title,
    )


@app.callback(
    Output("pred-output", "children"),
    Output("g-gauge", "figure"),
    Input("btn-predict", "n_clicks"),
    State("p-region", "value"),
    State("p-segment", "value"),
    State("p-contract", "value"),
    State("p-payment", "value"),
    State("p-service", "value"),
    State("p-tenure", "value"),
    State("p-monthly", "value"),
    State("p-complaints", "value"),
    State("p-calls", "value"),
    prevent_initial_call=True,
)
def run_prediction(_clicks, region, segment, contract, payment, service, tenure, monthly, complaints, calls):
    if any(value is None for value in (tenure, monthly, complaints, calls)):
        raise PreventUpdate

    row = pd.DataFrame(
        [{
            "tenure_months": tenure,
            "monthly_charges": monthly,
            "total_charges": monthly * tenure,
            "num_complaints": complaints,
            "support_calls": calls,
            "region": region,
            "segment": segment,
            "contract_type": contract,
            "payment_method": payment,
            "internet_service": service,
        }]
    )
    errors = validate_input(row, META)
    if errors:
        return html.Div([html.P(error, style={"color": "crimson"}) for error in errors]), px.scatter()

    result = predict(MODEL, row, META).iloc[0]
    recommendations = {
        "Tinggi": "Prioritaskan retensi: hubungi dalam 48 jam, tawarkan kontrak tahunan.",
        "Sedang": "Pantau: kirim survei kepuasan dan penawaran loyalitas.",
        "Rendah": "Pertahankan layanan standar.",
    }
    output = html.Div(
        [
            html.Div(
                f"{result.churn_probability:.1%}",
                style={
                    "fontSize": "40px",
                    "fontWeight": 700,
                    "color": RISK_COLOR[result.risk_category],
                },
            ),
            html.Div(
                f"Prediksi: {'Churn' if result.predicted_churn else 'Bertahan'} "
                f"· Risiko {result.risk_category}"
            ),
            html.P(recommendations[result.risk_category], style={"color": COLOR["muted"]}),
        ]
    )
    gauge = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=result.churn_probability * 100,
            number={"suffix": "%"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": RISK_COLOR[result.risk_category]},
                "steps": [
                    {"range": [0, 30], "color": "#E6F4EA"},
                    {"range": [30, 60], "color": "#FDF0E6"},
                    {"range": [60, 100], "color": "#FBE3E3"},
                ],
            },
        )
    )
    gauge.update_layout(margin=dict(t=20, b=10, l=20, r=20), height=240)
    return output, gauge


if __name__ == "__main__":
    app.run(debug=True, port=8053)
