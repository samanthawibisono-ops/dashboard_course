"""Dashboard Streamlit untuk prediksi churn pelanggan."""

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))  # Agar folder utils bisa di-import.

from utils.model_utils import load_metadata, load_model, predict, validate_input


st.set_page_config(page_title="Prediksi Churn", page_icon="🤖", layout="wide")


@st.cache_resource
def get_model():
    return load_model()


@st.cache_data
def get_meta():
    return load_metadata()


STATUS = {"Bertahan": "#2a78d6", "Churn": "#eb6834"}
KATEGORI = ["#2a78d6", "#1baf7a", "#eda100", "#4a3aa7", "#e87ba4"]
KONTRAK = ["Bulanan", "Tahunan", "Dua Tahun"]


@st.cache_data
def load_data():
    df = pd.read_csv(ROOT / "data" / "customer_churn.csv", parse_dates=["signup_date"])
    df["total_charges_missing"] = df["total_charges"].isna()  # catat sebelum imputasi
    df["total_charges"] = df["total_charges"].fillna(df["total_charges"].median())
    df["status"] = df["churn"].map({0: "Bertahan", 1: "Churn"})
    df["kelompok_tenure"] = pd.cut(df["tenure_months"], [0, 12, 24, 48, 72],
                                   labels=["0-12", "13-24", "25-48", "49-72"]).astype(str)
    return df


def gaya(fig, tinggi=360):
    fig.update_layout(template="plotly_white", height=tinggi, margin=dict(l=10, r=10, t=50, b=10),
                      title_font_size=15, legend_title_text="")
    return fig


def bar_komposisi(df, kolom, judul, warna):
    """Bar horizontal terurut: jumlah & persentase pelanggan per kategori."""
    k = df[kolom].value_counts().sort_values().rename_axis(kolom).reset_index(name="pelanggan")
    k["persen"] = k["pelanggan"] / k["pelanggan"].sum()
    fig = px.bar(k, x="pelanggan", y=kolom, orientation="h", title=judul,
                 text=k["persen"].map("{:.1%}".format), color_discrete_sequence=[warna])
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_layout(yaxis_title="", xaxis_title="Jumlah pelanggan")
    return fig


model, meta = get_model(), get_meta()
st.title("Dashboard Customer Churn")

with st.expander("ℹ️ Informasi model final", expanded=False):
    st.write(f"**{meta['model_name']}** · versi {meta['model_version']} · {meta['algorithm']}")
    st.write("Fitur input:", ", ".join(meta["feature_order"]))
    st.write(
        f"Threshold keputusan: {meta['decision_threshold']} · "
        f"Risk bins: {meta['risk_bins']}"
    )
    st.caption("Model tidak dilatih ulang di aplikasi ini; aplikasi hanya melakukan inference.")

COLOR = {"Rendah": "🟢", "Sedang": "🟡", "Tinggi": "🔴"}
tab_eda, tab_single, tab_batch = st.tabs(["🔎 EDA", "👤 Prediksi satu pelanggan", "📂 Prediksi batch (CSV)"])

with tab_eda:
    df = load_data()
    st.title("🔎 Exploratory Data Analysis — Customer Churn")
    st.caption("Data latihan sintetis · 3.000 pelanggan · pendaftaran Januari 2023 – Desember 2025")

    # ---------- KPI CARDS ----------
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Total pelanggan", f"{len(df):,}")
    k2.metric("Churn rate", f"{df.churn.mean():.1%}")
    k3.metric("Rata-rata tagihan", f"Rp{df.monthly_charges.mean():,.0f}".replace(",", "."))
    k4.metric("Rata-rata tenure", f"{df.tenure_months.mean():.1f} bln")
    k5.metric("Missing total_charges", f"{df.total_charges_missing.sum()} baris", help="Diisi median")

    # ---------- 1. KOMPOSISI: pie, donut, treemap ----------
    with st.container():
        st.subheader("Komposisi")
        c1, c2 = st.columns(2)
        pie = px.pie(df, names="status", color="status", color_discrete_map=STATUS,
                     title="Satu dari tiga pelanggan churn")
        pie.update_traces(textinfo="percent+label", textfont_size=14)
        c1.plotly_chart(gaya(pie), width="stretch")
        donut = px.pie(df, names="contract_type", hole=0.55, color_discrete_sequence=KATEGORI,
                       category_orders={"contract_type": KONTRAK},
                       title="Separuh pelanggan berkontrak bulanan")
        donut.update_traces(textinfo="percent+label")
        donut.add_annotation(text=f"<b>{len(df):,}</b><br>pelanggan", showarrow=False, font_size=16)
        c2.plotly_chart(gaya(donut), width="stretch")

        tm = df.groupby(["region", "segment"], as_index=False).agg(pelanggan=("customer_id", "count"),
                                                                   churn_rate=("churn", "mean"))
        tree = px.treemap(tm, path=["region", "segment"], values="pelanggan", color="churn_rate",
                          color_continuous_scale="Oranges",
                          title="Treemap: luas = jumlah pelanggan, warna = churn rate")
        st.plotly_chart(gaya(tree, 420), width="stretch")

        # ----- Latihan 1: pie internet_service vs donut region, lalu bar horizontal terurut -----
        st.subheader("Latihan 1 — Pie/donut vs bar horizontal terurut")
        c3, c4 = st.columns(2)
        pie_inet = px.pie(df, names="internet_service", color_discrete_sequence=KATEGORI,
                          title="Pie: komposisi internet_service (3 kategori)")
        pie_inet.update_traces(textinfo="percent+label")
        c3.plotly_chart(gaya(pie_inet), width="stretch")
        donut_reg = px.pie(df, names="region", hole=0.55, color_discrete_sequence=KATEGORI,
                           title="Donut: komposisi region (5 kategori)")
        donut_reg.update_traces(textinfo="percent")
        donut_reg.add_annotation(text=f"<b>{len(df):,}</b><br>pelanggan", showarrow=False, font_size=16)
        c4.plotly_chart(gaya(donut_reg), width="stretch")

        c5, c6 = st.columns(2)
        c5.plotly_chart(gaya(bar_komposisi(df, "internet_service",
                                           "Bar terurut: internet_service", KATEGORI[0])), width="stretch")
        c6.plotly_chart(gaya(bar_komposisi(df, "region",
                                           "Bar terurut: region", KATEGORI[0])), width="stretch")
        st.info(
            "**Perbandingan:** Pie masih cukup terbaca untuk 3 kategori (internet_service), "
            "tapi donut 5 kategori (region) sulit dibaca: irisan kecil seperti Kalimantan (11%) dan "
            "Bali-Nusa Tenggara (10%) hampir sama besar, dan pembaca harus bolak-balik melihat legenda. "
            "Bar horizontal terurut lebih mudah karena mata membandingkan **panjang** pada satu garis dasar "
            "(lebih akurat daripada sudut/luas), peringkat langsung terlihat, dan label kategori menempel "
            "pada batangnya."
        )

    # ---------- 2. PERBANDINGAN: bar, grouped bar, stacked 100% ----------
    with st.container():
        st.subheader("Perbandingan")
        kolom = st.selectbox("Bandingkan churn rate berdasarkan",
                             ["contract_type", "region", "segment", "payment_method", "internet_service"])
        rk = (df.groupby(kolom, as_index=False).agg(pelanggan=("customer_id", "count"),
                                                    churn_rate=("churn", "mean"))
              .sort_values("churn_rate"))
        bar = px.bar(rk, x="churn_rate", y=kolom, orientation="h", text_auto=".1%",
                     color_discrete_sequence=["#eb6834"], title=f"Churn rate per {kolom}")
        bar.add_vline(x=df.churn.mean(), line_dash="dot", annotation_text="rata-rata")
        bar.update_xaxes(tickformat=".0%")
        c1, c2 = st.columns([3, 2])
        c1.plotly_chart(gaya(bar), width="stretch")
        ringkasan = rk.sort_values("churn_rate", ascending=False)
        c2.dataframe(ringkasan.style.format({"churn_rate": "{:.1%}"})
                     .background_gradient(subset=["churn_rate"], cmap="Oranges"),
                     width="stretch", hide_index=True)
        # Latihan 6: unduh tabel ringkasan sebagai CSV
        c2.download_button("⬇️ Unduh ringkasan (CSV)",
                           data=ringkasan.to_csv(index=False).encode("utf-8"),
                           file_name=f"ringkasan_churn_{kolom}.csv", mime="text/csv")

        c3, c4 = st.columns(2)
        gb = df.groupby(["contract_type", "segment"], as_index=False)["churn"].mean()
        grouped = px.bar(gb, x="contract_type", y="churn", color="segment", barmode="group",
                         color_discrete_sequence=KATEGORI, category_orders={"contract_type": KONTRAK},
                         title="Grouped bar: kontrak × segmen")
        grouped.update_yaxes(tickformat=".0%")
        c3.plotly_chart(gaya(grouped), width="stretch")
        stacked = px.histogram(df, y="payment_method", color="status", barnorm="percent",
                               color_discrete_map=STATUS, title="Stacked bar 100%: status per pembayaran")
        stacked.update_layout(xaxis_title="Proporsi (%)", yaxis_title="")
        c4.plotly_chart(gaya(stacked), width="stretch")

        # ----- Latihan 2: stacked bar 100% status churn per internet_service -----
        urut_inet = df.groupby("internet_service")["churn"].mean().sort_values().index.tolist()
        st_inet = px.histogram(df, y="internet_service", color="status", barnorm="percent",
                               color_discrete_map=STATUS, text_auto=".1f",
                               category_orders={"internet_service": urut_inet,
                                                "status": ["Bertahan", "Churn"]},
                               title="Latihan 2 — Stacked bar 100%: status churn per internet_service")
        st_inet.update_layout(xaxis_title="Proporsi (%)", yaxis_title="")
        st.plotly_chart(gaya(st_inet, 300), width="stretch")

    # ---------- 3. DISTRIBUSI: histogram, box, violin ----------
    with st.container():
        st.subheader("Distribusi")
        num = st.selectbox("Variabel numerik", ["tenure_months", "monthly_charges",
                                                "num_complaints", "support_calls"])
        c1, c2 = st.columns(2)
        hist = px.histogram(df, x=num, color="status", barmode="overlay", opacity=0.7, nbins=30,
                            color_discrete_map=STATUS, marginal="box", title=f"Histogram {num}")
        c1.plotly_chart(gaya(hist, 400), width="stretch")
        vio = px.violin(df, x="contract_type", y=num, color="contract_type", box=True,
                        color_discrete_sequence=KATEGORI, category_orders={"contract_type": KONTRAK},
                        title=f"Violin {num} per kontrak")
        c2.plotly_chart(gaya(vio, 400), width="stretch")

        # ----- Latihan 3: box plot support_calls per status churn -----
        box_sc = px.box(df, x="status", y="support_calls", color="status", points="outliers",
                        color_discrete_map=STATUS, category_orders={"status": ["Bertahan", "Churn"]},
                        title="Latihan 3 — Box plot support_calls per status churn")
        box_sc.update_layout(showlegend=False, xaxis_title="", yaxis_title="Jumlah panggilan CS")
        st.plotly_chart(gaya(box_sc, 380), width="stretch")

    # ---------- 4. TREN: line, area ----------
    with st.container():
        st.subheader("Tren")
        bulanan = (df.set_index("signup_date").resample("ME")["customer_id"].count()
                   .rename("pelanggan").reset_index())
        bulanan["rata2_3bln"] = bulanan["pelanggan"].rolling(3).mean()
        line = px.line(bulanan, x="signup_date", y=["pelanggan", "rata2_3bln"], markers=True,
                       color_discrete_sequence=["#9dbbe6", "#2a78d6"],
                       title="Pelanggan baru per bulan dan rata-rata bergerak 3 bulan")
        st.plotly_chart(gaya(line), width="stretch")
        c1, c2 = st.columns(2)
        kt = df.groupby("kelompok_tenure", as_index=False)["churn"].mean()
        lt = px.line(kt, x="kelompok_tenure", y="churn", markers=True,
                     text=kt["churn"].map("{:.1%}".format),
                     color_discrete_sequence=["#eb6834"], title="Churn rate per kelompok tenure")
        lt.update_traces(textposition="top center", marker_size=10)
        lt.update_yaxes(tickformat=".0%", range=[0, 0.65])
        c1.plotly_chart(gaya(lt), width="stretch")
        q = (df.assign(kuartal=df.signup_date.dt.to_period("Q").astype(str))
             .groupby(["kuartal", "segment"], as_index=False)["customer_id"].count())
        area = px.area(q, x="kuartal", y="customer_id", color="segment",
                       color_discrete_sequence=KATEGORI, title="Area: pelanggan baru per kuartal")
        c2.plotly_chart(gaya(area), width="stretch")

        # ----- Latihan 4: churn rate per kuartal pendaftaran -----
        cq = (df.assign(kuartal=df.signup_date.dt.to_period("Q").astype(str))
              .groupby("kuartal", as_index=False).agg(pelanggan=("customer_id", "count"),
                                                      churn_rate=("churn", "mean")))
        lq = px.line(cq, x="kuartal", y="churn_rate", markers=True,
                     text=cq["churn_rate"].map("{:.0%}".format), hover_data=["pelanggan"],
                     color_discrete_sequence=["#eb6834"],
                     title="Latihan 4 — Churn rate per kuartal pendaftaran")
        lq.update_traces(textposition="top center", marker_size=8)
        lq.add_hline(y=df.churn.mean(), line_dash="dot", annotation_text="rata-rata")
        lq.update_yaxes(tickformat=".0%", rangemode="tozero")
        lq.update_layout(xaxis_title="Kuartal pendaftaran", yaxis_title="Churn rate")
        st.plotly_chart(gaya(lq), width="stretch")

    # ---------- 5. HUBUNGAN: scatter, heatmap ----------
    with st.container():
        st.subheader("Hubungan")
        c1, c2 = st.columns(2)
        sc = px.scatter(df, x="tenure_months", y="monthly_charges", color="status", opacity=0.55,
                        color_discrete_map=STATUS, hover_name="customer_id",
                        hover_data=["contract_type", "num_complaints"], title="Tenure vs tagihan")
        c1.plotly_chart(gaya(sc, 420), width="stretch")
        corr = df[["tenure_months", "monthly_charges", "num_complaints", "support_calls", "churn"]].corr()
        hm = px.imshow(corr.round(2), text_auto=True, color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                       title="Heatmap korelasi")
        c2.plotly_chart(gaya(hm, 420), width="stretch")
        pv = pd.pivot_table(df, values="churn", index="region", columns="segment", aggfunc="mean")
        hm2 = px.imshow(pv * 100, text_auto=".1f", color_continuous_scale="Oranges",
                        labels={"color": "Churn (%)"}, title="Heatmap churn rate region × segmen (%)")
        st.plotly_chart(gaya(hm2, 360), width="stretch")

    # ---------- 6. DATA: profil + tabel ----------
    with st.container():
        st.subheader("Data")
        profil = pd.DataFrame({"tipe": df.dtypes.astype(str),
                               "missing": df.isna().sum().where(df.columns != "total_charges",
                                                                df.total_charges_missing.sum()),
                               "unik": df.nunique()}).reset_index(names="kolom")
        c1, c2 = st.columns([2, 3])
        c1.markdown("**Profil kolom**")
        c1.dataframe(profil, width="stretch", hide_index=True)
        c2.markdown("**Statistik deskriptif**")
        c2.dataframe(df.describe(include="number").T.round(1), width="stretch")
        st.markdown("**10 baris pertama**")
        st.dataframe(df.head(10), width="stretch", hide_index=True)

    # ---------- 7. KELUHAN (Latihan 5): churn rate per jumlah keluhan ----------
    with st.container():
        st.subheader("Keluhan")
        kel = (df.groupby("num_complaints", as_index=False)
               .agg(pelanggan=("customer_id", "count"), churn_rate=("churn", "mean")))
        kel["num_complaints"] = kel["num_complaints"].astype(str)  # sumbu x kategorikal
        bk = px.bar(kel, x="num_complaints", y="churn_rate", text_auto=".1%",
                    hover_data=["pelanggan"], color_discrete_sequence=["#eb6834"],
                    title="Churn rate naik seiring jumlah keluhan")
        bk.add_hline(y=df.churn.mean(), line_dash="dot", annotation_text="rata-rata")
        bk.update_yaxes(tickformat=".0%", rangemode="tozero")
        bk.update_layout(xaxis_title="Jumlah keluhan", yaxis_title="Churn rate")
        c1, c2 = st.columns([3, 2])
        c1.plotly_chart(gaya(bk, 400), width="stretch")
        c2.dataframe(kel.style.format({"churn_rate": "{:.1%}"})
                     .background_gradient(subset=["churn_rate"], cmap="Oranges"),
                     width="stretch", hide_index=True)

with tab_single:
    cats = meta["categories"]
    with st.form("form_prediksi"):
        c1, c2, c3 = st.columns(3)
        region = c1.selectbox("Region", cats["region"])
        segment = c1.selectbox("Segmen", cats["segment"])
        contract = c2.selectbox("Jenis kontrak", cats["contract_type"])
        payment = c2.selectbox("Metode pembayaran", cats["payment_method"])
        service = c3.selectbox("Layanan internet", cats["internet_service"])
        tenure = c3.number_input("Tenure (bulan)", 1, 72, 6)

        d1, d2, d3 = st.columns(3)
        monthly = d1.number_input(
            "Tagihan bulanan (Rp)", 100_000, 2_000_000, 450_000, step=10_000
        )
        complaints = d2.number_input("Jumlah keluhan", 0, 10, 2)
        calls = d3.number_input("Panggilan ke CS", 0, 15, 3)
        submitted = st.form_submit_button("🔮 Prediksi", type="primary")

    if submitted:
        row = pd.DataFrame([{
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
        }])
        errs = validate_input(row, meta)
        if errs:
            for error in errs:
                st.error(error)
        else:
            res = predict(model, row, meta).iloc[0]
            m1, m2, m3 = st.columns(3)
            m1.metric("Probabilitas churn", f"{res.churn_probability:.1%}")
            m2.metric("Prediksi", "Churn" if res.predicted_churn else "Bertahan")
            m3.metric("Kategori risiko", f"{COLOR[res.risk_category]} {res.risk_category}")
            st.progress(float(res.churn_probability))

            if res.risk_category == "Tinggi":
                st.error(
                    "Rekomendasi: prioritaskan retensi — tawarkan migrasi ke kontrak "
                    "tahunan dan tindak lanjuti keluhan dalam 48 jam."
                )
            elif res.risk_category == "Sedang":
                st.warning(
                    "Rekomendasi: pantau, kirim survei kepuasan dan penawaran loyalitas."
                )
            else:
                st.success("Rekomendasi: pertahankan layanan standar.")

with tab_batch:
    up = st.file_uploader(
        "Upload CSV pelanggan baru (format sama dengan data/new_customers.csv)",
        type="csv",
    )
    pakai_contoh = st.checkbox("Gunakan contoh data/new_customers.csv", value=up is None)

    if up is not None:
        data = pd.read_csv(up)
    elif pakai_contoh:
        data = pd.read_csv(ROOT / "data" / "new_customers.csv")
    else:
        data = None
        st.info("Upload file atau centang contoh data.")

    if data is not None:
        errs = validate_input(data, meta)
        if errs:
            for error in errs:
                st.error(error)
        else:
            hasil = predict(model, data, meta)
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Pelanggan diproses", f"{len(hasil):,}")
            k2.metric("Prediksi churn", f"{hasil.predicted_churn.sum():,}")
            k3.metric("Rata-rata probabilitas", f"{hasil.churn_probability.mean():.1%}")
            k4.metric("Risiko tinggi", f"{(hasil.risk_category == 'Tinggi').sum():,}")

            risk_color = {"Rendah": "#1baf7a", "Sedang": "#eda100", "Tinggi": "#e34948"}
            a, b = st.columns(2)
            donut = px.pie(
                hasil,
                names="risk_category",
                hole=0.55,
                color="risk_category",
                color_discrete_map=risk_color,
                title="Komposisi kategori risiko",
                category_orders={"risk_category": ["Rendah", "Sedang", "Tinggi"]},
            )
            donut.update_traces(textinfo="percent+label")
            a.plotly_chart(donut.update_layout(height=330, showlegend=False), width="stretch")

            pk = hasil.groupby("contract_type", as_index=False)["churn_probability"].mean()
            bar = px.bar(
                pk,
                x="contract_type",
                y="churn_probability",
                text_auto=".1%",
                color_discrete_sequence=["#eb6834"],
                title="Rata-rata probabilitas churn per kontrak",
            )
            bar.update_yaxes(tickformat=".0%")
            b.plotly_chart(
                bar.update_layout(height=330, template="plotly_white", xaxis_title=""),
                width="stretch",
            )

            risiko = st.multiselect(
                "Tampilkan kategori risiko", ["Tinggi", "Sedang", "Rendah"],
                default=["Tinggi"],
            )
            tampil = hasil[hasil.risk_category.isin(risiko)].sort_values(
                "churn_probability", ascending=False
            )
            st.dataframe(
                tampil,
                width="stretch",
                height=320,
                column_config={
                    "churn_probability": st.column_config.ProgressColumn(
                        "Probabilitas churn", min_value=0.0, max_value=1.0, format="%.2f"
                    )
                },
            )
            st.download_button(
                "⬇️ Unduh hasil prediksi",
                hasil.to_csv(index=False).encode("utf-8"),
                file_name="hasil_prediksi_churn.csv",
                mime="text/csv",
            )
