import datetime as dt
import numpy as np
import pandas as pd
import streamlit as st
st.set_page_config(page_title="Komponen Dasar", page_icon="🧩", layout="centered")

st.title("Profile Peserta")
nama = st.text_input("Siapa nama Anda?")
if nama:
 st.success(f"Selamat datang di pelatihan dashboard, {nama}!")

divisi = st.selectbox("Divisi","HR")
tahunpengalaman = st.slider("Select your age", min_value=18, max_value=150, value=18, step=1)

Profilpeserta = st.info (tahunpengalaman), st.info(divisi), st.info(nama)