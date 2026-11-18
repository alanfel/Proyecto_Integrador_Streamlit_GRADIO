import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import io

# Desactivar logs de TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, Dropout
from tensorflow.keras.optimizers import Adam

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    classification_report, confusion_matrix, accuracy_score, f1_score,
    roc_curve, auc
)
from sklearn.preprocessing import label_binarize
from statsmodels.tsa.seasonal import seasonal_decompose

# Configuración de página de Streamlit
st.set_page_config(
    page_title="⚡ Dashboard Energía & Clima",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 1. GENERACIÓN DEL DATASET SINTÉTICO
@st.cache_data
def load_data():
    np.random.seed(42)
    n_records = 600
    fechas = pd.date_range(start="1975-01-01", periods=n_records, freq="ME")
    t = np.linspace(0, 10 * np.pi, n_records)
    indice_oni = 1.5 * np.sin(t / 2) + np.random.normal(0, 0.3, n_records)

    precipitacion = 180 - 45 * indice_oni + np.random.normal(0, 15, n_records)
    precipitacion = np.clip(precipitacion, 30, 450)

    temperatura = 21 + 1.2 * indice_oni + np.random.normal(0, 0.5, n_records)
    temperatura = np.clip(temperatura, 15, 30)

    aportes_hidricos = 100 - 25 * indice_oni + np.random.normal(0, 8, n_records)
    aportes_hidricos = np.clip(aportes_hidricos, 20, 160)

    volumen_embalses = 70 - 18 * indice_oni + np.random.normal(0, 6, n_records)
    volumen_embalses = np.clip(volumen_embalses, 15, 99)

    demanda_diaria = 120 + 0.15 * np.arange(n_records) + np.random.normal(0, 3, n_records)

    precio_bolsa = 120 + 150 * np.maximum(0, indice_oni)**1.5 + (100 - volumen_embalses) * 2.5 + np.random.normal(0, 20, n_records)
    precio_bolsa = np.clip(precio_bolsa, 50, 950)

    componente_g = 180 + 1.1 * precio_bolsa * 0.4 + 0.5 * np.arange(n_records) + np.random.normal(0, 15, n_records)
    componente_g = np.clip(componente_g, 120, 1200)

    df = pd.DataFrame({
        'Fecha': fechas,
        'Precio_Bolsa_Nacional_COP_kWh': precio_bolsa,
        'Componente_G_CEO_Cauca_COP_kWh': componente_g,
        'Aportes_Hidricos_Colombia_Pct': aportes_hidricos,
        'Volumen_Util_Embalses_Pct': volumen_embalses,
        'Demanda_Nacional_GWh_Dia': demanda_diaria,
        'Indice_ONI_Clima_Hist': indice_oni,
        'Precipitacion_Promedio_mm': precipitacion,
        'Temperatura_Promedio_C': temperatura
    })

    def categorizar_clima(oni):
        return 'El Niño (Cálido/Seco)' if oni >= 0.5 else ('La Niña (Frío/Lluvioso)' if oni <= -0.5 else 'Neutro')

    df['Estado_Climatico'] = df['Indice_ONI_Clima_Hist'].apply(categorizar_clima)
    return df

df_master = load_data()
num_cols = df_master.select_dtypes(include=[np.number]).columns.tolist()

with st.sidebar:
    st.markdown("# ⚡ Navegación & Control")
    st.info("Dashboard optimizado para la toma de decisiones.")
    st.metric(label="Muestras Históricas", value=f"{len(df_master)} meses")

st.markdown("<h1 style='text-align: center;'>⚡ Dashboard de Inteligencia Energética</h1>", unsafe_allow_html=True)
st.write(df_master.head(20))
