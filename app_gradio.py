import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

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
import gradio as gr

# 1. GENERACIÓN DEL DATASET SINTÉTICO
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

df_master = pd.DataFrame({
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

df_master['Estado_Climatico'] = df_master['Indice_ONI_Clima_Hist'].apply(categorizar_clima)
num_cols = df_master.select_dtypes(include=[np.number]).columns.tolist()

# --- FUNCIONES DE LA INTERFAZ ---
def tab1_get_data(num_rows):
    return df_master.head(int(num_rows))

def tab2_get_stats():
    return df_master[num_cols].describe().reset_index()

def tab3_plot_dist(variable):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.histplot(df_master[variable], kde=True, color='skyblue', ax=axes[0])
    sns.boxplot(y=df_master[variable], color='lightgreen', ax=axes[1])
    plt.tight_layout()
    return fig

def tab4_plot_bivar(var_x, var_y):
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.regplot(data=df_master, x=var_x, y=var_y, color='purple', ax=ax, line_kws={"color": "red"})
    corr_coef = df_master[var_x].corr(df_master[var_y])
    ax.set_title(f'{var_x} vs {var_y} (R de Pearson: {corr_coef:.4f})')
    plt.tight_layout()
    return fig

def tab5_ts_decomp(variable, period):
    ts_df = df_master.set_index('Fecha')[variable].resample('ME').mean()
    result = seasonal_decompose(ts_df, model='additive', period=int(period))
    fig = result.plot()
    fig.set_size_inches(12, 8)
    plt.tight_layout()
    return fig

def tab6_train_rnn(variable, lags, test_split):
    series = df_master[variable].values
    scaler = MinMaxScaler()
    series_scaled = scaler.fit_transform(series.reshape(-1, 1)).flatten()
    X, y = [], []
    for i in range(len(series_scaled) - lags):
        X.append(series_scaled[i:i+lags])
        y.append(series_scaled[i+lags])
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    split_idx = int(len(X) * (1 - test_split / 100.0))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    model = Sequential([
        SimpleRNN(32, activation='tanh', input_shape=(lags, 1)),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.01), loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=15, batch_size=32, verbose=0)
    preds_scaled = model.predict(X_test).flatten()
    preds = scaler.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()
    reales = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    mae = mean_absolute_error(reales, preds)
    rmse = np.sqrt(mean_squared_error(reales, preds))

    fig, ax = plt.subplots(figsize=(10, 4.5))
    index_range = range(split_idx + lags, len(series))
    ax.plot(index_range, reales, label='Histórico Real', color='blue')
    ax.plot(index_range, preds, label='Predicción RNN', color='orange', linestyle='--')
    ax.legend()
    plt.tight_layout()
    return fig, f"MAE: {mae:.4f}\nRMSE: {rmse:.4f}"

def tab7_corr_matrix():
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(df_master[num_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
    plt.tight_layout()
    return fig

def tab8_clustering(k_clusters):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_master[num_cols])
    kmeans = KMeans(n_clusters=int(k_clusters), random_state=42, n_init=10)
    clusters = kmeans.fit_predict(scaled_data)
    df_temp = df_master.copy()
    df_temp['Cluster'] = clusters
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.scatterplot(data=df_temp, x='Volumen_Util_Embalses_Pct', y='Precio_Bolsa_Nacional_COP_kWh', hue='Cluster', palette='tab10', ax=axes[0])
    sns.boxplot(data=df_temp, x='Cluster', y='Precio_Bolsa_Nacional_COP_kWh', palette='tab10', ax=axes[1])
    plt.tight_layout()
    return fig

def tab9_train_mlp(predictores, target, neuronas, test_size):
    if not predictores or not target: return "Error en variables", plt.figure()
    X = df_master[predictores]
    y = df_master[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100, random_state=42)
    scaler_X, scaler_y = StandardScaler(), StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).ravel()
    y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1)).ravel()

    mlp = MLPRegressor(hidden_layer_sizes=(int(neuronas),), max_iter=300, random_state=42)
    mlp.fit(X_train_scaled, y_train_scaled)
    preds_scaled = mlp.predict(X_test_scaled)
    preds = scaler_y.inverse_transform(preds_scaled.reshape(-1, 1)).ravel()
    r2 = r2_score(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].scatter(y_test, preds, alpha=0.6, color='g')
    axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    sns.histplot(y_test - preds, kde=True, color='red', ax=axes[1])
    plt.tight_layout()
    return f"R2: {r2:.4f} | RMSE: {rmse:.4f}", fig

def build_classification_target(target_type):
    if target_type == "Clima":
        labels = df_master['Indice_ONI_Clima_Hist'].apply(lambda x: 0 if x <= -0.5 else (1 if x < 0.5 else 2))
        class_names = ["La Niña", "Neutro", "El Niño"]
    elif target_type == "Alerta Embalses":
        labels = df_master['Volumen_Util_Embalses_Pct'].apply(lambda x: 0 if x < 50 else (1 if x <= 80 else 2))
        class_names = ["Crítico", "Normal", "Óptimo"]
    else:
        t33, t66 = np.percentile(df_master['Precio_Bolsa_Nacional_COP_kWh'], [33.3, 66.6])
        labels = df_master['Precio_Bolsa_Nacional_COP_kWh'].apply(lambda x: 0 if x <= t33 else (1 if x <= t66 else 2))
        class_names = ["Bajo", "Medio", "Alto"]
    return labels, class_names

def tab10_train_classifier(target_type, features, epochs, batch_size, lr, dropout, neurons):
    if not features: return "Seleccione features", plt.figure(), "No entrenado"
    y, class_names = build_classification_target(target_type)
    X = df_master[features]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = Sequential([
        Dense(int(neurons), activation='relu', input_shape=(len(features),)),
        Dropout(dropout),
        Dense(3, activation='softmax')
    ])
    model.compile(optimizer=Adam(learning_rate=lr), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(X_train_scaled, y_train, epochs=int(epochs), batch_size=int(batch_size), validation_split=0.1, verbose=0)
    preds_prob = model.predict(X_test_scaled)
    preds = np.argmax(preds_prob, axis=1)
    report = classification_report(y_test, preds, target_names=class_names)
    acc = accuracy_score(y_test, preds)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].plot(history.history['loss'], label='Entrenamiento')
    axes[0].plot(history.history['val_loss'], label='Validación')
    axes[0].legend()
    cm = confusion_matrix(y_test, preds)
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap='Blues', ax=axes[1])
    y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
    for i in range(3):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], preds_prob[:, i])
        axes[2].plot(fpr, tpr, label=f"{class_names[i]} (AUC = {auc(fpr, tpr):.2f})")
    axes[2].plot([0, 1], [0, 1], 'k--')
    axes[2].legend()
    plt.tight_layout()
    return report, fig, f"Estatus: Entrenado | Accuracy: {acc:.4f}"

with gr.Blocks(title="Dashboard Energía & Clima") as demo:
    gr.Markdown("# ⚡ Dashboard Gradio - Energía & Clima")
    with gr.Tabs():
        with gr.TabItem("📋 Datos"):
            slider_rows = gr.Slider(5, 100, step=5, value=15, label="Número de registros")
            btn_load_df = gr.Button("Cargar datos")
            out_table = gr.Dataframe(interactive=False)
            btn_load_df.click(tab1_get_data, inputs=slider_rows, outputs=out_table)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
