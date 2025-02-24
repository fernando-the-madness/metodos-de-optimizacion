import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import optuna
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Configuración de la página
st.set_page_config(
    page_title="Predicción de Series Temporales con Interpolación Fractal",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Función para generar datos de ejemplo
def generate_sample_data(start_year=2023, periods=12, base_revenue=10000, trend_factor=0.1, seasonality_factor=0.15, noise_factor=0.05):
    """
    Genera datos de ventas sintéticos
    """
    months = pd.date_range(start=f'{start_year}-01-01', periods=periods, freq='M').strftime('%b')
    
    # Generar revenue
    trend = np.linspace(0, base_revenue * trend_factor, periods)
    seasonal = base_revenue * seasonality_factor * np.sin(np.linspace(0, 2*np.pi, periods))
    noise = np.random.normal(0, base_revenue * noise_factor, periods)
    revenue = base_revenue + trend + seasonal + noise
    
    # Generar costs
    cost_ratio = 0.7
    cost_noise = np.random.normal(0, base_revenue * noise_factor/2, periods)
    costs = revenue * cost_ratio + cost_noise
    
    return pd.DataFrame({
        'month': months,
        'revenue': revenue.round(2),
        'cost': costs.round(2)
    })

def data_input_section():
    """
    Sección para entrada de datos
    """
    st.sidebar.header("Fuente de Datos")
    data_source = st.sidebar.radio(
        "Selecciona la fuente de datos:",
        ["Subir CSV", "Generar Datos de Ejemplo", "Crear Datos Personalizados"]
    )
    
    if data_source == "Subir CSV":
        uploaded_file = st.sidebar.file_uploader("Sube tu archivo CSV", type=['csv'])
        if uploaded_file:
            data = pd.read_csv(uploaded_file)
            return data
        else:
            st.sidebar.info("El archivo CSV debe tener columnas: month, revenue, cost")
            return None
            
    elif data_source == "Generar Datos de Ejemplo":
        periods = st.sidebar.slider("Número de meses", 6, 24, 12)
        data = generate_sample_data(periods=periods)
        if st.sidebar.button("Descargar Datos de Ejemplo"):
            csv = data.to_csv(index=False)
            st.sidebar.download_button(
                "Descargar CSV",
                csv,
                "datos_ejemplo.csv",
                "text/csv"
            )
        return data
        
    else:  # Crear Datos Personalizados
        st.sidebar.subheader("Crear Datos Personalizados")
        
        # Contenedor para los datos
        if 'custom_data' not in st.session_state:
            st.session_state.custom_data = pd.DataFrame(columns=['month', 'revenue', 'cost'])
        
        # Formulario para agregar datos
        with st.sidebar.form("datos_personalizados"):
            mes = st.selectbox(
                "Mes",
                ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            )
            revenue = st.number_input("Ingresos", min_value=0.0, value=10000.0)
            cost = st.number_input("Costos", min_value=0.0, value=7000.0)
            
            if st.form_submit_button("Agregar Datos"):
                new_row = pd.DataFrame({
                    'month': [mes],
                    'revenue': [revenue],
                    'cost': [cost]
                })
                st.session_state.custom_data = pd.concat([st.session_state.custom_data, new_row], ignore_index=True)
        
        # Mostrar datos actuales y opciones
        if not st.session_state.custom_data.empty:
            st.sidebar.dataframe(st.session_state.custom_data)
            
            if st.sidebar.button("Limpiar Datos"):
                st.session_state.custom_data = pd.DataFrame(columns=['month', 'revenue', 'cost'])
            
            if st.sidebar.button("Descargar Datos Personalizados"):
                csv = st.session_state.custom_data.to_csv(index=False)
                st.sidebar.download_button(
                    "Descargar CSV",
                    csv,
                    "datos_personalizados.csv",
                    "text/csv"
                )
            
            return st.session_state.custom_data
        return None

def fractal_interpolation(subset, si, n_interpolation=17):
    """
    Implementa la interpolación fractal
    """
    # Implementación básica de interpolación
    return np.linspace(subset[0], subset[1], n_interpolation)

def apply_interpolation(strategy, data):
    """
    Aplica la estrategia de interpolación seleccionada
    """
    interpolated_data = []
    for i in range(len(data) - 1):
        subset = data[i:i+2]
        
        if strategy == 'CHS':
            si = np.random.uniform(0, 0.2)
        elif strategy == 'CVS':
            def objective(trial):
                si = trial.suggest_uniform('si', -1, 1)
                interpolated = fractal_interpolation(subset, si)
                return np.mean(np.abs(np.diff(interpolated)))
            study = optuna.create_study(direction='minimize')
            study.optimize(objective, n_trials=15)
            si = study.best_params['si']
        else:  # FS
            si = (subset[1] - subset[0]) / np.sqrt((subset[-1] - subset[0])**2 + (subset[1] - subset[0])**2)
            
        interpolated = fractal_interpolation(subset, si)
        interpolated_data.extend(interpolated)
    return np.array(interpolated_data)

def main():
    st.title('Predicción de Series Temporales con Interpolación Fractal')
    
    # Sección de entrada de datos
    data = data_input_section()
    
    if data is not None and not data.empty:
        st.subheader("Datos Cargados")
        st.dataframe(data.head())
        
        # Preparación de datos
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data[['revenue', 'cost']].values)
        
        # Selección de variable objetivo
        target_variable = st.selectbox(
            "Selecciona la variable para análisis:",
            ['revenue', 'cost']
        )
        
        target_idx = 0 if target_variable == 'revenue' else 1
        target_data = scaled_data[:, target_idx]
        
        # Visualización de datos originales
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=data['month'],
            y=data[target_variable],
            mode='lines+markers',
            name='Datos Originales'
        ))
        st.plotly_chart(fig)
        
        # Interpolación
        strategy = st.selectbox(
            'Estrategia de Interpolación:',
            ['CHS', 'CVS', 'FS']
        )
        
        interpolated_data = apply_interpolation(strategy, target_data)
        
        # Visualización de datos interpolados
        fig_interpolated = go.Figure()
        fig_interpolated.add_trace(go.Scatter(
            y=interpolated_data,
            mode='lines',
            name='Datos Interpolados'
        ))
        st.plotly_chart(fig_interpolated)
        
        # Predicción LSTM
        if st.button("Realizar Predicción"):
            with st.spinner("Entrenando modelo LSTM..."):
                # Preparación de datos para LSTM
                def prepare_data(data, time_step=60):
                    X, y = [], []
                    for i in range(len(data) - time_step - 1):
                        X.append(data[i:(i + time_step)])
                        y.append(data[i + time_step])
                    return np.array(X), np.array(y)
                
                X, y = prepare_data(interpolated_data)
                if len(X) > 0:
                    X = X.reshape((X.shape[0], X.shape[1], 1))
                    
                    # Modelo LSTM
                    model = Sequential([
                        LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)),
                        Dropout(0.2),
                        LSTM(50, return_sequences=False),
                        Dropout(0.2),
                        Dense(25),
                        Dense(1)
                    ])
                    model.compile(optimizer='adam', loss='mean_squared_error')
                    
                    # Entrenamiento
                    model.fit(X, y, batch_size=32, epochs=10, verbose=0)
                    
                    # Predicciones
                    predictions = model.predict(X)
                    predictions = scaler.inverse_transform(
                        np.column_stack((predictions if target_idx == 0 else np.zeros_like(predictions),
                                       predictions if target_idx == 1 else np.zeros_like(predictions)))
                    )[:, target_idx]
                    
                    # Visualización final
                    fig_final = go.Figure()
                    fig_final.add_trace(go.Scatter(
                        x=data['month'],
                        y=data[target_variable],
                        mode='lines+markers',
                        name='Datos Originales'
                    ))
                    fig_final.add_trace(go.Scatter(
                        y=predictions,
                        mode='lines',
                        name='Predicciones'
                    ))
                    st.plotly_chart(fig_final)
                else:
                    st.warning("No hay suficientes datos para entrenar el modelo LSTM")

if __name__ == "__main__":
    main()