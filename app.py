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

class FractalInterpolator:
    def __init__(self, strategy='CVS'):
        self.strategy = strategy
        self.scaler = MinMaxScaler()
        
    def calculate_hurst_exponent(self, data):
        """
        Calcula el exponente de Hurst para la estrategia CHS
        """
        lags = range(2, min(len(data) // 2, 20))
        tau = [np.sqrt(np.std(np.subtract(data[lag:], data[:-lag]))) for lag in lags]
        reg = np.polyfit(np.log(lags), np.log(tau), 1)
        return reg[0]
    
    def get_vertical_scaling_factor(self, subset, strategy):
        """
        Determina el factor de escala vertical según la estrategia
        """
        if strategy == 'CHS':
            h = self.calculate_hurst_exponent(subset)
            return np.clip(h, 0.1, 0.9)
        
        elif strategy == 'CVS':
            def objective(trial):
                si = trial.suggest_float('si', -0.9, 0.9)
                interpolated = self.fractal_interpolate(subset, si)
                # Penalizar variaciones extremas
                variation = np.var(np.diff(interpolated))
                target_variation = np.var(np.diff(subset))
                return np.abs(variation - target_variation)
            
            study = optuna.create_study(direction='minimize')
            study.optimize(objective, n_trials=20)
            return study.best_params['si']
        
        else:  # FS - Formula Strategy
            dx = subset[-1] - subset[0]
            dy = subset[1:] - subset[:-1]
            return np.mean(dy / dx) if dx != 0 else 0.5

    def fractal_interpolate(self, data, si, n_points=5):
        """
        Implementa la interpolación fractal usando IFS (Iterated Function System)
        """
        result = []
        for i in range(len(data) - 1):
            x0, x1 = i, i + 1
            y0, y1 = data[i], data[i + 1]
            
            # Generar puntos intermedios usando IFS
            x = np.linspace(x0, x1, n_points)
            y = np.zeros(n_points)
            y[0], y[-1] = y0, y1
            
            # Aplicar transformaciones fractales
            for j in range(1, n_points - 1):
                t = (x[j] - x0) / (x1 - x0)
                y[j] = y0 + (y1 - y0) * t + si * t * (1 - t) * (y1 - y0)
            
            result.extend(y[:-1] if i < len(data) - 2 else y)
        
        return np.array(result)

    def interpolate(self, data, preserve_endpoints=True):
        """
        Aplica la interpolación fractal a toda la serie
        """
        scaled_data = self.scaler.fit_transform(data.reshape(-1, 1)).ravel()
        
        si = self.get_vertical_scaling_factor(scaled_data, self.strategy)
        interpolated = self.fractal_interpolate(scaled_data, si)
        
        # Preservar puntos originales si se requiere
        if preserve_endpoints:
            interpolated[0] = scaled_data[0]
            interpolated[-1] = scaled_data[-1]
        
        return self.scaler.inverse_transform(interpolated.reshape(-1, 1)).ravel()

def create_lstm_model(input_shape, complexity='medium'):
    """
    Crea un modelo LSTM con diferentes niveles de complejidad
    """
    model = Sequential()
    
    if complexity == 'simple':
        model.add(LSTM(50, input_shape=input_shape))
        model.add(Dense(1))
    
    elif complexity == 'medium':
        model.add(LSTM(100, return_sequences=True, input_shape=input_shape))
        model.add(Dropout(0.2))
        model.add(LSTM(50))
        model.add(Dropout(0.2))
        model.add(Dense(25))
        model.add(Dense(1))
    
    else:  # complex
        model.add(LSTM(200, return_sequences=True, input_shape=input_shape))
        model.add(Dropout(0.2))
        model.add(LSTM(100, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(50))
        model.add(Dropout(0.2))
        model.add(Dense(25))
        model.add(Dense(1))
    
    model.compile(optimizer='adam', loss='huber')
    return model

def prepare_sequences(data, seq_length):
    """
    Prepara secuencias para el entrenamiento LSTM
    """
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:(i + seq_length)])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

def main():
    st.title('Predicción Avanzada de Series Temporales con Interpolación Fractal')
    
    # Configuración de sidebar
    st.sidebar.header("Configuración")
    
    # Selección de estrategia
    strategy = st.sidebar.selectbox(
        'Estrategia de Interpolación:',
        ['CVS', 'CHS', 'FS'],
        help="CVS: Optimiza valores cercanos, CHS: Usa exponente Hurst, FS: Usa fórmula"
    )
    
    # Parámetros de interpolación
    n_interpolation_points = st.sidebar.slider(
        "Puntos de interpolación", 
        min_value=3, 
        max_value=20, 
        value=5
    )
    
    # Parámetros del modelo
    model_complexity = st.sidebar.selectbox(
        "Complejidad del modelo",
        ['simple', 'medium', 'complex']
    )
    
    sequence_length = st.sidebar.slider(
        "Longitud de secuencia", 
        min_value=3, 
        max_value=50, 
        value=10
    )
    
    # Carga de datos
    data = data_input_section()
    
    if data is not None and not data.empty:
        st.subheader("Datos Originales")
        st.dataframe(data.head())
        
        # Selección de variable objetivo
        target_variable = st.selectbox(
            "Variable objetivo:",
            ['revenue', 'cost']
        )
        
        # Preparación de datos
        target_data = data[target_variable].values
        
        # Interpolación fractal
        interpolator = FractalInterpolator(strategy=strategy)
        interpolated_data = interpolator.interpolate(target_data)
        
        # Visualización de datos originales vs interpolados
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=data['month'],
            y=target_data,
            mode='lines+markers',
            name='Datos Originales'
        ))
        
        # Crear índices para datos interpolados
        interpolated_x = np.linspace(0, len(data['month'])-1, len(interpolated_data))
        fig.add_trace(go.Scatter(
            x=interpolated_x,
            y=interpolated_data,
            mode='lines',
            name='Datos Interpolados'
        ))
        
        st.plotly_chart(fig)
        
        # Entrenamiento y predicción
        if st.button("Entrenar y Predecir"):
            with st.spinner("Entrenando modelo..."):
                # Preparar datos para LSTM
                X, y = prepare_sequences(interpolated_data, sequence_length)
                
                if len(X) > 0:
                    X = X.reshape((X.shape[0], X.shape[1], 1))
                    
                    # Crear y entrenar modelo
                    model = create_lstm_model(
                        input_shape=(sequence_length, 1),
                        complexity=model_complexity
                    )
                    
                    history = model.fit(
                        X, y,
                        epochs=50,
                        batch_size=32,
                        validation_split=0.2,
                        verbose=0
                    )
                    
                    # Predicciones
                    predictions = model.predict(X)
                    
                    # Visualización final
                    fig_final = go.Figure()
                    fig_final.add_trace(go.Scatter(
                        x=data['month'],
                        y=target_data,
                        mode='lines+markers',
                        name='Datos Originales'
                    ))
                    fig_final.add_trace(go.Scatter(
                        x=np.arange(len(predictions)),
                        y=predictions.flatten(),
                        mode='lines',
                        name='Predicciones'
                    ))
                    
                    st.plotly_chart(fig_final)
                    
                    # Métricas de rendimiento
                    mse = np.mean((y - predictions.flatten())**2)
                    mae = np.mean(np.abs(y - predictions.flatten()))
                    
                    st.subheader("Métricas de Rendimiento")
                    col1, col2 = st.columns(2)
                    col1.metric("Error Cuadrático Medio", f"{mse:.4f}")
                    col2.metric("Error Absoluto Medio", f"{mae:.4f}")
                    
                    # Gráfico de pérdida durante el entrenamient
                    # Gráfico de pérdida durante el entrenamiento
                    fig_loss = go.Figure()
                    fig_loss.add_trace(go.Scatter(
                        y=history.history['loss'],
                        name='Pérdida de entrenamiento'
                    ))
                    fig_loss.add_trace(go.Scatter(
                        y=history.history['val_loss'],
                        name='Pérdida de validación'
                    ))
                    fig_loss.update_layout(
                        title='Evolución del Entrenamiento',
                        xaxis_title='Época',
                        yaxis_title='Pérdida'
                    )
                    st.plotly_chart(fig_loss)
                    
                    # Añadir botón para descargar predicciones
                    pred_df = pd.DataFrame({
                        'Tiempo': np.arange(len(predictions)),
                        'Predicción': predictions.flatten()
                    })
                    csv = pred_df.to_csv(index=False)
                    st.download_button(
                        label="Descargar Predicciones",
                        data=csv,
                        file_name="predicciones.csv",
                        mime="text/csv"
                    )
                    
                else:
                    st.warning("No hay suficientes datos para el entrenamiento")

if __name__ == "__main__":
    main()