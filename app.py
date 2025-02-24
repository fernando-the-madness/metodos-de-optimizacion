import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import optuna
import plotly.graph_objects as go
from typing import Tuple, List
import seaborn as sns

# Configuración de la página
st.set_page_config(
    page_title="Predicción de Series Temporales con Interpolación Fractal",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Funciones de utilidad
def load_and_prepare_data(file) -> Tuple[pd.DataFrame, np.ndarray]:
    """Carga y prepara los datos iniciales."""
    data = pd.read_csv(file, index_col=0, parse_dates=True)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data.values)
    return data, scaled_data, scaler

def create_interactive_plot(data: pd.DataFrame, title: str) -> go.Figure:
    """Crea un gráfico interactivo usando Plotly."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data.values.flatten(),
        mode='lines',
        name='Datos'
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Tiempo",
        yaxis_title="Valor",
        template="plotly_dark"
    )
    return fig

# Clase para manejar la interpolación fractal
class FractalInterpolator:
    def __init__(self, strategy: str):
        self.strategy = strategy
        
    def calculate_si(self, subset: np.ndarray) -> float:
        """Calcula el factor de interpolación según la estrategia."""
        if self.strategy == 'CHS':
            return np.random.uniform(0, 0.2)
        elif self.strategy == 'CVS':
            def objective(trial):
                si = trial.suggest_uniform('si', -1, 1)
                interpolated = self.fractal_interpolation(subset, si)
                return np.mean(np.abs(np.diff(interpolated)))
            study = optuna.create_study(direction='minimize')
            study.optimize(objective, n_trials=15)
            return study.best_params['si']
        else:  # FS
            return (subset[1] - subset[0]) / np.sqrt((subset[-1] - subset[0])**2 + (subset[1] - subset[0])**2)

    def fractal_interpolation(self, subset: np.ndarray, si: float, n_points: int = 17) -> np.ndarray:
        """Implementa la interpolación fractal."""
        # Aquí puedes implementar tu lógica específica de interpolación
        # Por ahora usamos una interpolación lineal simple
        return np.linspace(subset[0], subset[1], n_points)

    def interpolate(self, data: np.ndarray) -> np.ndarray:
        """Aplica la interpolación fractal a todo el conjunto de datos."""
        interpolated_data = []
        for i in range(len(data) - 1):
            subset = data[i:i+2]
            si = self.calculate_si(subset)
            interpolated = self.fractal_interpolation(subset, si)
            interpolated_data.extend(interpolated)
        return np.array(interpolated_data)

# Clase para el modelo LSTM
class TimeSeriesLSTM:
    def __init__(self, input_shape: Tuple[int, int, int]):
        self.input_shape = input_shape
        
    def create_model(self, units: int) -> Sequential:
        """Crea el modelo LSTM con arquitectura mejorada."""
        model = Sequential([
            LSTM(units, return_sequences=True, input_shape=self.input_shape),
            Dropout(0.2),
            LSTM(units//2, return_sequences=False),
            Dropout(0.2),
            Dense(units//4, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model

    def prepare_data(self, data: np.ndarray, time_step: int = 60) -> Tuple[np.ndarray, np.ndarray]:
        """Prepara los datos para el entrenamiento."""
        X, y = [], []
        for i in range(len(data) - time_step - 1):
            X.append(data[i:(i + time_step), 0])
            y.append(data[i + time_step, 0])
        return np.array(X), np.array(y)

    def optimize_hyperparameters(self, X: np.ndarray, y: np.ndarray) -> dict:
        """Optimiza los hiperparámetros usando Optuna."""
        def objective(trial):
            units = trial.suggest_int('units', 32, 128, step=32)
            lr = trial.suggest_loguniform('lr', 1e-4, 1e-2)
            model = self.create_model(units)
            model.compile(optimizer='adam', loss='mean_squared_error')
            history = model.fit(X, y, batch_size=32, epochs=5, validation_split=0.2, verbose=0)
            return history.history['val_loss'][-1]
        
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=10)
        return study.best_params

# Interfaz principal de la aplicación
def main():
    st.title('Predicción de Series Temporales con Interpolación Fractal')
    st.sidebar.header('Configuración')
    
    # Carga de datos
    uploaded_file = st.sidebar.file_uploader('Sube tu archivo CSV', type=['csv'])
    if not uploaded_file:
        st.warning('Por favor sube un archivo CSV para comenzar.')
        return
        
    # Carga y preparación de datos
    data, scaled_data, scaler = load_and_prepare_data(uploaded_file)
    
    # Configuración de pestañas
    tabs = st.tabs(['Datos Originales', 'Interpolación', 'Predicción'])
    
    with tabs[0]:
        st.subheader('Datos Originales')
        st.plotly_chart(create_interactive_plot(data, 'Serie Temporal Original'))
        
    with tabs[1]:
        st.subheader('Interpolación Fractal')
        strategy = st.selectbox('Estrategia de Interpolación:', ['CHS', 'CVS', 'FS'])
        
        interpolator = FractalInterpolator(strategy)
        interpolated_data = interpolator.interpolate(scaled_data.flatten())
        
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(create_interactive_plot(
                pd.DataFrame(interpolated_data), 
                f'Datos Interpolados - {strategy}'
            ))
        
    with tabs[2]:
        st.subheader('Predicción LSTM')
        
        # Parámetros del modelo
        time_step = st.slider('Ventana temporal', 10, 100, 60)
        
        # Preparación y entrenamiento del modelo
        lstm_model = TimeSeriesLSTM((time_step, 1))
        X, y = lstm_model.prepare_data(interpolated_data.reshape(-1, 1), time_step)
        X = X.reshape((X.shape[0], X.shape[1], 1))
        
        if st.button('Entrenar Modelo'):
            with st.spinner('Entrenando modelo...'):
                best_params = lstm_model.optimize_hyperparameters(X, y)
                model = lstm_model.create_model(best_params['units'])
                history = model.fit(X, y, epochs=10, batch_size=32, validation_split=0.2)
                
                predictions = model.predict(X)
                predictions = scaler.inverse_transform(predictions.reshape(-1, 1))
                
                st.success('¡Entrenamiento completado!')
                st.line_chart(pd.DataFrame({
                    'Original': data.values.flatten(),
                    'Predicción': predictions.flatten()
                }))

if __name__ == '__main__':
    main()