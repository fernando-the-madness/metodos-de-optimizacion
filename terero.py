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
import os
import math

# Configuración de la página
st.set_page_config(
    page_title="Predicción de Series Temporales con Interpolación Fractal",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title('Predicción Avanzada de Series Temporales con Interpolación Fractal')
st.markdown("""
Esta aplicación implementa las estrategias de interpolación fractal descritas en el paper 
"Mejora de Predicciones de Series Temporales Meteorológicas mediante Interpolación Fractal".
""")

# Función para generar datos de ejemplo
def generate_sample_data(start_year=2020, periods=60, base_revenue=10000, trend_factor=0.1, seasonality_factor=0.15, noise_factor=0.05):
    """
    Genera datos sintéticos con tendencia, estacionalidad y ruido
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

# Función para cargar conjuntos de datos públicos mencionados en el paper
def load_public_datasets():
    datasets = {
        "Ninguno": None,
        "Ventas de Champú": pd.DataFrame({
            'month': pd.date_range(start='2008-01-01', periods=36, freq='M').strftime('%b'),
            'revenue': [266, 145.9, 183.1, 119.3, 180.3, 168.5, 231.8, 224.5, 192.8, 122.9, 336.5, 185.9,
                       194.3, 149.5, 210.1, 273.3, 191.4, 287, 226, 303.6, 289.9, 421.6, 264.5, 342.3, 
                       339.7, 440.4, 315.9, 439.3, 401.3, 437.4, 575.5, 407.6, 682, 475.3, 581.3, 646.9]
        }),
        "Pasajeros de Aerolíneas": pd.DataFrame({
            'month': pd.date_range(start='2008-01-01', periods=36, freq='M').strftime('%b'),
            'revenue': [112, 118, 132, 129, 121, 135, 148, 148, 136, 119, 104, 118, 
                       115, 126, 141, 135, 125, 149, 170, 170, 158, 133, 114, 140, 
                       145, 150, 178, 163, 172, 178, 199, 199, 184, 162, 146, 166]
        }),
        "Rendimiento de Cultivos": pd.DataFrame({
            'month': pd.date_range(start='2008-01-01', periods=36, freq='M').strftime('%b'),
            'revenue': [41.7, 24.0, 32.3, 37.3, 46.2, 29.0, 15.0, 40.0, 30.7, 25.5, 36.8, 23.9, 
                       43.2, 28.5, 35.1, 41.2, 50.1, 31.3, 18.2, 45.6, 35.8, 29.1, 40.2, 28.5, 
                       50.8, 32.6, 38.9, 45.8, 55.6, 33.8, 21.9, 49.7, 38.6, 33.2, 44.1, 31.7]
        }),
        "Temperatura de Brașov": pd.DataFrame({
            'month': pd.date_range(start='2008-01-01', periods=36, freq='M').strftime('%b'),
            'revenue': [5.2, 4.8, 7.1, 12.5, 16.8, 19.5, 20.1, 20.3, 14.2, 9.4, 5.1, 0.2, 
                       -0.3, 2.1, 5.9, 11.2, 17.0, 19.2, 21.5, 21.0, 15.3, 8.9, 7.2, 1.5, 
                       2.8, 3.5, 8.7, 13.8, 18.2, 21.3, 22.4, 20.6, 16.8, 10.7, 6.8, 2.0]
        })
    }
    return datasets
def data_input_section():
    """
    Sección para entrada de datos
    """
    st.sidebar.header("Fuente de Datos")
    data_source = st.sidebar.radio(
        "Selecciona la fuente de datos:",
        ["Subir CSV", "Generar Datos de Ejemplo", "Datos Públicos", "Crear Datos Personalizados"]
    )
    
    if data_source == "Subir CSV":
        uploaded_file = st.sidebar.file_uploader("Sube tu archivo CSV", type=['csv'])
        if uploaded_file:
            try:
                data = pd.read_csv(uploaded_file)
                required_cols = ['month']
                if not all(col in data.columns for col in required_cols):
                    st.sidebar.error("El CSV debe tener al menos una columna 'month' y una columna numérica para análisis")
                    return None
                return data
            except Exception as e:
                st.sidebar.error(f"Error al cargar el archivo: {e}")
                return None
        else:
            st.sidebar.info("El archivo CSV debe tener columnas: month y al menos una columna numérica")
            return None
            
    elif data_source == "Generar Datos de Ejemplo":
        periods = st.sidebar.slider("Número de meses", 6, 120, 36)
        start_year = st.sidebar.number_input("Año de inicio", 2000, 2023, 2020)
        base_value = st.sidebar.number_input("Valor base", 1000, 50000, 10000)
        trend = st.sidebar.slider("Factor de tendencia", 0.0, 0.5, 0.1)
        seasonality = st.sidebar.slider("Factor de estacionalidad", 0.0, 0.5, 0.15)
        noise = st.sidebar.slider("Factor de ruido", 0.0, 0.2, 0.05)
        
        data = generate_sample_data(
            start_year=start_year, 
            periods=periods, 
            base_revenue=base_value,
            trend_factor=trend,
            seasonality_factor=seasonality,
            noise_factor=noise
        )
        
        if st.sidebar.button("Descargar Datos de Ejemplo"):
            csv = data.to_csv(index=False)
            st.sidebar.download_button(
                "Descargar CSV",
                csv,
                "datos_ejemplo.csv",
                "text/csv"
            )
        return data
    
    elif data_source == "Datos Públicos":
        datasets = load_public_datasets()
        selected_dataset = st.sidebar.selectbox(
            "Selecciona un conjunto de datos:",
            list(datasets.keys())
        )
        
        if selected_dataset != "Ninguno" and datasets[selected_dataset] is not None:
            data = datasets[selected_dataset]
            if st.sidebar.button("Descargar Datos"):
                csv = data.to_csv(index=False)
                st.sidebar.download_button(
                    "Descargar CSV",
                    csv,
                    f"{selected_dataset.lower().replace(' ', '_')}.csv",
                    "text/csv"
                )
            return data
        else:
            st.sidebar.info("Selecciona un conjunto de datos para continuar")
            return None
        
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
            revenue = st.number_input("Valor", min_value=0.0, value=10000.0)
            
            if st.form_submit_button("Agregar Datos"):
                new_row = pd.DataFrame({
                    'month': [mes],
                    'revenue': [revenue]
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
        """
        Inicializa el interpolador fractal con la estrategia elegida
        
        Parámetros:
        strategy (str): Estrategia de interpolación:
            - CHS: Closest Hurst Strategy
            - CVS: Closest Values Strategy
            - FS: Formula Strategy
        """
        self.strategy = strategy
        self.scaler = MinMaxScaler()
        
def calculate_hurst_exponent(self, data):
        """
        Calcula el exponente de Hurst para la estrategia CHS
        
        El exponente de Hurst mide la persistencia de una serie temporal
        - H > 0.5: La serie tiene tendencia (persistente)
        - H = 0.5: La serie es aleatoria
        - H < 0.5: La serie tiende a revertir a la media (anti-persistente)
        """
        lags = range(2, min(len(data) // 2, 20))
        if len(lags) == 0:
            return 0.5  # Valor por defecto
            
        tau = [np.sqrt(np.std(np.subtract(data[lag:], data[:-lag]))) for lag in lags]
        reg = np.polyfit(np.log(lags), np.log(tau), 1)
        return reg[0]
    
def get_vertical_scaling_factor(self, subset, strategy):
        """
        Determina el factor de escala vertical según la estrategia elegida
        """
        if strategy == 'CHS':
            # Closest Hurst Strategy: usa el exponente de Hurst como guía
            h = self.calculate_hurst_exponent(subset)
            # Ajustar el factor de escala basado en el exponente de Hurst
            # Para series persistentes (H > 0.5), usamos valores positivos
            # Para series anti-persistentes (H < 0.5), usamos valores negativos
            si = (h - 0.5) * 0.9  # Escalar para mantener si entre -0.45 y 0.45
            return np.clip(si, -0.9, 0.9)
        
        elif strategy == 'CVS':
            # Closest Values Strategy: optimiza para mantener características de la serie
            def objective(trial):
                # Probar diferentes valores de si
                si = trial.suggest_float('si', -0.9, 0.9)
                interpolated = self.fractal_interpolate(subset, si)
                
                # Métricas para evaluar la calidad de la interpolación
                # 1. Preservar la variabilidad general
                var_original = np.var(np.diff(subset))
                var_interp = np.var(np.diff(interpolated))
                var_score = np.abs(var_original - var_interp) / max(var_original, 1e-10)
                
                # 2. Preservar la tendencia
                trend_original = np.polyfit(np.arange(len(subset)), subset, 1)[0]
                trend_interp = np.polyfit(np.arange(len(interpolated)), interpolated, 1)[0]
                trend_score = np.abs(trend_original - trend_interp) / max(abs(trend_original), 1e-10)
                
                # 3. Preservar la rugosidad (similar al exponente de Hurst)
                h_original = self.calculate_hurst_exponent(subset)
                h_interp = self.calculate_hurst_exponent(interpolated)
                h_score = np.abs(h_original - h_interp)
                
                # Ponderación de las métricas
                return 0.4 * var_score + 0.3 * trend_score + 0.3 * h_score
            
            # Optimización con Optuna
            study = optuna.create_study(direction='minimize')
            study.optimize(objective, n_trials=30)
            return study.best_params['si']
        
        else:  # FS - Formula Strategy
            # Formula Strategy: usa una fórmula basada en la primera derivada
            if len(subset) < 2:
                return 0.0
                
            # Calcular pendientes entre puntos sucesivos
            dx = np.arange(1, len(subset))
            dy = np.diff(subset)
            slopes = dy / dx
            
            # Si hay cambios bruscos, usar un valor más negativo para suavizar
            max_slope = np.max(np.abs(slopes))
            if max_slope > 0.2:
                return -0.3
            
            # Si hay tendencia clara, mantener la tendencia
            mean_slope = np.mean(slopes)
            if abs(mean_slope) > 0.05:
                return np.sign(mean_slope) * 0.2
                
            # Por defecto, un valor ligeramente negativo para suavizar
            return -0.1

def fractal_interpolate(self, data, si, n_points=5):
        """
        Implementa la interpolación fractal usando IFS (Iterated Function System)
        
        Parámetros:
        data: Serie de datos a interpolar
        si: Factor de escala vertical
        n_points: Número de puntos a generar entre cada par de puntos originales
        """
        if len(data) < 2:
            return data
            
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
                # Fórmula de interpolación fractal
                y[j] = y0 + (y1 - y0) * t + si * t * (1 - t) * (y1 - y0)
            
            result.extend(y[:-1] if i < len(data) - 2 else y)
        
        return np.array(result)

def interpolate(self, data, preserve_endpoints=True):
        """
        Aplica la interpolación fractal a toda la serie
        """
        if len(data) < 2:
            return data
            
        # Escalar datos para normalizar
        scaled_data = self.scaler.fit_transform(data.reshape(-1, 1)).ravel()
        
        # Obtener factor de escala vertical según la estrategia
        si = self.get_vertical_scaling_factor(scaled_data, self.strategy)
        
        # Realizar la interpolación
        interpolated = self.fractal_interpolate(scaled_data, si)
        
        # Preservar puntos originales si se requiere
        if preserve_endpoints:
            interpolated[0] = scaled_data[0]
            interpolated[-1] = scaled_data[-1]
        
        # Revertir la escala
        return self.scaler.inverse_transform(interpolated.reshape(-1, 1)).ravel()
def create_lstm_model(input_shape, complexity='medium', optimizer='adam', loss='mean_squared_error'):
    """
    Crea un modelo LSTM con diferentes niveles de complejidad
    
    Parámetros:
    input_shape: Forma de los datos de entrada (secuencia, características)
    complexity: Nivel de complejidad del modelo ('simple', 'medium', 'complex')
    optimizer: Optimizador para el entrenamiento
    loss: Función de pérdida
    
    Retorna:
    Modelo LSTM compilado
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
    
    model.compile(optimizer=optimizer, loss=loss)
    return model

def optimize_lstm_hyperparameters(X, y, seq_length, validation_split=0.2, n_trials=20):
    """
    Optimiza hiperparámetros del modelo LSTM usando Optuna
    
    Parámetros:
    X, y: Datos de entrenamiento
    seq_length: Longitud de la secuencia
    validation_split: Proporción de datos para validación
    n_trials: Número de pruebas para optimización
    
    Retorna:
    Mejores hiperparámetros encontrados
    """
    def objective(trial):
        # Parámetros a optimizar
        n_units_1 = trial.suggest_categorical('n_units_1', [50, 100, 200])
        dropout_1 = trial.suggest_float('dropout_1', 0.0, 0.5)
        
        # Número de capas LSTM
        n_layers = trial.suggest_int('n_layers', 1, 3)
        
        # Construir modelo
        model = Sequential()
        model.add(LSTM(n_units_1, return_sequences=(n_layers > 1), input_shape=(seq_length, 1)))
        model.add(Dropout(dropout_1))
        
        if n_layers >= 2:
            n_units_2 = trial.suggest_categorical('n_units_2', [25, 50, 100])
            dropout_2 = trial.suggest_float('dropout_2', 0.0, 0.5)
            model.add(LSTM(n_units_2, return_sequences=(n_layers > 2)))
            model.add(Dropout(dropout_2))
            
        if n_layers == 3:
            n_units_3 = trial.suggest_categorical('n_units_3', [25, 50])
            dropout_3 = trial.suggest_float('dropout_3', 0.0, 0.5)
            model.add(LSTM(n_units_3))
            model.add(Dropout(dropout_3))
        
        # Capa de salida
        dense_units = trial.suggest_categorical('dense_units', [0, 10, 25])
        if dense_units > 0:
            model.add(Dense(dense_units))
        model.add(Dense(1))
        
        # Compilar y entrenar
        learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
        optimizer = trial.suggest_categorical('optimizer', ['adam', 'rmsprop'])
        loss = trial.suggest_categorical('loss', ['mean_squared_error', 'mean_absolute_error', 'huber'])
        
        if optimizer == 'adam':
            opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        else:
            opt = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
            
        model.compile(optimizer=opt, loss=loss)
        
        batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
        
        # Entrenar con early stopping
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=10, restore_best_weights=True
        )
        
        history = model.fit(
            X, y,
            epochs=100,  # Máximo de épocas
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[early_stop],
            verbose=0
        )
        
        # Retornar mejor valor de validación
        return min(history.history['val_loss'])
    
    # Crear y ejecutar estudio
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)
    
    return study.best_params

def prepare_sequences(data, seq_length):
    """
    Prepara secuencias para el entrenamiento LSTM
    
    Parámetros:
    data: Serie temporal
    seq_length: Longitud de la secuencia
    
    Retorna:
    X, y: Secuencias de entrada y valores objetivo
    """
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:(i + seq_length)])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

def train_and_evaluate_lstm(data, seq_length, complexity='auto', test_size=0.2, epochs=100, batch_size=32, optimize=False):
    """
    Entrena y evalúa un modelo LSTM
    
    Parámetros:
    data: Serie temporal
    seq_length: Longitud de la secuencia
    complexity: Complejidad del modelo ('simple', 'medium', 'complex', 'auto')
    test_size: Proporción de datos para prueba
    epochs: Número de épocas para entrenamiento
    batch_size: Tamaño del lote para entrenamiento
    optimize: Si True, optimiza hiperparámetros
    
    Retorna:
    model: Modelo entrenado
    history: Historial de entrenamiento
    metrics: Métricas de rendimiento
    predictions: Predicciones
    """
    # Preparar datos
    X, y = prepare_sequences(data, seq_length)
    if len(X) == 0:
        return None, None, None, None
        
    X = X.reshape((X.shape[0], X.shape[1], 1))
    
    # Dividir en conjuntos de entrenamiento y prueba
    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Auto-seleccionar complejidad
    if complexity == 'auto':
        if len(X) < 50:
            complexity = 'simple'
        elif len(X) < 100:
            complexity = 'medium'
        else:
            complexity = 'complex'
    
    # Crear y entrenar modelo
    if optimize and len(X_train) >= 30:
        st.info("Optimizando hiperparámetros del modelo LSTM...")
        best_params = optimize_lstm_hyperparameters(X_train, y_train, seq_length, n_trials=10)
        
        # Construir modelo con mejores parámetros
        model = Sequential()
        # Configurar capas según mejores parámetros...
        # (Código detallado omitido por brevedad)
        model = create_lstm_model(
            input_shape=(seq_length, 1),
            complexity=complexity
        )
    else:
        model = create_lstm_model(
            input_shape=(seq_length, 1),
            complexity=complexity
        )
    
    # Early stopping
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=10, restore_best_weights=True
    )
    
    # Entrenar modelo
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        callbacks=[early_stop],
        verbose=0
    )
    
    # Predicciones
    predictions_train = model.predict(X_train).flatten()
    predictions_test = model.predict(X_test).flatten()
    
    # Métricas
    mse_train = np.mean((y_train - predictions_train)**2)
    mae_train = np.mean(np.abs(y_train - predictions_train))
    mse_test = np.mean((y_test - predictions_test)**2)
    mae_test = np.mean(np.abs(y_test - predictions_test))
    
    metrics = {
        'MSE_train': mse_train,
        'MAE_train': mae_train,
        'MSE_test': mse_test,
        'MAE_test': mae_test,
    }
    
    # Predicciones para todos los datos
    predictions = model.predict(X).flatten()
    
    return model, history, metrics, predictions
def visualize_data_and_interpolations(data, target_variable, interpolated_data):
    """
    Visualiza datos originales vs interpolados con Plotly
    """
    target_data = data[target_variable].values
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data['month'],
        y=target_data,
        mode='lines+markers',
        name='Datos Originales',
        line=dict(color='black', width=2),
        marker=dict(size=8)
    ))
    
    colors = {
        'CVS': 'rgb(31, 119, 180)',  # Azul
        'CHS': 'rgb(255, 127, 14)',  # Naranja
        'FS': 'rgb(44, 160, 44)'     # Verde
    }
    
    for strat, interpolated in interpolated_data.items():
        # Crear índices para datos interpolados
        interpolated_x = np.linspace(0, len(data['month'])-1, len(interpolated))
        interp_months = []
        for i in range(len(interpolated_x)):
            if int(interpolated_x[i]) >= len(data['month']):
                idx = len(data['month']) - 1
            else:
                idx = int(interpolated_x[i])
            interp_months.append(data['month'].iloc[idx])
        
        fig.add_trace(go.Scatter(
            x=interp_months,
            y=interpolated,
            mode='lines',
            name=f'Interpolación {strat}',
            line=dict(color=colors[strat], width=2)
        ))
    
    fig.update_layout(
        title=f'Serie Temporal Original e Interpolada: {target_variable}',
        xaxis_title='Mes',
        yaxis_title=target_variable,
        legend_title='Datos',
        template='plotly_white',
        height=500
    )
    
    return fig

def visualize_forecasts(data, target_variable, forecasts, forecast_intervals=None):
    """
    Visualiza predicciones y bandas de confianza
    """
    fig = go.Figure()
    
    # Datos originales
    target_data = data[target_variable].values
    fig.add_trace(go.Scatter(
        x=list(range(len(data))),
        y=target_data,
        mode='lines+markers',
        name='Datos Históricos',
        line=dict(color='black', width=2),
        marker=dict(size=8)
    ))
    
    # Agregar línea de predicción para cada modelo
    colors = ['rgb(31, 119, 180)', 'rgb(255, 127, 14)', 'rgb(44, 160, 44)']
    for i, (model_name, forecast) in enumerate(forecasts.items()):
        # Predicciones dentro del período histórico
        history_len = len(target_data)
        history_pred = forecast[:history_len]
        
        # Predicciones para el futuro
        future_pred = forecast[history_len:]
        future_x = list(range(history_len, history_len + len(future_pred)))
        
        # Agregar predicciones históricas
        fig.add_trace(go.Scatter(
            x=list(range(len(history_pred))),
            y=history_pred,
            mode='lines',
            name=f'Ajuste {model_name}',
            line=dict(color=colors[i % len(colors)], width=1.5, dash='dot')
        ))
        
        # Agregar predicciones futuras
        fig.add_trace(go.Scatter(
            x=future_x,
            y=future_pred,
            mode='lines',
            name=f'Predicción {model_name}',
            line=dict(color=colors[i % len(colors)], width=2.5)
        ))
        
        # Agregar intervalos de confianza si están disponibles
        if forecast_intervals and model_name in forecast_intervals:
            lower_bound = forecast_intervals[model_name]['lower'][history_len:]
            upper_bound = forecast_intervals[model_name]['upper'][history_len:]
            
            fig.add_trace(go.Scatter(
                x=future_x + future_x[::-1],
                y=list(upper_bound) + list(lower_bound)[::-1],
                fill='toself',
                fillcolor=f'rgba({colors[i % len(colors)].replace("rgb", "").replace("(", "").replace(")", "")}, 0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name=f'Intervalo 95% {model_name}'
            ))
    
    # Últimos x valores históricos + predicciones futuras
    last_n = min(12, len(target_data))
    x_min, x_max = len(target_data) - last_n, max(future_x) if future_x else len(target_data)
    
    # Crear etiquetas de mes para el eje x
    if 'month' in data.columns:
        months = data['month'].tolist()
        # Extender con meses futuros
        future_months = []
        if future_x:
            last_month_idx = months.index(months[-1])
            all_months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            for i in range(1, len(future_x) + 1):
                future_month_idx = (last_month_idx + i) % 12
                future_months.append(all_months[future_month_idx])
        
        all_months_labels = months + future_months
        tick_vals = list(range(len(all_months_labels)))
        
        fig.update_layout(
            xaxis=dict(
                tickmode='array',
                tickvals=tick_vals,
                ticktext=all_months_labels
            )
        )
    
    fig.update_layout(
        title=f'Predicción de {target_variable}',
        xaxis_title='Período',
        yaxis_title=target_variable,
        legend_title='Series',
        template='plotly_white',
        height=600,
        xaxis=dict(range=[x_min, x_max])
    )
    
    return fig

def generate_forecast(model, last_sequence, steps, scaler=None):
    """
    Genera predicciones futuras usando el modelo LSTM
    
    Parámetros:
    model: Modelo LSTM entrenado
    last_sequence: Última secuencia de datos observados
    steps: Número de pasos a predecir
    scaler: Scaler para normalizar y desnormalizar los datos
    
    Retorna:
    forecast: Predicciones futuras
    """
    forecast = []
    curr_seq = last_sequence.copy()
    
    for _ in range(steps):
        # Preparar secuencia para predicción
        curr_seq_reshaped = curr_seq.reshape((1, curr_seq.shape[0], 1))
        
        # Predecir siguiente valor
        next_pred = model.predict(curr_seq_reshaped)[0, 0]
        forecast.append(next_pred)
        
        # Actualizar secuencia para siguiente predicción
        curr_seq = np.append(curr_seq[1:], next_pred)
    
    return np.array(forecast)

def calculate_confidence_intervals(forecast, std_dev=0.1, z_score=1.96):
    """
    Calcula intervalos de confianza para las predicciones
    
    Parámetros:
    forecast: Predicciones puntuales
    std_dev: Desviación estándar estimada (puede ser calculada de errores históricos)
    z_score: Valor z para el nivel de confianza deseado (1.96 para 95%)
    
    Retorna:
    intervals: Diccionario con límites inferior y superior
    """
    margin = z_score * std_dev * np.abs(forecast)
    lower_bound = forecast - margin
    upper_bound = forecast + margin
    
    return {
        'lower': lower_bound,
        'upper': upper_bound
    }

def monte_carlo_forecast(model, last_sequence, steps, n_simulations=100, noise_level=0.05):
    """
    Genera múltiples trayectorias de predicción con simulación Monte Carlo
    
    Parámetros:
    model: Modelo LSTM entrenado
    last_sequence: Última secuencia de datos observados
    steps: Número de pasos a predecir
    n_simulations: Número de simulaciones a ejecutar
    noise_level: Nivel de ruido para las simulaciones
    
    Retorna:
    simulations: Matriz de simulaciones [n_simulations, steps]
    """
    simulations = np.zeros((n_simulations, steps))
    
    for i in range(n_simulations):
        curr_seq = last_sequence.copy()
        
        for j in range(steps):
            # Preparar secuencia
            curr_seq_reshaped = curr_seq.reshape((1, curr_seq.shape[0], 1))
            
            # Predecir con ruido
            next_pred = model.predict(curr_seq_reshaped)[0, 0]
            noise = np.random.normal(0, noise_level * np.abs(next_pred))
            next_pred_with_noise = next_pred + noise
            
            simulations[i, j] = next_pred_with_noise
            
            # Actualizar secuencia
            curr_seq = np.append(curr_seq[1:], next_pred_with_noise)
    
    return simulations

def main():
    # Configuración de la interfaz
    st.sidebar.title("Configuración")
    
    # Sección de entrada de datos
    data = data_input_section()
    
    if data is not None and not data.empty:
        # Seleccionar variable objetivo
        numeric_cols = [col for col in data.columns if np.issubdtype(data[col].dtype, np.number)]
        if not numeric_cols:
            st.error("No se encontraron columnas numéricas en los datos")
            return
        
        target_variable = st.sidebar.selectbox(
            "Selecciona la variable objetivo:",
            numeric_cols
        )
        
        # Mostrar datos
        st.subheader("Datos Cargados")
        st.dataframe(data)
        
        # Interpolación Fractal
        st.sidebar.subheader("Interpolación Fractal")
        apply_interpolation = st.sidebar.checkbox("Aplicar Interpolación Fractal", value=True)
        
        if apply_interpolation:
            interpolation_strategies = st.sidebar.multiselect(
                "Estrategias de Interpolación:",
                ['CVS', 'CHS', 'FS'],
                default=['CVS']
            )
            
            if interpolation_strategies:
                st.subheader("Análisis de Interpolación Fractal")
                
                target_data = data[target_variable].values
                interpolated_data = {}
                
                for strategy in interpolation_strategies:
                    interpolator = FractalInterpolator(strategy=strategy)
                    interpolated = interpolator.interpolate(target_data)
                    interpolated_data[strategy] = interpolated
                
                # Visualizar interpolaciones
                fig_interp = visualize_data_and_interpolations(data, target_variable, interpolated_data)
                st.plotly_chart(fig_interp, use_container_width=True)
                
                # Mostrar resumen de interpolación
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Estadísticas de Series Originales")
                    original_stats = {
                        "Media": np.mean(target_data),
                        "Desviación Estándar": np.std(target_data),
                        "Mínimo": np.min(target_data),
                        "Máximo": np.max(target_data),
                        "Rango": np.max(target_data) - np.min(target_data),
                        "Varianza": np.var(target_data),
                        "Exponente de Hurst": FractalInterpolator().calculate_hurst_exponent(target_data)
                    }
                    st.table(pd.DataFrame([original_stats]).T.rename(columns={0: "Valor"}))
                
                with col2:
                    st.subheader("Comparación de Interpolaciones")
                    interp_stats = {}
                    for strategy, interp_data in interpolated_data.items():
                        interp_stats[strategy] = {
                            "Media": np.mean(interp_data),
                            "Desviación Estándar": np.std(interp_data),
                            "Puntos Generados": len(interp_data),
                            "Factor de Aumento": len(interp_data) / len(target_data),
                            "Exponente de Hurst": FractalInterpolator().calculate_hurst_exponent(interp_data)
                        }
                    
                    st.table(pd.DataFrame(interp_stats))
        
        # Predicción con LSTM
        st.sidebar.subheader("Modelo LSTM")
        apply_lstm = st.sidebar.checkbox("Aplicar Modelo LSTM", value=True)
        
        if apply_lstm:
            # Parámetros del modelo
            seq_length = st.sidebar.slider("Longitud de Secuencia", 2, 20, 4)
            test_size = st.sidebar.slider("Proporción de Prueba", 0.1, 0.5, 0.2)
            complexity = st.sidebar.selectbox(
                "Complejidad del Modelo",
                ['simple', 'medium', 'complex', 'auto'],
                index=3
            )
            
            # Entrenar con datos originales o interpolados
            train_options = ['Datos Originales']
            if apply_interpolation and interpolation_strategies:
                train_options.extend([f'Interpolación {s}' for s in interpolation_strategies])
            
            training_data_option = st.sidebar.selectbox(
                "Datos para Entrenamiento",
                train_options,
                index=0
            )
            
            # Parámetros de predicción
            forecast_steps = st.sidebar.slider("Pasos a predecir", 1, 24, 6)
            
            # Opciones avanzadas
            with st.sidebar.expander("Opciones Avanzadas"):
                epochs = st.number_input("Épocas de Entrenamiento", 10, 500, 100)
                batch_size = st.selectbox("Tamaño de Lote", [16, 32, 64, 128], index=1)
                optimize_params = st.checkbox("Optimizar Hiperparámetros", value=False)
                show_validation = st.checkbox("Mostrar Métricas de Validación", value=True)
                confidence_interval = st.checkbox("Mostrar Intervalos de Confianza", value=True)
                monte_carlo = st.checkbox("Simulación Monte Carlo", value=False)
                if monte_carlo:
                    n_simulations = st.slider("Número de Simulaciones", 10, 1000, 100)
                    noise_level = st.slider("Nivel de Ruido", 0.01, 0.2, 0.05)
            
            # Entrenar y evaluar modelo
            if st.button("Entrenar Modelo y Predecir"):
                st.subheader("Entrenamiento y Predicción con LSTM")
                
                # Seleccionar datos de entrenamiento
                if training_data_option == 'Datos Originales':
                    training_data = target_data
                    training_label = 'Original'
                else:
                    strategy = training_data_option.replace('Interpolación ', '')
                    training_data = interpolated_data[strategy]
                    training_label = f'Interpolación {strategy}'
                
                with st.spinner(f'Entrenando modelo LSTM con datos {training_label}...'):
                    # Normalizar datos
                    scaler = MinMaxScaler()
                    normalized_data = scaler.fit_transform(training_data.reshape(-1, 1)).ravel()
                    
                    # Entrenar modelo
                    model, history, metrics, predictions = train_and_evaluate_lstm(
                        normalized_data, seq_length, complexity, 
                        test_size, epochs, batch_size, optimize_params
                    )
                    
                    if model is None:
                        st.error("No se pudo entrenar el modelo. Verifica los datos y parámetros.")
                        return
                    
                    # Mostrar métricas
                    st.subheader("Métricas de Rendimiento")
                    metrics_df = pd.DataFrame({
                        'MSE Entrenamiento': [metrics['MSE_train']],
                        'MAE Entrenamiento': [metrics['MAE_train']],
                        'MSE Prueba': [metrics['MSE_test']],
                        'MAE Prueba': [metrics['MAE_test']]
                    })
                    st.table(metrics_df)
                    
                    # Visualizar historial de entrenamiento
                    if show_validation:
                        fig_train = go.Figure()
                        fig_train.add_trace(go.Scatter(
                            y=history.history['loss'],
                            name='Pérdida Entrenamiento'
                        ))
                        fig_train.add_trace(go.Scatter(
                            y=history.history['val_loss'],
                            name='Pérdida Validación'
                        ))
                        fig_train.update_layout(
                            title='Historial de Entrenamiento',
                            xaxis_title='Época',
                            yaxis_title='Pérdida',
                            template='plotly_white'
                        )
                        st.plotly_chart(fig_train, use_container_width=True)
                    
                    # Generar predicciones futuras
                    st.subheader("Predicciones")
                    
                    # Obtener última secuencia
                    last_sequence = normalized_data[-seq_length:]
                    
                    # Generar predicción puntual
                    future_normalized = generate_forecast(model, last_sequence, forecast_steps)
                    future_forecast = scaler.inverse_transform(
                        future_normalized.reshape(-1, 1)
                    ).flatten()
                    
                    # Combinar predicciones con valores históricos para visualización
                    historical_predictions = scaler.inverse_transform(
                        predictions.reshape(-1, 1)
                    ).flatten()
                    
                    full_forecast = np.concatenate([historical_predictions, future_forecast])
                    
                    # Crear diccionario de forecasts
                    forecasts = {training_label: full_forecast}
                    
                    # Calcular intervalos de confianza
                    forecast_intervals = {}
                    if confidence_interval:
                        # Estimar std_dev de errores de predicción en datos de prueba
                        split_idx = int(len(normalized_data) * (1 - test_size))
                        y_true = normalized_data[split_idx:]
                        y_pred = predictions[-len(y_true):]
                        forecast_error = y_true - y_pred
                        std_dev = np.std(forecast_error)
                        
                        intervals = calculate_confidence_intervals(full_forecast, std_dev)
                        forecast_intervals[training_label] = intervals
                    
                    # Simulación Monte Carlo
                    if monte_carlo:
                        simulations = monte_carlo_forecast(
                            model, last_sequence, forecast_steps, 
                            n_simulations, noise_level
                        )
                        
                        # Desnormalizar simulaciones
                        simulations_denorm = np.zeros_like(simulations)
                        for i in range(simulations.shape[0]):
                            simulations_denorm[i] = scaler.inverse_transform(
                                simulations[i].reshape(-1, 1)
                            ).flatten()
                        
                        # Calcular percentiles para bandas de confianza
                        lower_bound = np.percentile(simulations_denorm, 5, axis=0)
                        upper_bound = np.percentile(simulations_denorm, 95, axis=0)
                        
                        # Preparar valores históricos para completar intervalo
                        hist_lower = np.full(len(historical_predictions), np.nan)
                        hist_upper = np.full(len(historical_predictions), np.nan)
                        
                        intervals = {
                            'lower': np.concatenate([hist_lower, lower_bound]),
                            'upper': np.concatenate([hist_upper, upper_bound])
                        }
                        
                        forecast_intervals[f'{training_label} (Monte Carlo)'] = intervals
                    
                    # Visualizar predicciones
                    fig_forecast = visualize_forecasts(
                        data, target_variable, forecasts, forecast_intervals
                    )
                    st.plotly_chart(fig_forecast, use_container_width=True)
                    
                    # Descargar predicciones
                    future_dates = []
                    if 'month' in data.columns:
                        # Generar meses futuros
                        months = data['month'].tolist()
                        last_month_idx = list(map(lambda x: x.lower()[:3], months)).index(months[-1].lower()[:3])
                        all_months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                        for i in range(1, forecast_steps + 1):
                            future_month_idx = (last_month_idx + i) % 12
                            future_dates.append(all_months[future_month_idx])
                    else:
                        future_dates = [f'Período+{i}' for i in range(1, forecast_steps + 1)]
                    
                    # Crear dataframe con predicciones
                    forecast_df = pd.DataFrame({
                        'Período': future_dates,
                        f'Predicción {target_variable}': future_forecast
                    })
                    
                    if confidence_interval:
                        forecast_df['Límite Inferior (95%)'] = intervals['lower'][-forecast_steps:]
                        forecast_df['Límite Superior (95%)'] = intervals['upper'][-forecast_steps:]
                    
                    st.dataframe(forecast_df)
                    
                    csv = forecast_df.to_csv(index=False)
                    st.download_button(
                        label="Descargar Predicciones",
                        data=csv,
                        file_name="predicciones.csv",
                        mime="text/csv"
                    )

if __name__ == "__main__":
    main()