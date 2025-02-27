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
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Configuración de la página
st.set_page_config(
    page_title="Predicción de Series Temporales con Interpolación Fractal",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Función para generar datos de ejemplo mejorados
def generate_sample_data(start_year=2023, periods=36, base_revenue=10000, trend_factor=0.1, 
                         seasonality_factor=0.2, cycle_factor=0.15, noise_factor=0.03):
    """
    Genera datos de ventas sintéticos con patrones reconocibles para mejorar predicciones
    """
    dates = pd.date_range(start=f'{start_year}-01-01', periods=periods, freq='M')
    months = dates.strftime('%b')
    
    # Componente de tendencia (crecimiento constante)
    trend = np.linspace(0, base_revenue * trend_factor * periods/12, periods)
    
    # Componente estacional (patrón anual)
    seasonal = base_revenue * seasonality_factor * np.sin(np.linspace(0, 2*np.pi * (periods/12), periods))
    
    # Componente cíclico (ciclo de negocio)
    cycle = base_revenue * cycle_factor * np.sin(np.linspace(0, 2*np.pi * (periods/36), periods))
    
    # Ruido controlado
    noise = np.random.normal(0, base_revenue * noise_factor, periods)
    
    # Generar revenue combinando todos los componentes
    revenue = base_revenue + trend + seasonal + cycle + noise
    
    # Generar costs con relación predecible con revenue pero con variación
    cost_ratio_base = 0.6
    cost_ratio_seasonal = 0.05 * np.sin(np.linspace(0, 4*np.pi, periods))  # Costos varían estacionalmente
    costs = revenue * (cost_ratio_base + cost_ratio_seasonal) + np.random.normal(0, base_revenue * noise_factor/2, periods)
    
    # Crear DataFrame
    df = pd.DataFrame({
        'month': months,
        'revenue': revenue.round(2),
        'cost': costs.round(2),
        'date': dates
    })
    
    # Añadir algunas columnas derivadas útiles
    df['profit'] = (df['revenue'] - df['cost']).round(2)
    df['margin'] = ((df['profit'] / df['revenue']) * 100).round(2)
    df['month_num'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    
    return df[['month', 'revenue', 'cost', 'profit', 'margin']]

# Función mejorada para generar datasets optimizados para predicción
def generate_optimized_dataset(periods=36, pattern_type='standard'):
    """
    Genera datasets optimizados para diferentes patrones de series temporales
    """
    if pattern_type == 'standard':
        # Dataset con patrones equilibrados para buena predicción general
        data = generate_sample_data(
            periods=periods, 
            seasonality_factor=0.25,  # Estacionalidad fuerte
            trend_factor=0.15,        # Tendencia clara
            cycle_factor=0.2,         # Ciclo de negocio marcado
            noise_factor=0.02         # Poco ruido
        )
    elif pattern_type == 'seasonal':
        # Dataset con fuerte componente estacional
        data = generate_sample_data(
            periods=periods, 
            seasonality_factor=0.4,   # Estacionalidad muy fuerte
            trend_factor=0.1,         # Tendencia moderada
            cycle_factor=0.1,         # Ciclo débil
            noise_factor=0.01         # Ruido mínimo
        )
    elif pattern_type == 'trend':
        # Dataset con fuerte tendencia
        data = generate_sample_data(
            periods=periods, 
            seasonality_factor=0.1,   # Estacionalidad débil
            trend_factor=0.3,         # Tendencia fuerte
            cycle_factor=0.05,        # Ciclo muy débil
            noise_factor=0.01         # Ruido mínimo
        )
    elif pattern_type == 'cyclic':
        # Dataset con fuerte componente cíclico
        data = generate_sample_data(
            periods=periods, 
            seasonality_factor=0.1,   # Estacionalidad débil
            trend_factor=0.05,        # Tendencia débil
            cycle_factor=0.35,        # Ciclo muy fuerte
            noise_factor=0.01         # Ruido mínimo
        )
    else:  # 'complex'
        # Dataset con múltiples patrones complejos
        data = generate_sample_data(
            periods=periods, 
            seasonality_factor=0.3,   # Estacionalidad fuerte
            trend_factor=0.2,         # Tendencia fuerte
            cycle_factor=0.25,        # Ciclo fuerte
            noise_factor=0.02         # Poco ruido
        )
    
    return data

def data_input_section():
    """
    Sección para entrada de datos
    """
    st.sidebar.header("Fuente de Datos")
    data_source = st.sidebar.radio(
        "Selecciona la fuente de datos:",
        ["Subir CSV", "Generar Datos de Ejemplo", "Crear Datos Personalizados", "Usar Dataset Optimizado"]
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
        periods = st.sidebar.slider("Número de meses", 12, 48, 36)
        seasonality = st.sidebar.slider("Factor de estacionalidad", 0.1, 0.5, 0.2, 0.05)
        trend = st.sidebar.slider("Factor de tendencia", 0.0, 0.3, 0.1, 0.05)
        noise = st.sidebar.slider("Factor de ruido", 0.01, 0.1, 0.03, 0.01)
        
        data = generate_sample_data(periods=periods, seasonality_factor=seasonality, 
                                    trend_factor=trend, noise_factor=noise)
        if st.sidebar.button("Descargar Datos de Ejemplo"):
            csv = data.to_csv(index=False)
            st.sidebar.download_button(
                "Descargar CSV",
                csv,
                "datos_ejemplo.csv",
                "text/csv"
            )
        return data
        
    elif data_source == "Usar Dataset Optimizado":
        # Dataset predefinido para predicciones óptimas
        periods = st.sidebar.slider("Número de meses", 24, 60, 36)
        
        pattern_type = st.sidebar.selectbox(
            "Tipo de patrón para optimizar:",
            ['standard', 'seasonal', 'trend', 'cyclic', 'complex'],
            help="Selecciona el tipo de patrón que quieres predecir"
        )
        
        # Creamos un dataset optimizado según el patrón seleccionado
        data = generate_optimized_dataset(periods=periods, pattern_type=pattern_type)
        
        if st.sidebar.button("Descargar Dataset Optimizado"):
            csv = data.to_csv(index=False)
            st.sidebar.download_button(
                "Descargar CSV",
                csv,
                "dataset_optimizado.csv",
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

    def interpolate(self, data, preserve_endpoints=True, n_points=5):
        """
        Aplica la interpolación fractal a toda la serie
        """
        scaled_data = self.scaler.fit_transform(data.reshape(-1, 1)).ravel()
        
        si = self.get_vertical_scaling_factor(scaled_data, self.strategy)
        interpolated = self.fractal_interpolate(scaled_data, si, n_points=n_points)
        
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

def evaluate_interpolation_methods(data, target_variable, n_interpolation_points, sequence_length, model_complexity):
    """
    Evalúa las diferentes estrategias de interpolación y retorna los resultados
    """
    target_data = data[target_variable].values
    
    # Diccionario para almacenar resultados
    results = {}
    
    # Para cada estrategia
    for strategy in ['CVS', 'CHS', 'FS']:
        # Crear interpolador y aplicar la estrategia
        interpolator = FractalInterpolator(strategy=strategy)
        interpolated_data = interpolator.interpolate(target_data, n_points=n_interpolation_points)
        
        # Preparar datos para el modelo
        X, y = prepare_sequences(interpolated_data, sequence_length)
        
        if len(X) > 0:
            X = X.reshape((X.shape[0], X.shape[1], 1))
            
            # Crear y entrenar modelo
            model = create_lstm_model(
                input_shape=(sequence_length, 1),
                complexity=model_complexity
            )
            
            # Entrenamiento
            history = model.fit(
                X, y,
                epochs=50,
                batch_size=32,
                validation_split=0.2,
                verbose=0
            )
            
            # Predicciones
            predictions = model.predict(X)
            
            # Métricas de rendimiento
            mse = mean_squared_error(y, predictions.flatten())
            mae = mean_absolute_error(y, predictions.flatten())
            
            # Guardar resultados
            results[strategy] = {
                'interpolated_data': interpolated_data,
                'predictions': predictions.flatten(),
                'mse': mse,
                'mae': mae,
                'history': history.history
            }
    
    return results

# Función para generar predicciones futuras
def generate_future_predictions(model, last_sequence, future_steps, interpolator=None):
    """
    Genera predicciones futuras usando el modelo entrenado
    """
    curr_sequence = last_sequence.copy()
    future_predictions = []
    
    for _ in range(future_steps):
        # Redimensionar para el modelo
        x_input = curr_sequence.reshape(1, len(curr_sequence), 1)
        # Predecir siguiente valor
        next_value = model.predict(x_input)[0][0]
        # Añadir a las predicciones
        future_predictions.append(next_value)
        # Actualizar secuencia (eliminar el primer elemento y añadir la predicción)
        curr_sequence = np.append(curr_sequence[1:], next_value)
        
    return np.array(future_predictions)

def main():
    st.title('Predicción Avanzada de Series Temporales con Interpolación Fractal')
    
    # Configuración de sidebar
    st.sidebar.header("Configuración")
    
    # Selección de estrategia
    compare_strategies = st.sidebar.checkbox("Comparar todas las estrategias", value=True)
    
    if not compare_strategies:
        strategy = st.sidebar.selectbox(
            'Estrategia de Interpolación:',
            ['CVS', 'CHS', 'FS'],
            help="CVS: Optimiza valores cercanos, CHS: Usa exponente Hurst, FS: Usa fórmula"
        )
    else:
        strategy = "Comparativa"
    
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
    
    # Parámetros de entrenamiento
    train_test_split = st.sidebar.slider(
        "División entrenamiento/prueba (%)", 
        min_value=50, 
        max_value=90, 
        value=80
    )
    
    # Predicción futura
    future_prediction = st.sidebar.checkbox("Generar predicción futura", value=False)
    if future_prediction:
        future_steps = st.sidebar.slider(
            "Número de pasos futuros a predecir", 
            min_value=1, 
            max_value=24, 
            value=6
        )
    
    # Carga de datos
    data = data_input_section()
    
    if data is not None and not data.empty:
        st.subheader("Datos Originales")
        st.dataframe(data)
        
        # Selección de variable objetivo
        target_variable = st.selectbox(
            "Variable objetivo:",
            ['revenue', 'cost', 'profit', 'margin'] if 'profit' in data.columns else ['revenue', 'cost']
        )
        
        # Preparación de datos
        target_data = data[target_variable].values
        
        if not compare_strategies:
            # Interpolación fractal con una sola estrategia
            interpolator = FractalInterpolator(strategy=strategy)
            interpolated_data = interpolator.interpolate(target_data, n_points=n_interpolation_points)
            
            # Visualización de datos originales vs interpolados
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=list(range(len(target_data))),
                y=target_data,
                mode='lines+markers',
                name='Datos Originales'
            ))
            
            # Crear índices para datos interpolados
            interpolated_x = np.linspace(0, len(target_data)-1, len(interpolated_data))
            fig.add_trace(go.Scatter(
                x=interpolated_x,
                y=interpolated_data,
                mode='lines',
                name=f'Datos Interpolados ({strategy})'
            ))
            
            fig.update_layout(
                title=f'Datos Originales vs Interpolados - Estrategia {strategy}',
                xaxis_title='Índice de Tiempo',
                yaxis_title=target_variable
            )
            
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
                            x=list(range(len(target_data))),
                            y=target_data,
                            mode='lines+markers',
                            name='Datos Originales'
                        ))
                        fig_final.add_trace(go.Scatter(
                            x=list(range(sequence_length, len(predictions) + sequence_length)),
                            y=predictions.flatten(),
                            mode='lines',
                            name='Predicciones'
                        ))
                        
                        # Predicciones futuras si se selecciona
                        if future_prediction:
                            # Obtener la última secuencia conocida
                            last_sequence = interpolated_data[-sequence_length:]
                            
                            # Generar predicciones futuras
                            future_preds = generate_future_predictions(
                                model, 
                                last_sequence, 
                                future_steps
                            )
                            
                            # Añadir a la gráfica
                            last_time_idx = len(interpolated_data) - 1
                            future_indices = list(range(last_time_idx + 1, last_time_idx + future_steps + 1))
                            
                            fig_final.add_trace(go.Scatter(
                                x=future_indices,
                                y=future_preds,
                                mode='lines+markers',
                                name='Predicción Futura',
                                line=dict(color='red', dash='dash')
                            ))
                            
                        fig_final.update_layout(
                            title='Datos Originales vs Predicciones',
                            xaxis_title='Índice de Tiempo',
                            yaxis_title=target_variable
                        )
                        
                        st.plotly_chart(fig_final)
                        
                        # Métricas de rendimiento
                        mse = np.mean((y - predictions.flatten())**2)
                        mae = np.mean(np.abs(y - predictions.flatten()))
                        
                        st.subheader("Métricas de Rendimiento")
                        col1, col2 = st.columns(2)
                        col1.metric("Error Cuadrático Medio", f"{mse:.4f}")
                        col2.metric("Error Absoluto Medio", f"{mae:.4f}")
                        
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
                        
                        # Añadir predicciones futuras si existen
                        if future_prediction:
                            future_df = pd.DataFrame({
                                'Tiempo': np.arange(len(predictions), len(predictions) + future_steps),
                                'Predicción': future_preds
                            })
                            pred_df = pd.concat([pred_df, future_df], ignore_index=True)
                        
                        csv = pred_df.to_csv(index=False)
                        st.download_button(
                            label="Descargar Predicciones",
                            data=csv,
                            file_name="predicciones.csv",
                            mime="text/csv"
                        )
                        
                    else:
                        st.warning("No hay suficientes datos para el entrenamiento")
        
        else:
            # MODO COMPARATIVO - Evaluar todas las estrategias de interpolación
            if st.button("Comparar Estrategias de Interpolación"):
                with st.spinner("Evaluando diferentes estrategias de interpolación..."):
                    # Evaluar todas las estrategias
                    results = evaluate_interpolation_methods(
                        data, 
                        target_variable, 
                        n_interpolation_points,
                        sequence_length,
                        model_complexity
                    )
                    
                    if results:
                        # Mostrar gráfica comparativa de interpolaciones
                        fig_comparison = go.Figure()
                        
                        # Datos originales
                        fig_comparison.add_trace(go.Scatter(
                            x=list(range(len(target_data))),
                            y=target_data,
                            mode='lines+markers',
                            name='Datos Originales',
                            line=dict(color='black', width=2)
                        ))
                        
                        # Colores para cada estrategia
                        colors = {'CVS': 'blue', 'CHS': 'red', 'FS': 'green'}
                        
                        # Añadir interpolaciones
                        for strategy, result in results.items():
                            interpolated_x = np.linspace(0, len(target_data)-1, len(result['interpolated_data']))
                            fig_comparison.add_trace(go.Scatter(
                                x=interpolated_x,
                                y=result['interpolated_data'],
                                mode='lines',
                                name=f'Interpolación {strategy}',
                                line=dict(color=colors[strategy])
                            ))
                        
                        fig_comparison.update_layout(
                            title='Comparación de Estrategias de Interpolación',
                            xaxis_title='Índice de Tiempo',
                            yaxis_title=target_variable
                        )
                        
                        st.plotly_chart(fig_comparison)
                        
                        # Mostrar gráfica comparativa de predicciones
                        fig_predictions = go.Figure()
                        
                        # Datos originales
                        fig_predictions.add_trace(go.Scatter(
                            x=list(range(len(target_data))),
                            y=target_data,
                            mode='lines+markers',
                            name='Datos Originales',
                            line=dict(color='black', width=2)
                        ))
                        
                        # Añadir predicciones
                        for strategy, result in results.items():
                            # Índices ajustados para alinear con los datos originales
                            pred_indices = list(range(sequence_length, sequence_length + len(result['predictions'])))
                            fig_predictions.add_trace(go.Scatter(
                                x=pred_indices,
                                y=result['predictions'],
                                mode='lines',
                                name=f'Predicción {strategy}',
                                line=dict(color=colors[strategy])
                            ))
                        
                        # Añadir predicciones futuras si se selecciona
                        if future_prediction:
                            last_time_idx = len(target_data) - 1
                            future_indices = list(range(last_time_idx + 1, last_time_idx + future_steps + 1))
                            
                            for strategy, result in results.items():
                                # Obtener la última secuencia conocida
                                last_sequence = result['interpolated_data'][-sequence_length:]
                                # Crear y entrenar un modelo con los datos interpolados
                                model = create_lstm_model(
                                    input_shape=(sequence_length, 1),
                                    complexity=model_complexity
                                )
                                
                                # Preparar los datos para LSTM con la interpolación específica
                                X_future, y_future = prepare_sequences(result['interpolated_data'], sequence_length)
                                if len(X_future) > 0:
                                    X_future = X_future.reshape((X_future.shape[0], X_future.shape[1], 1))
                                    
                                    # Entrenar el modelo para predicciones futuras
                                    model.fit(
                                        X_future, y_future,
                                        epochs=50,
                                        batch_size=32,
                                        verbose=0
                                    )
                                    
                                    # Generar predicciones futuras
                                    future_preds = generate_future_predictions(
                                        model, 
                                        last_sequence, 
                                        future_steps
                                    )
                                    
                                    # Añadir a la gráfica con estilo de línea punteada
                                    fig_predictions.add_trace(go.Scatter(
                                        x=future_indices,
                                        y=future_preds,
                                        mode='lines+markers',
                                        name=f'Predicción Futura {strategy}',
                                        line=dict(color=colors[strategy], dash='dash')
                                    ))
                        
                        fig_predictions.update_layout(
                            title='Comparación de Predicciones por Estrategia',
                            xaxis_title='Índice de Tiempo',
                            yaxis_title=target_variable
                        )
                        
                        st.plotly_chart(fig_predictions)
                        
                        # Métricas comparativas
                        st.subheader("Comparación de Métricas")
                        
                        # Crear un DataFrame para comparar métricas
                        metrics_df = pd.DataFrame({
                            'Estrategia': list(results.keys()),
                            'MSE': [results[s]['mse'] for s in results.keys()],
                            'MAE': [results[s]['mae'] for s in results.keys()]
                        })
                        
                        st.dataframe(metrics_df)
                        
                        # Identificar la mejor estrategia
                        best_strategy = metrics_df.loc[metrics_df['MSE'].idxmin(), 'Estrategia']
                        st.success(f"La estrategia con mejor rendimiento es: {best_strategy}")
                        
                        # Gráficos de pérdida durante el entrenamiento para cada estrategia
                        st.subheader("Evolución del Entrenamiento por Estrategia")
                        
                        fig_loss_comparison = go.Figure()
                        for strategy, result in results.items():
                            fig_loss_comparison.add_trace(go.Scatter(
                                y=result['history']['loss'],
                                name=f'Pérdida {strategy} (entrenamiento)'
                            ))
                            fig_loss_comparison.add_trace(go.Scatter(
                                y=result['history']['val_loss'],
                                name=f'Pérdida {strategy} (validación)',
                                line=dict(dash='dash')
                            ))
                        
                        fig_loss_comparison.update_layout(
                            title='Comparación de Curvas de Aprendizaje',
                            xaxis_title='Época',
                            yaxis_title='Pérdida'
                        )
                        st.plotly_chart(fig_loss_comparison)
                        
                        # Añadir botón para descargar resultados comparativos
                        # Primero, preparar un DataFrame con todas las predicciones
                        results_df = pd.DataFrame({'Tiempo': list(range(len(target_data)))})
                        results_df['Datos_Originales'] = target_data
                        
                        for strategy, result in results.items():
                            # Predicciones alineadas con los datos originales
                            pred_aligned = np.full(len(target_data), np.nan)
                            pred_aligned[sequence_length:sequence_length+len(result['predictions'])] = result['predictions']
                            results_df[f'Predicción_{strategy}'] = pred_aligned
                            
                            # Añadir predicciones futuras si existen
                            if future_prediction:
                                # Crear DataFrame para predicciones futuras
                                future_df = pd.DataFrame({
                                    'Tiempo': future_indices,
                                    f'Predicción_Futura_{strategy}': future_preds
                                })
                                
                                # Añadir al DataFrame principal
                                results_df = pd.merge(results_df, future_df, on='Tiempo', how='outer')
                        
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            label="Descargar Resultados Comparativos",
                            data=csv,
                            file_name="resultados_comparativos.csv",
                            mime="text/csv"
                        )
                        
                    else:
                        st.warning("No hay suficientes datos para la evaluación")

    else:
        st.info("Por favor, selecciona o genera datos para comenzar.")

# Añadir información explicativa sobre las estrategias
with st.expander("Información sobre Estrategias de Interpolación Fractal"):
    st.markdown("""
    ### Estrategias de Interpolación Fractal
    
    Esta aplicación implementa tres estrategias diferentes de interpolación fractal:
    
    1. **CVS (Close Value Strategy)**: Optimiza los valores interpolados para que sean cercanos a los datos originales. Utiliza optimización para encontrar el mejor factor de escala vertical.
    
    2. **CHS (Calculated Hurst Strategy)**: Utiliza el exponente de Hurst para determinar el factor de escala. El exponente de Hurst mide la persistencia o anti-persistencia en una serie temporal.
    
    3. **FS (Formula Strategy)**: Aplica una fórmula basada en la tasa de cambio para establecer el factor de escala vertical.
    
    ### ¿Cuándo usar cada estrategia?
    
    - **CVS**: Ideal para series con cambios graduales y patrones persistentes.
    - **CHS**: Funciona mejor para series con cierta auto-similitud estadística.
    - **FS**: Buena opción para series con tendencias lineales.
    
    La comparativa le permite identificar cuál funciona mejor para sus datos específicos.
    """)

# Añadir información sobre cómo interpretar los resultados
with st.expander("Cómo Interpretar los Resultados"):
    st.markdown("""
    ### Interpretación de Resultados
    
    - **MSE (Error Cuadrático Medio)**: Mide el promedio de los errores al cuadrado. Valores más bajos indican mejor ajuste.
    
    - **MAE (Error Absoluto Medio)**: Mide el promedio de los errores absolutos. Es menos sensible a valores atípicos que el MSE.
    
    - **Curvas de Aprendizaje**: Muestran cómo evoluciona el error durante el entrenamiento. Una buena curva debe estabilizarse sin mostrar sobreajuste (cuando la validación comienza a empeorar).
    
    - **Predicciones Futuras**: Representan la proyección más allá de los datos conocidos. Estas son inherentemente más inciertas que las predicciones dentro de la muestra.
    
    ### Mejores Prácticas
    
    1. Compare siempre múltiples estrategias para encontrar la más adecuada para sus datos.
    2. Use más puntos de interpolación para capturar detalles más finos (pero cuidado con el sobreajuste).
    3. Ajuste la longitud de secuencia según el patrón temporal que quiera capturar.
    4. Para predicciones más robustas, use el conjunto de datos optimizado que mejor se adapte a sus patrones.
    """)

if __name__ == "__main__":
    main()