import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from scipy.optimize import minimize
import datetime as dt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

# Configuración de la página
st.set_page_config(
    page_title="Optimizador de Carteras de Inversión",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título de la aplicación
st.title("Optimizador de Carteras de Inversión - Modelo de Markowitz")
st.markdown("""
Esta aplicación implementa el modelo de Markowitz para optimizar carteras de inversión.
Selecciona acciones, un periodo de tiempo y tu nivel de aversión al riesgo para encontrar tu cartera óptima.
""")

# Inicializar variables de estado en session_state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'data_ohlc' not in st.session_state:
    st.session_state.data_ohlc = None
if 'pesos_optimos' not in st.session_state:
    st.session_state.pesos_optimos = None
if 'rendimientos' not in st.session_state:
    st.session_state.rendimientos = None
if 'mostrar_simulacion' not in st.session_state:
    st.session_state.mostrar_simulacion = False
if 'simulacion_params' not in st.session_state:
    st.session_state.simulacion_params = {
        'capital_inicial': 10000,
        'horizonte': 10,
        'num_simulaciones': 1000
    }
if 'resultados_simulacion' not in st.session_state:
    st.session_state.resultados_simulacion = None

# Sidebar para entrada de datos
with st.sidebar:
    st.header("Parámetros de la Cartera")
    
    # Opción para elegir el origen de los datos
    opcion_datos = st.radio("Selecciona el origen de los datos:", ["Datos simulados", "Cargar archivo CSV", "Yahoo Finance (yfinance)"])
    
    if opcion_datos == "Datos simulados":
        st.subheader("Generar datos simulados")
        num_acciones = st.number_input("Número de acciones:", min_value=2, max_value=10, value=3)
        fecha_inicio = st.date_input("Fecha de inicio:", dt.date(2020, 1, 1))
        fecha_fin = st.date_input("Fecha de fin:", dt.date.today())
        
        # Generar datos simulados
        np.random.seed(42)
        fechas = pd.date_range(start=fecha_inicio, end=fecha_fin, freq="D")
        precios = np.cumprod(1 + np.random.normal(0.001, 0.02, (len(fechas), num_acciones)), axis=0) * 100
        st.session_state.data = pd.DataFrame(precios, index=fechas, columns=[f"Acción {i+1}" for i in range(num_acciones)])
        
        # Mostrar datos simulados
        with st.expander("Ver datos simulados", expanded=False):
            st.dataframe(st.session_state.data)
    
    elif opcion_datos == "Cargar archivo CSV":
        st.subheader("Cargar archivo CSV")
        archivo = st.file_uploader("Sube un archivo CSV con precios históricos:", type=["csv"])
        
        if archivo is not None:
            st.session_state.data = pd.read_csv(archivo, index_col="Date", parse_dates=True)
            with st.expander("Ver datos cargados", expanded=False):
                st.dataframe(st.session_state.data)
        else:
            st.warning("Por favor, sube un archivo CSV.")
            st.session_state.data = None
    
    else:  # Yahoo Finance
        st.subheader("Yahoo Finance (yfinance)")
        
        # Entrada de acciones
        acciones_input = st.text_input("Ingresa símbolos de acciones separados por comas (ej: AAPL,MSFT,AMZN):")
        acciones_seleccionadas = [accion.strip() for accion in acciones_input.split(',')] if acciones_input else []
        
        # Verificar acciones y periodos disponibles
        if acciones_seleccionadas:
            st.markdown("**Información de las acciones seleccionadas:**")
            for accion in acciones_seleccionadas:
                try:
                    ticker = yf.Ticker(accion)
                    info = ticker.history(period="max")
                    if not info.empty:
                        fecha_min = info.index.min().strftime('%Y-%m-%d')
                        fecha_max = info.index.max().strftime('%Y-%m-%d')
                        st.write(f"**{accion}**: Datos disponibles desde {fecha_min} hasta {fecha_max}.")
                    else:
                        st.warning(f"No hay datos disponibles para {accion}.")
                except:
                    st.warning(f"No se pudo obtener información para {accion}.")
        
        # Periodo de tiempo
        st.subheader("Periodo de Tiempo")
        fecha_inicio = st.date_input("Fecha de inicio:", dt.date(2020, 1, 1))
        fecha_fin = st.date_input("Fecha de fin:", dt.date.today())
        
        # Descargar datos de Yahoo Finance
        if acciones_seleccionadas:
            try:
                with st.spinner('Descargando datos de Yahoo Finance...'):
                    # Descargar datos completos (OHLCV) para gráficos de precios
                    data_completa = yf.download(acciones_seleccionadas, start=fecha_inicio, end=fecha_fin)
                    st.session_state.data = data_completa['Adj Close']
                    if st.session_state.data.empty:
                        st.error("No se encontraron datos para las acciones y el periodo seleccionado.")
                        st.session_state.data = None
                        st.session_state.data_ohlc = None
                    else:
                        with st.expander("Ver datos descargados", expanded=False):
                            st.dataframe(st.session_state.data)
                            
                        # Guardar datos completos para gráficos
                        st.session_state.data_ohlc = data_completa
            except Exception as e:
                st.error(f"Error al descargar datos: {e}")
                st.session_state.data = None
                st.session_state.data_ohlc = None
        else:
            st.warning("Ingresa al menos un símbolo de acción válido.")
            st.session_state.data = None
            st.session_state.data_ohlc = None
    
    # Parámetros de optimización
    st.subheader("Preferencia de Riesgo-Rendimiento")
    aversion_riesgo = st.slider("Nivel de aversión al riesgo (mayor = más conservador):", 0.0, 10.0, 2.0, 0.1)
    
    # Opciones adicionales
    st.subheader("Opciones Adicionales")
    
    # Restricciones personalizadas (limitando el máximo peso por activo)
    max_peso_por_activo = st.slider("Peso máximo por activo:", 0.1, 1.0, 1.0, 0.05, 
                                  help="Limita el porcentaje máximo que puede tener un activo en la cartera")
    
    # Botón para calcular
    if st.button("Calcular Cartera Óptima", type="primary"):
        st.session_state.mostrar_simulacion = False

# Funciones para optimización de carteras
def calcular_rendimientos(precios):
    """Calcula los rendimientos logarítmicos diarios."""
    return np.log(precios / precios.shift(1)).dropna()

def calcular_estadisticas_cartera(pesos, rendimientos):
    """Calcula el rendimiento y riesgo esperado de una cartera."""
    rendimiento_cartera = np.sum(rendimientos.mean() * pesos) * 252  # Anualizado
    volatilidad_cartera = np.sqrt(np.dot(pesos.T, np.dot(rendimientos.cov() * 252, pesos)))
    return rendimiento_cartera, volatilidad_cartera

def max_sharpe_ratio(rendimientos, max_peso=1.0):
    """Encuentra la cartera con el máximo Ratio de Sharpe."""
    num_activos = len(rendimientos.columns)
    args = (rendimientos)
    restricciones = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    limite = tuple((0, max_peso) for _ in range(num_activos))
    
    def neg_sharpe(pesos, rendimientos):
        r, v = calcular_estadisticas_cartera(pesos, rendimientos)
        return -r/v
    
    pesos_iniciales = np.array([1/num_activos] * num_activos)
    resultado = minimize(neg_sharpe, pesos_iniciales, args=args, method='SLSQP', bounds=limite, constraints=restricciones)
    return resultado['x']

def min_varianza(rendimientos, max_peso=1.0):
    """Encuentra la cartera de mínima varianza."""
    num_activos = len(rendimientos.columns)
    args = (rendimientos)
    restricciones = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    limite = tuple((0, max_peso) for _ in range(num_activos))
    
    def varianza(pesos, rendimientos):
        return calcular_estadisticas_cartera(pesos, rendimientos)[1]**2
    
    pesos_iniciales = np.array([1/num_activos] * num_activos)
    resultado = minimize(varianza, pesos_iniciales, args=args, method='SLSQP', bounds=limite, constraints=restricciones)
    return resultado['x']

def cartera_optima(rendimientos, aversion_riesgo, max_peso=1.0):
    """Encuentra la cartera óptima según la aversión al riesgo del inversor."""
    num_activos = len(rendimientos.columns)
    args = (rendimientos, aversion_riesgo)
    restricciones = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    limite = tuple((0, max_peso) for _ in range(num_activos))
    
    def utilidad(pesos, rendimientos, aversion_riesgo):
        r, v = calcular_estadisticas_cartera(pesos, rendimientos)
        return -(r - aversion_riesgo * v**2)
    
    pesos_iniciales = np.array([1/num_activos] * num_activos)
    resultado = minimize(utilidad, pesos_iniciales, args=args, method='SLSQP', bounds=limite, constraints=restricciones)
    return resultado['x']

def generar_frontera_eficiente(rendimientos, max_peso=1.0, num_carteras=100):
    """Genera puntos a lo largo de la frontera eficiente."""
    rendimientos_carteras = []
    volatilidades_carteras = []
    pesos_todas_carteras = []
    
    num_activos = len(rendimientos.columns)
    
    # Cartera de min varianza y max sharpe para definir los límites
    pesos_min_var = min_varianza(rendimientos, max_peso)
    ret_min_var, vol_min_var = calcular_estadisticas_cartera(pesos_min_var, rendimientos)
    
    pesos_max_sharpe = max_sharpe_ratio(rendimientos, max_peso)
    ret_max_sharpe, vol_max_sharpe = calcular_estadisticas_cartera(pesos_max_sharpe, rendimientos)
    
    # Gama de objetivos de rendimiento
    rendimiento_min = ret_min_var
    rendimiento_max = ret_max_sharpe * 1.2  # Un poco más para extender la frontera
    
    rendimientos_objetivo = np.linspace(rendimiento_min, rendimiento_max, num_carteras)
    
    # Para cada rendimiento objetivo, encontrar la cartera de mínima varianza
    for objetivo in rendimientos_objetivo:
        restricciones = (
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
            {'type': 'eq', 'fun': lambda x: calcular_estadisticas_cartera(x, rendimientos)[0] - objetivo}
        )
        limite = tuple((0, max_peso) for _ in range(num_activos))
        
        def volatilidad(pesos, rendimientos):
            return calcular_estadisticas_cartera(pesos, rendimientos)[1]
        
        pesos_iniciales = np.array([1/num_activos] * num_activos)
        
        try:
            resultado = minimize(volatilidad, pesos_iniciales, args=(rendimientos), method='SLSQP', bounds=limite, constraints=restricciones)
            if resultado['success']:
                pesos = resultado['x']
                ret, vol = calcular_estadisticas_cartera(pesos, rendimientos)
                
                rendimientos_carteras.append(ret)
                volatilidades_carteras.append(vol)
                pesos_todas_carteras.append(pesos)
        except:
            pass
    
    return volatilidades_carteras, rendimientos_carteras, pesos_todas_carteras

def crear_grafico_precio_accion(data_ohlc, ticker):
    """Crea un gráfico de velas para una acción."""
    df = data_ohlc.xs(ticker, axis=1, level=1)
    
    fig = go.Figure(data=[go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name=ticker
    )])
    
    # Añadir línea de precio ajustado
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['Adj Close'],
        line=dict(color='rgba(0, 230, 115, 0.8)', width=1),
        name='Precio Ajustado'
    ))
    
    # Añadir volumen como barras en un subplot
    fig.add_trace(go.Bar(
        x=df.index,
        y=df['Volume'],
        marker_color='rgba(128, 128, 128, 0.5)',
        name='Volumen',
        yaxis='y2'
    ))
    
    # Actualizar el diseño
    fig.update_layout(
        title=f'Precio histórico de {ticker}',
        xaxis_title='Fecha',
        yaxis_title='Precio',
        height=500,
        xaxis_rangeslider_visible=False,
        yaxis2=dict(
            title='Volumen',
            overlaying='y',
            side='right',
            showgrid=False
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        template='plotly_dark',
        margin=dict(l=50, r=50, t=50, b=50)
    )
    
    return fig

def crear_grafico_precios_normalizados(data):
    """Crea un gráfico de precios normalizados para todas las acciones."""
    # Normalizar precios (base 100)
    df_norm = data.copy()
    for col in df_norm.columns:
        df_norm[col] = df_norm[col] / df_norm[col].iloc[0] * 100
    
    fig = px.line(df_norm, x=df_norm.index, y=df_norm.columns,
                 title='Evolución de precios normalizados (Base 100)')
    
    fig.update_layout(
        xaxis_title='Fecha',
        yaxis_title='Precio (Base 100)',
        height=400,
        legend_title='Activos',
        hovermode='x unified',
        template='plotly_dark',
        margin=dict(l=50, r=50, t=50, b=50)
    )
    
    return fig

def crear_grafico_correlacion(rendimientos):
    """Crea un mapa de calor de correlaciones entre activos."""
    corr = rendimientos.corr()
    
    fig = px.imshow(corr, 
                   text_auto='.2f',
                   color_continuous_scale='RdBu_r',
                   title='Matriz de Correlación entre Activos')
    
    fig.update_layout(
        height=500,
        template='plotly_dark',
        margin=dict(l=50, r=50, t=50, b=50)
    )
    
    return fig

def ejecutar_simulacion(rendimientos, pesos_optimos, capital_inicial, horizonte, num_simulaciones):
    """Ejecuta la simulación Monte Carlo y devuelve los resultados."""
    # Parámetros de la simulación
    dias_por_anio = 252
    total_dias = dias_por_anio * horizonte
    
    # Media y covarianza de los rendimientos diarios
    media_diaria = rendimientos.mean()
    cov_diaria = rendimientos.cov()
    
    # Inicializar la matriz de resultados
    resultados_simulacion = np.zeros((num_simulaciones, total_dias))
    resultados_simulacion[:, 0] = capital_inicial
    
    # Ejecutar simulaciones
    np.random.seed(42)
    for sim in range(num_simulaciones):
        # Generar rendimientos aleatorios correlacionados
        Z = np.random.multivariate_normal(media_diaria, cov_diaria, total_dias)
        rendimientos_cartera = np.sum(Z * pesos_optimos, axis=1)
        
        # Acumular capital
        for t in range(1, total_dias):
            resultados_simulacion[sim, t] = resultados_simulacion[sim, t-1] * (1 + rendimientos_cartera[t])
    
    # Convertir a DataFrame para visualización
    fechas_futuras = pd.date_range(start=dt.date.today(), periods=total_dias, freq='B')
    df_resultados = pd.DataFrame(resultados_simulacion.T, index=fechas_futuras)
    
    # Calcular estadísticas
    percentiles = np.percentile(df_resultados.iloc[-1], [5, 25, 50, 75, 95])
    capital_final_medio = df_resultados.iloc[-1].mean()
    capital_final_mediano = percentiles[2]
    
    # Crear DataFrame con percentiles para gráficos
    df_percentiles = pd.DataFrame({
        'P5': np.percentile(df_resultados, 5, axis=1),
        'P25': np.percentile(df_resultados, 25, axis=1),
        'P50': np.percentile(df_resultados, 50, axis=1),
        'P75': np.percentile(df_resultados, 75, axis=1),
        'P95': np.percentile(df_resultados, 95, axis=1)
    }, index=fechas_futuras)
    
    # Calcular probabilidades
    prob_positivo = (df_resultados.iloc[-1] > capital_inicial).mean() * 100
    prob_doble = (df_resultados.iloc[-1] > capital_inicial * 2).mean() * 100
    prob_triple = (df_resultados.iloc[-1] > capital_inicial * 3).mean() * 100
    df_prob = pd.DataFrame({
        'Escenario': ['Rendimiento Positivo', 'Duplicar Capital', 'Triplicar Capital'],
        'Probabilidad (%)': [prob_positivo, prob_doble, prob_triple]
    })
    
    return {
        'df_resultados': df_resultados,
        'df_percentiles': df_percentiles,
        'percentiles': percentiles,
        'capital_final_medio': capital_final_medio,
        'capital_final_mediano': capital_final_mediano,
        'df_prob': df_prob,
        'fechas_futuras': fechas_futuras,
        'capital_inicial': capital_inicial,
        'horizonte': horizonte
    }

# Función para mostrar resultados de simulación
def mostrar_resultados_simulacion(resultados):
    """Muestra los gráficos y estadísticas de la simulación."""
    # Desempaquetar resultados
    df_resultados = resultados['df_resultados']
    df_percentiles = resultados['df_percentiles']
    percentiles = resultados['percentiles']
    capital_final_mediano = resultados['capital_final_mediano']
    df_prob = resultados['df_prob']
    fechas_futuras = resultados['fechas_futuras']
    capital_inicial = resultados['capital_inicial']
    horizonte = resultados['horizonte']
    
    # Gráfico de simulación
    fig_sim = go.Figure()
    
    # Añadir todas las simulaciones con baja opacidad
    for i in range(min(100, df_resultados.shape[1])):  # limitamos a 100 líneas para rendimiento
        fig_sim.add_trace(go.Scatter(
            x=fechas_futuras,
            y=df_resultados[i],
            mode='lines',
            line=dict(width=1, color='rgba(100, 100, 100, 0.1)'),
            showlegend=False
        ))
    
    # Añadir percentiles
    fig_sim.add_trace(go.Scatter(
        x=fechas_futuras,
        y=df_percentiles['P50'],
        mode='lines',
        line=dict(width=3, color='blue'),
        name='Mediana'
    ))
    
    fig_sim.add_trace(go.Scatter(
        x=fechas_futuras,
        y=df_percentiles['P75'],
        mode='lines',
        line=dict(width=2, color='green'),
        name='Percentil 75'
    ))
    
    fig_sim.add_trace(go.Scatter(
        x=fechas_futuras,
        y=df_percentiles['P25'],
        mode='lines',
        line=dict(width=2, color='orange'),
        name='Percentil 25'
    ))
    
    fig_sim.add_trace(go.Scatter(
        x=fechas_futuras,
        y=df_percentiles['P95'],
        mode='lines',
        line=dict(width=1, color='darkgreen', dash='dash'),
        name='Percentil 95'
    ))
    
    fig_sim.add_trace(go.Scatter(
        x=fechas_futuras,
        y=df_percentiles['P5'],
        mode='lines',
        line=dict(width=1, color='red', dash='dash'),
        name='Percentil 5'
    ))
    
    # Configuración del gráfico
    fig_sim.update_layout(
        title=f'Simulación Monte Carlo: Evolución de Capital (${capital_inicial:,} por {horizonte} años)',
        xaxis_title='Fecha',
        yaxis_title='Capital ($)',
        height=600,
        template='plotly_dark',
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    st.plotly_chart(fig_sim, use_container_width=True)
    
    # Mostrar estadísticas de la simulación
    st.subheader("Resultados de la Simulación")
    col_sim1, col_sim2, col_sim3 = st.columns(3)
    
    with col_sim1:
        st.metric("Capital Inicial", f"${capital_inicial:,}")
        st.metric("Capital Final (Mediana)", f"${capital_final_mediano:,.2f}")
    
    with col_sim2:
        st.metric("Rendimiento Total (Mediano)", f"{(capital_final_mediano/capital_inicial - 1) * 100:.2f}%")
        st.metric("Rendimiento Anual (Mediano)", f"{((capital_final_mediano/capital_inicial)**(1/horizonte) - 1) * 100:.2f}%")
    
    with col_sim3:
        st.metric("Mejor Escenario (P95)", f"${percentiles[4]:,.2f}")
        st.metric("Peor Escenario (P5)", f"${percentiles[0]:,.2f}")
    
    # Histograma de resultados finales
    fig_hist = px.histogram(
        df_resultados.iloc[-1],
        nbins=50,
        title="Distribución de Capital Final",
        labels={'value': 'Capital Final ($)', 'count': 'Frecuencia'},
        color_discrete_sequence=['green']
    )
    
    fig_hist.add_vline(x=capital_inicial, line_dash="dash", line_color="red", annotation_text="Capital Inicial")
    fig_hist.add_vline(x=capital_final_mediano, line_dash="solid", line_color="blue", annotation_text="Mediana")
    
    fig_hist.update_layout(
        template='plotly_dark',
        height=400
    )
    
    st.plotly_chart(fig_hist, use_container_width=True)
    
    # Tabla de probabilidades
    st.subheader("Probabilidades de Rendimiento")
    st.dataframe(df_prob.set_index('Escenario').style.format({
        'Probabilidad (%)': '{:.2f}'
    }))

# Función principal para mostrar resultados
def mostrar_resultados(data, data_ohlc, aversion_riesgo, max_peso_por_activo):
    if data is None:
        st.error("No hay datos disponibles. Por favor, carga un archivo o genera datos simulados.")
        return
    
    # Calcular rendimientos
    st.session_state.rendimientos = calcular_rendimientos(data)
    rendimientos = st.session_state.rendimientos
    
    # Panel de pestañas para organizar la información
    tab1, tab2, tab3 = st.tabs(["Cartera Óptima", "Análisis de Activos", "Simulación de Inversión"])
    
    with tab1:
        # Calcular carteras óptimas
        with st.spinner('Calculando carteras óptimas...'):
            pesos_min_var = min_varianza(rendimientos, max_peso_por_activo)
            ret_min_var, vol_min_var = calcular_estadisticas_cartera(pesos_min_var, rendimientos)
            
            pesos_max_sharpe = max_sharpe_ratio(rendimientos, max_peso_por_activo)
            ret_max_sharpe, vol_max_sharpe = calcular_estadisticas_cartera(pesos_max_sharpe, rendimientos)
            
            st.session_state.pesos_optimos = cartera_optima(rendimientos, aversion_riesgo, max_peso_por_activo)
            pesos_optimos = st.session_state.pesos_optimos
            ret_optimo, vol_optimo = calcular_estadisticas_cartera(pesos_optimos, rendimientos)
            
            # Generar frontera eficiente
            volatilidades, rendimientos_ef, pesos_ef = generar_frontera_eficiente(rendimientos, max_peso_por_activo)
        
        st.header("Resultados de la Optimización")
        
        # Mostrar asignación de activos en la cartera óptima
        col1, col2 = st.columns([3, 2])
        
        with col1:
            # Gráfico de la frontera eficiente con Plotly
            fig_ef = go.Figure()
            
            # Frontera eficiente
            fig_ef.add_trace(go.Scatter(
                x=[vol*100 for vol in volatilidades],
                y=[ret*100 for ret in rendimientos_ef],
                mode='lines',
                name='Frontera Eficiente',
                line=dict(color='blue', width=4)
            ))
            
            # Activos individuales
            for i, accion in enumerate(rendimientos.columns):
                ret_anual = rendimientos.mean()[i] * 252 * 100
                vol_anual = rendimientos.std()[i] * np.sqrt(252) * 100
                fig_ef.add_trace(go.Scatter(
                    x=[vol_anual],
                    y=[ret_anual],
                    mode='markers',
                    name=accion,
                    marker=dict(size=12, symbol='circle')
                ))
            
            # Carteras clave
            fig_ef.add_trace(go.Scatter(
                x=[vol_min_var*100],
                y=[ret_min_var*100],
                mode='markers',
                name='Mínima Varianza',
                marker=dict(size=16, symbol='star', color='green')
            ))
            
            fig_ef.add_trace(go.Scatter(
                x=[vol_max_sharpe*100],
                y=[ret_max_sharpe*100],
                mode='markers',
                name='Máximo Sharpe',
                marker=dict(size=16, symbol='star', color='red')
            ))
            
            fig_ef.add_trace(go.Scatter(
                x=[vol_optimo*100],
                y=[ret_optimo*100],
                mode='markers',
                name='Tu Cartera Óptima',
                marker=dict(size=16, symbol='star', color='orange')
            ))
            
            # Configuración del gráfico
            fig_ef.update_layout(
                title='Frontera Eficiente de Markowitz',
                xaxis_title='Volatilidad Anualizada (%)',
                yaxis_title='Rendimiento Esperado Anualizado (%)',
                height=600,
                template='plotly_dark',
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01
                ),
                hovermode='closest'
            )
            
            st.plotly_chart(fig_ef, use_container_width=True)
        
        with col2:
            # Mostrar estadísticas de la cartera óptima
            st.subheader("Estadísticas de tu Cartera Óptima")
            
            st.metric("Rendimiento Esperado (Anual)", f"{ret_optimo*100:.2f}%")
            st.metric("Volatilidad (Anual)", f"{vol_optimo*100:.2f}%")
            st.metric("Ratio de Sharpe", f"{ret_optimo/vol_optimo:.2f}")
            
            # Gráfico de composición de la cartera
            fig_pie = px.pie(
                values=pesos_optimos*100, 
                names=rendimientos.columns,
                title="Composición de la Cartera Óptima"
            )
            fig_pie.update_layout(template='plotly_dark')
            st.plotly_chart(fig_pie, use_container_width=True)
            
            # Tabla de asignación de activos
            st.subheader("Asignación de Activos")
            df_pesos = pd.DataFrame({
                'Activo': rendimientos.columns,
                'Peso (%)': pesos_optimos * 100
            }).sort_values(by='Peso (%)', ascending=False)
            
            st.dataframe(df_pesos.style.format({
                'Peso (%)': '{:.2f}'
            }))
            
            # Exportar pesos a CSV
            csv = df_pesos.to_csv(index=False)
            st.download_button("Descargar Asignación de Activos", 
                             data=csv, 
                             file_name="cartera_optima.csv",
                             mime="text/csv")
        
        # Comparación de carteras
        st.subheader("Comparación de Carteras Óptimas")
        col_comp1, col_comp2, col_comp3 = st.columns(3)
        
        with col_comp1:
            st.markdown("**Cartera de Mínima Varianza**")
            st.metric("Rendimiento Esperado", f"{ret_min_var*100:.2f}%")
            st.metric("Volatilidad", f"{vol_min_var*100:.2f}%")
            st.metric("Ratio de Sharpe", f"{ret_min_var/vol_min_var:.2f}")
        
        with col_comp2:
            st.markdown("**Tu Cartera Óptima**")
            st.metric("Rendimiento Esperado", f"{ret_optimo*100:.2f}%")
            st.metric("Volatilidad", f"{vol_optimo*100:.2f}%")
            st.metric("Ratio de Sharpe", f"{ret_optimo/vol_optimo:.2f}")
        
        with col_comp3:
            st.markdown("**Cartera de Máximo Sharpe**")
            st.metric("Rendimiento Esperado", f"{ret_max_sharpe*100:.2f}%")
            st.metric("Volatilidad", f"{vol_max_sharpe*100:.2f}%")
            st.metric("Ratio de Sharpe", f"{ret_max_sharpe/vol_max_sharpe:.2f}")
    
    with tab2:
        st.header("Análisis de Activos")
        
        # Gráficos de precios
        if data_ohlc is not None:
            st.subheader("Gráficos de Precios")
            accion_seleccionada = st.selectbox("Selecciona una acción para ver su gráfico de velas:", 
                                             options=rendimientos.columns,
                                             index=0)
            
            # Mostrar gráfico de velas
            fig_velas = crear_grafico_precio_accion(data_ohlc, accion_seleccionada)
            st.plotly_chart(fig_velas, use_container_width=True)
        
        # Gráfico de precios normalizados
        st.subheader("Evolución de Precios Normalizados")
        fig_norm = crear_grafico_precios_normalizados(data)
        st.plotly_chart(fig_norm, use_container_width=True)
        
        # Estadísticas descriptivas de rendimientos
        st.subheader("Estadísticas de Rendimientos")
        stats_rendimientos = rendimientos.describe().T
        stats_rendimientos['anual_return'] = rendimientos.mean() * 252 * 100
        stats_rendimientos['anual_volatility'] = rendimientos.std() * np.sqrt(252) * 100
        stats_rendimientos['sharpe_ratio'] = stats_rendimientos['anual_return'] / stats_rendimientos['anual_volatility']
        
        # Formatear estadísticas
        stats_display = stats_rendimientos[['anual_return', 'anual_volatility', 'sharpe_ratio']].rename(columns={
            'anual_return': 'Rendimiento Anual (%)',
            'anual_volatility': 'Volatilidad Anual (%)',
            'sharpe_ratio': 'Ratio de Sharpe'
        })
        
        st.dataframe(stats_display.style.format({
            'Rendimiento Anual (%)': '{:.2f}',
            'Volatilidad Anual (%)': '{:.2f}',
            'Ratio de Sharpe': '{:.2f}'
        }))
        
        # Matriz de correlación
        st.subheader("Matriz de Correlación")
        fig_corr = crear_grafico_correlacion(rendimientos)
        st.plotly_chart(fig_corr, use_container_width=True)
        
        # Distribución de rendimientos
        st.subheader("Distribución de Rendimientos")
        activo_hist = st.selectbox("Selecciona un activo para ver la distribución de sus rendimientos:",
                                 options=rendimientos.columns,
                                 index=0)
        
        fig_hist = px.histogram(
            rendimientos[activo_hist],
            nbins=50,
            title=f"Distribución de Rendimientos Diarios: {activo_hist}",
            labels={'value': 'Rendimiento Diario', 'count': 'Frecuencia'},
            color_discrete_sequence=['blue']
        )
        
        # Añadir línea de la media
        fig_hist.add_vline(x=rendimientos[activo_hist].mean(), line_dash="dash", line_color="red", 
                         annotation_text="Media")
        
        fig_hist.update_layout(
            template='plotly_dark',
            height=400
        )
        
        st.plotly_chart(fig_hist, use_container_width=True)
        
        # Mostrar detalles adicionales
        st.subheader("Resumen de Estadísticas por Activo")
        st.dataframe(rendimientos.describe().T.style.format('{:.4f}'))
    
    with tab3:
        st.header("Simulación de Inversión")
        
        # Parámetros de la simulación
        st.subheader("Configuración de la Simulación Monte Carlo")
        
        col_sim1, col_sim2, col_sim3 = st.columns(3)
        
        with col_sim1:
            capital_inicial = st.number_input("Capital Inicial ($):", 
                                            min_value=1000, 
                                            max_value=10000000, 
                                            value=st.session_state.simulacion_params['capital_inicial'],
                                            step=1000)
        
        with col_sim2:
            horizonte = st.number_input("Horizonte de Inversión (años):", 
                                       min_value=1, 
                                       max_value=30, 
                                       value=st.session_state.simulacion_params['horizonte'])
        
        with col_sim3:
            num_simulaciones = st.number_input("Número de Simulaciones:", 
                                            min_value=100, 
                                            max_value=10000, 
                                            value=st.session_state.simulacion_params['num_simulaciones'],
                                            step=100)
        
        # Guardar parámetros
        st.session_state.simulacion_params = {
            'capital_inicial': capital_inicial,
            'horizonte': horizonte,
            'num_simulaciones': num_simulaciones
        }
        
        if st.button("Ejecutar Simulación", type="primary"):
            if st.session_state.pesos_optimos is not None and st.session_state.rendimientos is not None:
                with st.spinner("Ejecutando simulación Monte Carlo..."):
                    st.session_state.resultados_simulacion = ejecutar_simulacion(
                        st.session_state.rendimientos,
                        st.session_state.pesos_optimos,
                        capital_inicial,
                        horizonte,
                        num_simulaciones
                    )
                st.session_state.mostrar_simulacion = True
            else:
                st.error("Necesitas calcular primero la cartera óptima.")
        
        if st.session_state.mostrar_simulacion and st.session_state.resultados_simulacion is not None:
            mostrar_resultados_simulacion(st.session_state.resultados_simulacion)

# Ejecutar la aplicación
if st.session_state.data is not None:
    mostrar_resultados(st.session_state.data, st.session_state.data_ohlc, aversion_riesgo, max_peso_por_activo)