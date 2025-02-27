import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from scipy.optimize import minimize
import datetime as dt

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
        data = pd.DataFrame(precios, index=fechas, columns=[f"Acción {i+1}" for i in range(num_acciones)])
        
        # Mostrar datos simulados
        with st.expander("Ver datos simulados", expanded=False):
            st.dataframe(data)
    
    elif opcion_datos == "Cargar archivo CSV":
        st.subheader("Cargar archivo CSV")
        archivo = st.file_uploader("Sube un archivo CSV con precios históricos:", type=["csv"])
        
        if archivo is not None:
            data = pd.read_csv(archivo, index_col="Date", parse_dates=True)
            with st.expander("Ver datos cargados", expanded=False):
                st.dataframe(data)
        else:
            st.warning("Por favor, sube un archivo CSV.")
            data = None
    
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
                    data = yf.download(acciones_seleccionadas, start=fecha_inicio, end=fecha_fin)['Adj Close']
                    # También descargamos los datos completos para los gráficos
                    data_completa = yf.download(acciones_seleccionadas, start=fecha_inicio, end=fecha_fin)
                    if data.empty:
                        st.error("No se encontraron datos para las acciones y el periodo seleccionado.")
                        data = None
                        data_completa = None
                    else:
                        with st.expander("Ver datos descargados", expanded=False):
                            st.dataframe(data)
            except Exception as e:
                st.error(f"Error al descargar datos: {e}")
                data = None
                data_completa = None
        else:
            st.warning("Ingresa al menos un símbolo de acción válido.")
            data = None
            data_completa = None
    
    # Parámetros de optimización
    st.subheader("Preferencia de Riesgo-Rendimiento")
    aversion_riesgo = st.slider("Nivel de aversión al riesgo (mayor = más conservador):", 0.0, 10.0, 2.0, 0.1)
    
    # Botón para calcular
    calcular = st.button("Calcular Cartera Óptima", type="primary")

# Funciones para optimización de carteras
def calcular_rendimientos(precios):
    """Calcula los rendimientos logarítmicos diarios."""
    return np.log(precios / precios.shift(1)).dropna()

def calcular_estadisticas_cartera(pesos, rendimientos):
    """Calcula el rendimiento y riesgo esperado de una cartera."""
    rendimiento_cartera = np.sum(rendimientos.mean() * pesos) * 252  # Anualizado
    volatilidad_cartera = np.sqrt(np.dot(pesos.T, np.dot(rendimientos.cov() * 252, pesos)))
    return rendimiento_cartera, volatilidad_cartera

def max_sharpe_ratio(rendimientos):
    """Encuentra la cartera con el máximo Ratio de Sharpe."""
    num_activos = len(rendimientos.columns)
    args = (rendimientos)
    restricciones = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    limite = tuple((0, 1) for _ in range(num_activos))
    
    def neg_sharpe(pesos, rendimientos):
        r, v = calcular_estadisticas_cartera(pesos, rendimientos)
        return -r/v
    
    pesos_iniciales = np.array([1/num_activos] * num_activos)
    resultado = minimize(neg_sharpe, pesos_iniciales, args=args, method='SLSQP', bounds=limite, constraints=restricciones)
    return resultado['x']

def min_varianza(rendimientos):
    """Encuentra la cartera de mínima varianza."""
    num_activos = len(rendimientos.columns)
    args = (rendimientos)
    restricciones = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    limite = tuple((0, 1) for _ in range(num_activos))
    
    def varianza(pesos, rendimientos):
        return calcular_estadisticas_cartera(pesos, rendimientos)[1]**2
    
    pesos_iniciales = np.array([1/num_activos] * num_activos)
    resultado = minimize(varianza, pesos_iniciales, args=args, method='SLSQP', bounds=limite, constraints=restricciones)
    return resultado['x']

def cartera_optima(rendimientos, aversion_riesgo):
    """Encuentra la cartera óptima según la aversión al riesgo del inversor."""
    num_activos = len(rendimientos.columns)
    args = (rendimientos, aversion_riesgo)
    restricciones = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    limite = tuple((0, 1) for _ in range(num_activos))
    
    def utilidad(pesos, rendimientos, aversion_riesgo):
        r, v = calcular_estadisticas_cartera(pesos, rendimientos)
        return -(r - aversion_riesgo * v**2)
    
    pesos_iniciales = np.array([1/num_activos] * num_activos)
    resultado = minimize(utilidad, pesos_iniciales, args=args, method='SLSQP', bounds=limite, constraints=restricciones)
    return resultado['x']

def generar_frontera_eficiente(rendimientos, num_carteras=100):
    """Genera puntos a lo largo de la frontera eficiente."""
    rendimientos_carteras = []
    volatilidades_carteras = []
    pesos_todas_carteras = []
    
    num_activos = len(rendimientos.columns)
    
    # Cartera de min varianza y max sharpe para definir los límites
    pesos_min_var = min_varianza(rendimientos)
    ret_min_var, vol_min_var = calcular_estadisticas_cartera(pesos_min_var, rendimientos)
    
    pesos_max_sharpe = max_sharpe_ratio(rendimientos)
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
        limite = tuple((0, 1) for _ in range(num_activos))
        
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

# Nueva función para mostrar gráficos de acciones individuales
def mostrar_graficos_acciones(data_completa, tickers):
    """Muestra gráficos individuales para cada acción seleccionada."""
    st.header("Gráficos de Acciones Individuales")
    
    # Determinar el número de columnas para los gráficos (2 gráficos por fila)
    num_columnas = 2
    num_filas = (len(tickers) + num_columnas - 1) // num_columnas
    
    # Crear gráficos para cada acción
    for i in range(num_filas):
        cols = st.columns(num_columnas)
        for j in range(num_columnas):
            idx = i * num_columnas + j
            if idx < len(tickers):
                ticker = tickers[idx]
                with cols[j]:
                    st.subheader(f"Gráfico de {ticker}")
                    
                    # Preparar los datos para este ticker
                    if isinstance(data_completa['Close'], pd.DataFrame):
                        # Si hay múltiples acciones, seleccionar la columna específica
                        datos_accion = data_completa['Close'][ticker]
                        datos_volumen = data_completa['Volume'][ticker] if 'Volume' in data_completa else None
                    else:
                        # Si solo hay una acción, usar la serie directamente
                        datos_accion = data_completa['Close']
                        datos_volumen = data_completa['Volume'] if 'Volume' in data_completa else None
                    
                    # Calcular media móvil de 20 y 50 días
                    ma_20 = datos_accion.rolling(window=20).mean()
                    ma_50 = datos_accion.rolling(window=50).mean()
                    
                    # Crear figura con dos paneles: precio y volumen
                    fig, ax = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
                    
                    # Panel superior para precios
                    ax[0].plot(datos_accion.index, datos_accion, label=f'{ticker} Precio')
                    ax[0].plot(ma_20.index, ma_20, label='MA 20 días', linestyle='--', alpha=0.7)
                    ax[0].plot(ma_50.index, ma_50, label='MA 50 días', linestyle='-.', alpha=0.7)
                    ax[0].set_title(f'Precio histórico de {ticker}')
                    ax[0].set_ylabel('Precio')
                    ax[0].legend(loc='best')
                    ax[0].grid(True, linestyle='--', alpha=0.6)
                    
                    # Panel inferior para volumen si está disponible
                    if datos_volumen is not None:
                        ax[1].bar(datos_volumen.index, datos_volumen, color='gray', alpha=0.6, label='Volumen')
                        ax[1].set_ylabel('Volumen')
                        ax[1].set_xlabel('Fecha')
                        ax[1].grid(True, linestyle='--', alpha=0.6)
                    else:
                        ax[1].set_visible(False)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Mostrar estadísticas de la acción
                    rendimiento_total = (datos_accion.iloc[-1] / datos_accion.iloc[0] - 1) * 100
                    volatilidad_anual = datos_accion.pct_change().std() * np.sqrt(252) * 100
                    
                    estadisticas = pd.DataFrame({
                        'Métrica': ['Precio inicial', 'Precio final', 'Rendimiento total', 'Volatilidad anual'],
                        'Valor': [
                            f'{datos_accion.iloc[0]:.2f}',
                            f'{datos_accion.iloc[-1]:.2f}',
                            f'{rendimiento_total:.2f}%',
                            f'{volatilidad_anual:.2f}%'
                        ]
                    })
                    
                    st.dataframe(estadisticas, hide_index=True)

# Función principal para mostrar resultados
def mostrar_resultados(data, aversion_riesgo, data_completa=None):
    if data is None:
        st.error("No hay datos disponibles. Por favor, carga un archivo o genera datos simulados.")
        return
    
    # Calcular rendimientos
    rendimientos = calcular_rendimientos(data)
    
    # Mostrar gráficos de acciones individuales si hay datos completos disponibles
    if data_completa is not None and opcion_datos == "Yahoo Finance (yfinance)":
        mostrar_graficos_acciones(data_completa, acciones_seleccionadas)
    
    # Calcular carteras óptimas
    with st.spinner('Calculando carteras óptimas...'):
        pesos_min_var = min_varianza(rendimientos)
        ret_min_var, vol_min_var = calcular_estadisticas_cartera(pesos_min_var, rendimientos)
        
        pesos_max_sharpe = max_sharpe_ratio(rendimientos)
        ret_max_sharpe, vol_max_sharpe = calcular_estadisticas_cartera(pesos_max_sharpe, rendimientos)
        
        pesos_optimos = cartera_optima(rendimientos, aversion_riesgo)
        ret_optimo, vol_optimo = calcular_estadisticas_cartera(pesos_optimos, rendimientos)
        
        # Generar frontera eficiente
        volatilidades, rendimientos_ef, pesos_ef = generar_frontera_eficiente(rendimientos)
    
    # Visualización principal
    st.header("Resultados de la Optimización")
    
    # Mostrar asignación de activos en la cartera óptima
    col1, col2 = st.columns([3, 2])
    
    with col1:
        # Gráfico de la frontera eficiente
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Graficar frontera eficiente
        ax.plot(volatilidades, rendimientos_ef, 'b-', linewidth=3, label='Frontera Eficiente')
        
        # Graficar activos individuales
        for i, accion in enumerate(rendimientos.columns):
            ret_anual = rendimientos.mean()[i] * 252
            vol_anual = rendimientos.std()[i] * np.sqrt(252)
            ax.scatter(vol_anual, ret_anual, s=100, marker='o', label=accion)
        
        # Marcar las carteras clave
        ax.scatter(vol_min_var, ret_min_var, s=200, color='g', marker='*', label='Mínima Varianza')
        ax.scatter(vol_max_sharpe, ret_max_sharpe, s=200, color='r', marker='*', label='Máximo Sharpe')
        ax.scatter(vol_optimo, ret_optimo, s=200, color='orange', marker='*', label='Tu Cartera Óptima')
        
        # Configuración del gráfico
        ax.set_xlabel('Volatilidad Anual (%)')
        ax.set_ylabel('Rendimiento Esperado Anual (%)')
        ax.set_title('Frontera Eficiente y Carteras Óptimas')
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Convertir a porcentajes
        ax.set_xticklabels([f'{x:.1f}%' for x in ax.get_xticks() * 100])
        ax.set_yticklabels([f'{y:.1f}%' for y in ax.get_yticks() * 100])
        
        # Añadir leyenda
        ax.legend(loc='best')
        
        st.pyplot(fig)
    
    with col2:
        st.subheader("Tu Cartera Óptima")
        
        # Mostrar pesos de la cartera óptima
        pesos_df = pd.DataFrame({
            'Activo': rendimientos.columns,
            'Asignación (%)': [peso * 100 for peso in pesos_optimos]
        }).sort_values(by='Asignación (%)', ascending=False)
        
        st.dataframe(pesos_df)
        
        # Mostrar métricas clave
        st.metric("Rendimiento Esperado Anual", f"{ret_optimo * 100:.2f}%")
        st.metric("Volatilidad Anual", f"{vol_optimo * 100:.2f}%")
        st.metric("Ratio de Sharpe", f"{ret_optimo/vol_optimo:.2f}")
        
        # Gráfico de pie para asignación de activos
        fig2, ax2 = plt.subplots(figsize=(8, 8))
        
        # Filtrar activos con peso muy pequeño para la visualización
        umbral = 0.02  # 2%
        pesos_filtrados = pesos_optimos.copy()
        activos_filtrados = list(rendimientos.columns)
        
        otros = 0
        for i, peso in enumerate(pesos_optimos):
            if peso < umbral:
                otros += peso
                pesos_filtrados[i] = 0
        
        # Crear nuevo array de pesos y etiquetas
        etiquetas = []
        pesos_grafico = []
        
        for i, peso in enumerate(pesos_filtrados):
            if peso > 0:
                etiquetas.append(activos_filtrados[i])
                pesos_grafico.append(peso)
        
        if otros > 0:
            etiquetas.append('Otros')
            pesos_grafico.append(otros)
        
        # Gráfico de pastel
        ax2.pie(pesos_grafico, labels=etiquetas, autopct='%1.1f%%', shadow=False, startangle=90)
        ax2.axis('equal')  # Para que el gráfico sea un círculo
        ax2.set_title('Asignación de Activos en la Cartera Óptima')
        
        st.pyplot(fig2)

# Ejecutar el cálculo cuando se presiona el botón
if calcular and data is not None:
    if opcion_datos == "Yahoo Finance (yfinance)":
        mostrar_resultados(data, aversion_riesgo, data_completa)
    else:
        mostrar_resultados(data, aversion_riesgo)
else:
    st.info("Configura los parámetros en el panel lateral y haz clic en 'Calcular Cartera Óptima' para ver los resultados.")
