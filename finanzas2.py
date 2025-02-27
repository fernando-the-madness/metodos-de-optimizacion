import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from scipy.optimize import minimize
import datetime as dt
import plotly.graph_objects as go
import plotly.express as px
from sklearn.covariance import LedoitWolf
import base64
from io import BytesIO

# Configuración de la página
st.set_page_config(
    page_title="Optimizador de Carteras de Inversión",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos personalizados
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #424242;
    }
    .card {
        background-color: #f9f9f9;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .metric-card {
        background-color: #EEF5FF;
        border-left: 5px solid #1E88E5;
        padding: 10px;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Título de la aplicación
st.markdown("<h1 class='main-header'>Optimizador de Carteras de Inversión</h1>", unsafe_allow_html=True)
st.markdown("""
Esta aplicación implementa modelos avanzados para optimización de carteras de inversión, incluido el clásico modelo de Markowitz.
Selecciona acciones, un periodo de tiempo y tus preferencias para encontrar tu cartera óptima personalizada.
""")

# Sidebar para entrada de datos
with st.sidebar:
    st.markdown("<h2 class='sub-header'>Parámetros de la Cartera</h2>", unsafe_allow_html=True)
    
    # Listas predefinidas de acciones populares
    acciones_populares = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'TSLA', 'JPM', 'V', 'PG', 'JNJ', 'KO', 'DIS']
    
    # Entrada de acciones
    opcion_entrada = st.radio("Selecciona método de entrada de acciones:", ["Usar acciones predefinidas", "Ingresar tus propias acciones"])
    
    if opcion_entrada == "Usar acciones predefinidas":
        acciones_seleccionadas = st.multiselect("Selecciona acciones:", acciones_populares, default=acciones_populares[:5])
    else:
        acciones_input = st.text_input("Ingresa símbolos de acciones separados por comas (ej: AAPL,MSFT,AMZN):")
        acciones_seleccionadas = [accion.strip() for accion in acciones_input.split(',')] if acciones_input else []
    
    # Verificar que hay suficientes acciones seleccionadas
    if len(acciones_seleccionadas) < 2:
        st.warning("Por favor selecciona al menos 2 acciones para construir una cartera.")
    
    # Periodo de tiempo
    st.markdown("<h3 class='sub-header'>Periodo de Tiempo</h3>", unsafe_allow_html=True)
    fecha_inicio = st.date_input("Fecha de inicio:", dt.date(2020, 1, 1))
    fecha_fin = st.date_input("Fecha de fin:", dt.date.today())
    
    # Parámetros de optimización
    st.markdown("<h3 class='sub-header'>Preferencia de Riesgo-Rendimiento</h3>", unsafe_allow_html=True)
    aversion_riesgo = st.slider("Nivel de aversión al riesgo (mayor = más conservador):", 0.0, 10.0, 2.0, 0.1)
    
    # Botón para calcular
    calcular = st.button("Calcular Cartera Óptima", type="primary", use_container_width=True)

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

# Función principal para mostrar resultados
def mostrar_resultados(acciones, fecha_inicio, fecha_fin, aversion_riesgo):
    # Obtener los datos históricos
    try:
        # Descargar precios históricos
        st.info("Descargando datos históricos de precios...")
        data = yf.download(acciones, start=fecha_inicio, end=fecha_fin)['Adj Close']
        
        # Manejar caso de una sola acción
        if isinstance(data, pd.Series):
            data = pd.DataFrame(data, columns=[acciones[0]])
        
        # Verificar si faltan datos
        if data.isnull().values.any():
            st.warning("Algunos datos están faltantes. Eliminando filas con valores NaN.")
            data = data.dropna()
        
        if len(data) < 30:
            st.error("No hay suficientes datos para realizar el análisis. Por favor, ajusta el periodo de tiempo o selecciona otras acciones.")
            return
        
        # Mostrar datos de precios
        with st.expander("Ver datos de precios", expanded=False):
            st.dataframe(data)
        
        # Calcular rendimientos
        rendimientos = calcular_rendimientos(data)
        
        # Mostrar estadísticas de los activos
        with st.expander("Ver estadísticas de rendimientos por activo", expanded=False):
            # Rendimientos anualizados
            rendimientos_anuales = rendimientos.mean() * 252
            volatilidad_anual = rendimientos.std() * np.sqrt(252)
            ratios_sharpe = rendimientos_anuales / volatilidad_anual
            
            estadisticas_df = pd.DataFrame({
                'Rendimiento Anual (%)': rendimientos_anuales * 100,
                'Volatilidad Anual (%)': volatilidad_anual * 100,
                'Ratio de Sharpe': ratios_sharpe
            })
            
            st.dataframe(estadisticas_df)
        
        # Calcular carteras óptimas
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
        
        # Comparación de carteras
        st.subheader("Comparación de Carteras Óptimas")
        
        col3, col4, col5 = st.columns(3)
        
        with col3:
            st.markdown("### Cartera de Mínima Varianza")
            min_var_df = pd.DataFrame({
                'Activo': rendimientos.columns,
                'Asignación (%)': [peso * 100 for peso in pesos_min_var]
            }).sort_values(by='Asignación (%)', ascending=False)
            
            st.dataframe(min_var_df)
            st.metric("Rendimiento Esperado Anual", f"{ret_min_var * 100:.2f}%")
            st.metric("Volatilidad Anual", f"{vol_min_var * 100:.2f}%")
            st.metric("Ratio de Sharpe", f"{ret_min_var/vol_min_var:.2f}")
        
        with col4:
            st.markdown("### Cartera de Máximo Sharpe")
            max_sharpe_df = pd.DataFrame({
                'Activo': rendimientos.columns,
                'Asignación (%)': [peso * 100 for peso in pesos_max_sharpe]
            }).sort_values(by='Asignación (%)', ascending=False)
            
            st.dataframe(max_sharpe_df)
            st.metric("Rendimiento Esperado Anual", f"{ret_max_sharpe * 100:.2f}%")
            st.metric("Volatilidad Anual", f"{vol_max_sharpe * 100:.2f}%")
            st.metric("Ratio de Sharpe", f"{ret_max_sharpe/vol_max_sharpe:.2f}")
        
        with col5:
            st.markdown("### Cartera Óptima Personalizada")
            optima_df = pd.DataFrame({
                'Activo': rendimientos.columns,
                'Asignación (%)': [peso * 100 for peso in pesos_optimos]
            }).sort_values(by='Asignación (%)', ascending=False)
            
            st.dataframe(optima_df)
            st.metric("Rendimiento Esperado Anual", f"{ret_optimo * 100:.2f}%")
            st.metric("Volatilidad Anual", f"{vol_optimo * 100:.2f}%")
            st.metric("Ratio de Sharpe", f"{ret_optimo/vol_optimo:.2f}")
        
        # Análisis de correlación
        st.subheader("Matriz de Correlación")
        
        fig3, ax3 = plt.subplots(figsize=(10, 8))
        correlacion = rendimientos.corr()
        im = ax3.imshow(correlacion, cmap='coolwarm')
        
        # Etiquetas
        ax3.set_xticks(np.arange(len(rendimientos.columns)))
        ax3.set_yticks(np.arange(len(rendimientos.columns)))
        ax3.set_xticklabels(rendimientos.columns)
        ax3.set_yticklabels(rendimientos.columns)
        
        # Rotación de etiquetas en x
        plt.setp(ax3.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Añadir valores numéricos
        for i in range(len(rendimientos.columns)):
            for j in range(len(rendimientos.columns)):
                ax3.text(j, i, f"{correlacion.iloc[i, j]:.2f}", ha="center", va="center", color="black" if abs(correlacion.iloc[i, j]) < 0.7 else "white")
        
        ax3.set_title("Matriz de Correlación entre Activos")
        fig3.colorbar(im)
        fig3.tight_layout()
        
        st.pyplot(fig3)
        
        # Rendimiento histórico
        st.subheader("Rendimiento Histórico de los Activos")
        
        # Rendimientos acumulados
        rendimientos_acum = (1 + rendimientos).cumprod()
        
        fig4, ax4 = plt.subplots(figsize=(12, 6))
        rendimientos_acum.plot(ax=ax4)
        ax4.set_title("Rendimiento Acumulado")
        ax4.set_ylabel("Valor (Normalizado a 1)")
        ax4.grid(True, linestyle='--', alpha=0.7)
        ax4.legend(loc='best')
        
        st.pyplot(fig4)
        
        # Descargar resultados
        st.subheader("Descargar Resultados")
        
        # Crear DataFrame con todos los resultados
        resultados = {
            "Cartera": ["Mínima Varianza", "Máximo Sharpe", "Óptima Personalizada"],
            "Rendimiento Esperado Anual (%)": [ret_min_var * 100, ret_max_sharpe * 100, ret_optimo * 100],
            "Volatilidad Anual (%)": [vol_min_var * 100, vol_max_sharpe * 100, vol_optimo * 100],
            "Ratio de Sharpe": [ret_min_var/vol_min_var, ret_max_sharpe/vol_max_sharpe, ret_optimo/vol_optimo]
        }
        
        for i, accion in enumerate(rendimientos.columns):
            resultados[f"{accion} (%)"] = [pesos_min_var[i] * 100, pesos_max_sharpe[i] * 100, pesos_optimos[i] * 100]
        
        resultados_df = pd.DataFrame(resultados)
        csv = resultados_df.to_csv(index=False)
        st.download_button(
            label="Descargar Resultados CSV",
            data=csv,
            file_name="resultados_optimizacion_cartera.csv",
            mime="text/csv"
        )
    
    except Exception as e:
        st.error(f"Error durante el análisis: {str(e)}")
        st.error("Por favor, verifica que los símbolos de las acciones sean válidos y que haya datos disponibles para el periodo seleccionado.")

# Ejecutar el cálculo cuando se presiona el botón
if calcular and len(acciones_seleccionadas) >= 2:
    mostrar_resultados(acciones_seleccionadas, fecha_inicio, fecha_fin, aversion_riesgo)
elif len(acciones_seleccionadas) < 2:
    st.info("Selecciona al menos 2 acciones en el panel lateral para comenzar.")
else:
    st.info("Configura los parámetros en el panel lateral y haz clic en 'Calcular Cartera Óptima' para ver los resultados.")

# Información educativa
with st.expander("Acerca del Modelo de Markowitz"):
    st.markdown("""
    ### Teoría Moderna de Carteras (Modelo de Markowitz)
    
    El modelo de Markowitz, desarrollado por Harry Markowitz en 1952, es el fundamento de la Teoría Moderna de Carteras (MPT). Esta teoría propone que:
    
    - Los inversores son adversos al riesgo y prefieren carteras con mayor rendimiento esperado y menor riesgo.
    - El riesgo de una cartera depende no solo de los riesgos individuales de los activos, sino también de sus correlaciones.
    - Para cada nivel de riesgo, existe una cartera que maximiza el rendimiento (carteras de la frontera eficiente).
    
    #### Conceptos clave:
    
    1. **Frontera Eficiente**: Conjunto de carteras que ofrecen el máximo rendimiento esperado para un nivel de riesgo dado.
    2. **Cartera de Mínima Varianza**: La cartera con menor volatilidad posible.
    3. **Cartera de Máximo Sharpe**: La cartera que maximiza la relación rendimiento/riesgo (asumiendo una tasa libre de riesgo de cero).
    4. **Diversificación**: Estrategia para reducir el riesgo mediante la inversión en diferentes activos con correlaciones bajas o negativas.
    
    #### Limitaciones:
    
    - Asume que los rendimientos siguen una distribución normal
    - No considera costos de transacción, impuestos o liquidez
    - Se basa en datos históricos que pueden no predecir el futuro
    
    Esta aplicación implementa el modelo de Markowitz para ayudarte a encontrar carteras óptimas según tu nivel de aversión al riesgo.
    """)

# Footer
st.markdown("---")
st.markdown("Aplicación creada con Streamlit y Python. Implementa el modelo de Markowitz para optimización de carteras de inversión.")
st.markdown("Nota: Esta aplicación es solo para fines educativos. No constituye asesoramiento financiero o de inversión.")