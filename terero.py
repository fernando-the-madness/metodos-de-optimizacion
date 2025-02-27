import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Función para obtener datos de precios ajustados
@st.cache_data
def obtener_precios(tickers, start, end):
    datos = yf.download(tickers, start=start, end=end)['Adj Close']
    return datos.dropna()

# Función para calcular métricas de la cartera
def calcular_metricas(pesos, retornos, covarianza):
    rendimiento = np.sum(retornos * pesos) * 252
    riesgo = np.sqrt(np.dot(pesos.T, np.dot(covarianza, pesos))) * np.sqrt(252)
    return rendimiento, riesgo

# Función para optimizar la cartera usando Markowitz
def optimizar_cartera(retornos, covarianza):
    num_activos = len(retornos)
    args = (retornos, covarianza)
    restricciones = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    limites = tuple((0, 1) for _ in range(num_activos))
    funcion_objetivo = lambda pesos, retornos, cov: -calcular_metricas(pesos, retornos, cov)[0] / calcular_metricas(pesos, retornos, cov)[1]
    resultado = minimize(funcion_objetivo, num_activos * [1. / num_activos,], args=args, method='SLSQP', bounds=limites, constraints=restricciones)
    return resultado.x

# Configuración de Streamlit
st.title('Optimización de Carteras - Modelo de Markowitz')
st.sidebar.header('Parámetros')

# Selección de parámetros por el usuario
tickers = st.sidebar.text_input('Tickers (separados por espacio)', 'AAPL MSFT GOOGL AMZN TSLA').split()
fecha_inicio = st.sidebar.date_input('Fecha de inicio', pd.to_datetime('2020-01-01'))
fecha_fin = st.sidebar.date_input('Fecha de fin', pd.to_datetime('2023-01-01'))

if st.sidebar.button('Optimizar cartera'):
    precios = obtener_precios(tickers, fecha_inicio, fecha_fin)
    retornos_diarios = precios.pct_change().dropna()
    retornos_esperados = retornos_diarios.mean()
    matriz_covarianza = retornos_diarios.cov()
    pesos_optimos = optimizar_cartera(retornos_esperados, matriz_covarianza)
    rendimiento_opt, riesgo_opt = calcular_metricas(pesos_optimos, retornos_esperados, matriz_covarianza)
    st.subheader('Resultados de la Optimización')
    st.write(f'Rendimiento Esperado: {rendimiento_opt:.2%}')
    st.write(f'Riesgo (Desviación Estándar): {riesgo_opt:.2%}')
    st.write('Pesos Óptimos:')
    for ticker, peso in zip(tickers, pesos_optimos):
        st.write(f'{ticker}: {peso:.2%}')

    # Visualización de la composición de la cartera
    fig, ax = plt.subplots()
    ax.pie(pesos_optimos, labels=tickers, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')
    st.pyplot(fig)
