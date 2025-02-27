import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Generar datos simulados
np.random.seed(42)
assets = ['AAPL', 'GOOGL', 'AMZN', 'MSFT', 'TSLA']
dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
data = pd.DataFrame(np.random.randn(100, len(assets)) * 0.02 + 0.001, index=dates, columns=assets)
data = (1 + data).cumprod()  # Simula precios de activos

# Función para calcular métricas de la cartera
def portfolio_metrics(weights, returns, cov_matrix):
    port_return = np.sum(returns * weights)
    port_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return port_return, port_volatility

# Función para optimizar la cartera (Modelo de Markowitz)
def optimize_portfolio(returns, cov_matrix, risk_tolerance):
    num_assets = len(returns)
    init_guess = np.ones(num_assets) / num_assets
    bounds = [(0, 1) for _ in range(num_assets)]
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    
    def neg_sharpe_ratio(weights):
        port_return, port_volatility = portfolio_metrics(weights, returns, cov_matrix)
        return -port_return / port_volatility  # Maximizar el Sharpe Ratio
    
    result = minimize(neg_sharpe_ratio, init_guess, bounds=bounds, constraints=constraints)
    return result.x

# Interfaz con Streamlit
st.title('Optimización de Carteras con el Modelo de Markowitz')

# Mostrar datos simulados
st.write("Datos simulados de precios de activos:", data.head())

# Cálculo de retornos
returns = data.pct_change().dropna()
mean_returns = returns.mean()
cov_matrix = returns.cov()

# Selección del nivel de riesgo
risk_tolerance = st.slider("Nivel de aversión al riesgo", min_value=0.1, max_value=10.0, value=2.0)

# Optimización
optimal_weights = optimize_portfolio(mean_returns, cov_matrix, risk_tolerance)
optimal_allocation = pd.DataFrame({'Activo': data.columns, 'Pesos óptimos': optimal_weights})
st.write("Distribución óptima de activos:", optimal_allocation)

# Visualización de la frontera eficiente
st.subheader("Frontera Eficiente")
fig, ax = plt.subplots()
ax.scatter(np.sqrt(np.diag(cov_matrix)), mean_returns, c='blue', label='Activos individuales')
opt_ret, opt_vol = portfolio_metrics(optimal_weights, mean_returns, cov_matrix)
ax.scatter(opt_vol, opt_ret, c='red', marker='*', s=200, label='Cartera Óptima')
ax.set_xlabel("Volatilidad")
ax.set_ylabel("Retorno Esperado")
ax.legend()
st.pyplot(fig)
