# streamlit_app.py

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px

from models.dqn_model import DQNModel
from utils.data_loader import load_data
from utils.visualization import plot_metrics

st.set_page_config(page_title='Adaptive Network Slicing in 5G', layout='wide')
st.title('Adaptive Network Slicing in 5G for Intelligent Vehicular Systems')

# Sección 1: Introducción
st.header('Introducción')
st.markdown('''Este proyecto implementa un modelo de Deep Reinforcement Learning para la asignación adaptativa de recursos en 5G, basado en el paper "Deep Reinforcement Learning for Adaptive Network Slicing in 5G for Intelligent Vehicular Systems and Smart Cities". El objetivo es optimizar el Grade of Service (GoS), la utilización de recursos y minimizar el Cloud Avoidance.
''')

# Sección 2: Configuración del Modelo
st.header('Configuración del Modelo')
gamma = st.slider('Factor de Descuento (Gamma)', 0.0, 1.0, 0.99)
learning_rate = st.slider('Learning Rate', 0.0001, 0.01, 0.001)

dqn = DQNModel(gamma=gamma, learning_rate=learning_rate)

# Sección 3: Entrenamiento y Evaluación
st.header('Entrenamiento y Evaluación')
data = load_data()
rewards, metrics = dqn.train(data)
st.success('Entrenamiento completado!')

# Sección 4: Visualización de Resultados
st.header('Visualización de Resultados')
fig = plot_metrics(metrics)
st.plotly_chart(fig)

# Sección 5: Conclusiones
st.header('Conclusiones y Discusión')
st.markdown('''El modelo DRL mostró mejoras en el Grade of Service (GoS) y la utilización de recursos en comparación con métodos tradicionales.
''')
