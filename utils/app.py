# utils/app.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from env.network_env import NetworkEnv
from env.fog_node import FogNode
from dgn.dgn_agent import DQNAgent
from utils.config import Config
from utils.visualization import plot_go_scores, plot_resource_utilization

# Título de la aplicación
st.title("Network Slicing en 5G con Deep Reinforcement Learning")

# Descripción
st.write("""
Esta aplicación simula la asignación dinámica de recursos en una red 5G utilizando Deep Q-Learning (DQN).
""")

# Parámetros configurables
st.sidebar.header("Configuración")
num_fog_nodes = st.sidebar.slider("Número de Fog Nodes", 1, 10, 7)
resource_capacity = st.sidebar.slider("Capacidad de recursos por Fog Node", 1, 10, 7)
num_episodes = st.sidebar.slider("Número de episodios de entrenamiento", 1, 1000, 100)

# Crear Fog Nodes
fog_nodes = [FogNode(i, resource_capacity) for i in range(num_fog_nodes)]
env = NetworkEnv(fog_nodes)

# Crear el agente DQN
state_size = Config.STATE_SIZE
action_size = num_fog_nodes + 1  # Acciones: servir en FN1, FN2, ..., rechazar
agent = DQNAgent(state_size, action_size)

# Botón para iniciar la simulación
if st.sidebar.button("Iniciar Simulación"):
    st.write("Entrenando el modelo DQN...")

    # Listas para almacenar métricas
    go_scores = []
    resource_utilization = []

    # Entrenar el agente
    for episode in range(num_episodes):
        state = env.reset()
        state = np.reshape(state, [1, state_size])

        for time in range(500):
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state

            if done:
                break

        agent.replay(batch_size=Config.BATCH_SIZE)

        # Calcular métricas (GoS y utilización de recursos)
        go_scores.append(env.calculate_go_scores())
        resource_utilization.append(env.calculate_resource_utilization())

    st.write("Simulación completada.")

    # Mostrar resultados
    st.subheader("Resultados")

    # Gráfico de GoS
    st.write("### Grade of Service (GoS)")
    fig, ax = plt.subplots()
    ax.plot(go_scores, label="GoS")
    ax.set_xlabel("Episodio")
    ax.set_ylabel("GoS")
    ax.legend()
    st.pyplot(fig)

    # Gráfico de utilización de recursos
    st.write("### Utilización de Recursos")
    fig, ax = plt.subplots()
    ax.plot(resource_utilization, label="Utilización de Recursos")
    ax.set_xlabel("Episodio")
    ax.set_ylabel("Utilización (%)")
    ax.legend()
    st.pyplot(fig)