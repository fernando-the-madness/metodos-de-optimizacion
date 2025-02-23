import streamlit as st
import numpy as np
from utils.environment import FogRANEnvironment
from utils.dqn_agent import DQNAgent

# Configuración de la aplicación
st.title("Fog-RAN Network Slicing con DRL")
st.write("Esta aplicación simula la asignación de recursos en una red Fog-RAN utilizando Deep Reinforcement Learning.")

# Inicialización del entorno y el agente
env = FogRANEnvironment()
state_size = len(env.reset())
action_size = env.num_fog_nodes + 1  # Acciones: fog nodes + rechazar
agent = DQNAgent(state_size, action_size)

# Entrenamiento del agente
if st.button("Entrenar Agente"):
    progress_bar = st.progress(0)
    for episode in range(100):  # 100 episodios de entrenamiento
        state = env.reset()
        total_reward = 0
        done = False
        while not done:
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
        agent.replay(32)  # Entrenar con un batch de 32 muestras
        progress_bar.progress((episode + 1) / 100)
    st.success("Entrenamiento completado!")

# Simulación en tiempo real
if st.button("Simular"):
    state = env.reset()
    st.write("Estado inicial:", state)
    for _ in range(10):  # Simular 10 pasos
        action = agent.act(state)
        next_state, reward, done = env.step(action)
        st.write(f"Acción: {action}, Recompensa: {reward}, Estado siguiente: {next_state}")
        state = next_state