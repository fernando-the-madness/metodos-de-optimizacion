import streamlit as st
import numpy as np
import tensorflow as tf
from collections import deque
import random

# Título de la aplicación
st.title("Adaptive Network Slicing in 5G using Deep Reinforcement Learning")

# --- Configuración del entorno ---
class NetworkEnvironment:
    def __init__(self, num_fog_nodes, resource_capacity, max_holding_time):
        self.num_fog_nodes = num_fog_nodes
        self.resource_capacity = resource_capacity
        self.max_holding_time = max_holding_time
        self.state = np.zeros((num_fog_nodes, 2))  # [recursos ocupados, carga de trabajo]
        self.task_queue = deque()

    def reset(self):
        self.state = np.zeros((self.num_fog_nodes, 2))
        self.task_queue.clear()
        return self.state.flatten()

    def step(self, action, task):
        # task = (utility, required_resources, holding_time)
        utility, required_resources, holding_time = task
        reward = 0
        done = False

        if action < self.num_fog_nodes:  # Asignar a un nodo de borde
            if self.state[action, 0] + required_resources <= self.resource_capacity:
                self.state[action, 0] += required_resources
                self.state[action, 1] += holding_time
                reward = 24 if utility >= 8 else -3  # Recompensa basada en la utilidad
            else:
                reward = -12  # Penalización por rechazo
        else:  # Rechazar y enviar a la nube
            reward = -12 if utility >= 8 else 3  # Recompensa basada en la utilidad

        # Liberar recursos después del tiempo de retención
        for i in range(self.num_fog_nodes):
            if self.state[i, 1] > 0:
                self.state[i, 1] -= 1
                if self.state[i, 1] == 0:
                    self.state[i, 0] = 0

        # Verificar si se alcanza el límite de utilización
        if np.sum(self.state[:, 0]) >= self.resource_capacity * self.num_fog_nodes * 0.8:
            done = True

        return self.state.flatten(), reward, done

# --- Modelo DQN ---
class DQN(tf.keras.Model):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.dense2 = tf.keras.layers.Dense(64, activation='relu')
        self.output_layer = tf.keras.layers.Dense(action_size)

    def call(self, state):
        x = self.dense1(state)
        x = self.dense2(x)
        return self.output_layer(x)

# --- Algoritmo DQN ---
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # Factor de descuento
        self.epsilon = 1.0  # Tasa de exploración
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = DQN(state_size, action_size)
        self.target_model = DQN(state_size, action_size)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        q_values = self.model.predict(np.array([state]))
        return np.argmax(q_values[0])

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        states = np.array([i[0] for i in minibatch])
        actions = np.array([i[1] for i in minibatch])
        rewards = np.array([i[2] for i in minibatch])
        next_states = np.array([i[3] for i in minibatch])
        dones = np.array([i[4] for i in minibatch])

        targets = rewards + self.gamma * np.amax(self.target_model.predict(next_states), axis=1) * (1 - dones)
        target_q_values = self.model.predict(states)
        target_q_values[np.arange(batch_size), actions] = targets

        self.model.fit(states, target_q_values, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

# --- Interfaz de Streamlit ---
num_fog_nodes = st.sidebar.slider("Número de nodos de borde (Fog Nodes)", 1, 10, 5)
resource_capacity = st.sidebar.slider("Capacidad de recursos por nodo", 5, 20, 10)
max_holding_time = st.sidebar.slider("Tiempo máximo de retención", 1, 10, 5)
num_episodes = st.sidebar.slider("Número de episodios de entrenamiento", 1, 1000, 100)
batch_size = st.sidebar.slider("Tamaño del batch", 1, 64, 32)

# Inicializar el entorno y el agente DQN
env = NetworkEnvironment(num_fog_nodes, resource_capacity, max_holding_time)
state_size = num_fog_nodes * 2
action_size = num_fog_nodes + 1  # Acciones: asignar a un nodo o rechazar
agent = DQNAgent(state_size, action_size)

# Entrenamiento del modelo
if st.sidebar.button("Entrenar modelo"):
    st.write("Entrenando el modelo DQN...")
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        while not done:
            # Generar una tarea aleatoria
            utility = np.random.randint(1, 11)
            required_resources = np.random.randint(1, 5)
            holding_time = np.random.randint(1, max_holding_time + 1)
            task = (utility, required_resources, holding_time)

            # Tomar una acción
            action = agent.act(state)
            next_state, reward, done = env.step(action, task)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            if len(agent.memory) > batch_size:
                agent.replay(batch_size)

        agent.update_target_model()
        st.write(f"Episodio {episode + 1}, Recompensa total: {total_reward}")

    st.success("¡Entrenamiento completado!")

# Simulación en tiempo real
if st.button("Simular asignación de tareas"):
    st.write("Simulando la llegada de tareas...")
    state = env.reset()
    for _ in range(10):  # Simular 10 tareas
        utility = np.random.randint(1, 11)
        required_resources = np.random.randint(1, 5)
        holding_time = np.random.randint(1, max_holding_time + 1)
        task = (utility, required_resources, holding_time)

        action = agent.act(state)
        next_state, reward, done = env.step(action, task)
        state = next_state

        st.write(f"Tarea: Utilidad={utility}, Recursos={required_resources}, Tiempo={holding_time}")
        st.write(f"Acción: {'Asignar a nodo ' + str(action) if action < num_fog_nodes else 'Rechazar y enviar a la nube'}")
        st.write(f"Recompensa: {reward}")
        st.bar_chart(state.reshape(num_fog_nodes, 2)[:, 0])  # Mostrar recursos ocupados