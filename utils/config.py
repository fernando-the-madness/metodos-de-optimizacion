# utils/config.py
class Config:
    # Parámetros de la red
    NUM_FOG_NODES = 7
    RESOURCE_CAPACITY = 7  # Capacidad de recursos por Fog Node
    STATE_SIZE = 18  # Dimensión del estado (ajustar según el paper)
    ACTION_SIZE = 8  # Número de acciones (servir en FN1, FN2, ..., rechazar)

    # Parámetros del DQN
    LEARNING_RATE = 0.001
    GAMMA = 0.95  # Factor de descuento
    EPSILON = 1.0  # Exploración inicial
    EPSILON_MIN = 0.01
    EPSILON_DECAY = 0.995
    BATCH_SIZE = 32
    MEMORY_SIZE = 2000  # Tamaño de la memoria de repetición