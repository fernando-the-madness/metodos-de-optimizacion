# dqn/dqn_model.py
import tensorflow as tf
from utils.config import Config

class DQNModel:
    def __init__(self, state_size, action_size):
        self.model = self.build_model(state_size, action_size)

    def build_model(self, state_size, action_size):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, input_dim=state_size, activation='relu'),
            tf.keras.layers.Dense(24, activation='relu'),
            tf.keras.layers.Dense(action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=Config.LEARNING_RATE))
        return model