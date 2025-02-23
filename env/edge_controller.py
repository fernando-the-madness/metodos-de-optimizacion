# env/edge_controller.py
from env.fog_node import FogNode

class EdgeController:
    def __init__(self, fog_nodes):
        self.fog_nodes = fog_nodes  # Lista de Fog Nodes

    def decide_task_allocation(self, task):
        # Aquí implementarás la lógica de DQN para decidir dónde asignar la tarea
        pass