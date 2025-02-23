# env/network_env.py
import numpy as np

class NetworkEnv:
    def __init__(self, fog_nodes):
        self.fog_nodes = fog_nodes
        self.high_priority_tasks_served = 0
        self.high_priority_tasks_received = 0

    def reset(self):
        # Reiniciar el estado de los Fog Nodes
        for node in self.fog_nodes:
            node.occupied_resources = 0
        self.high_priority_tasks_served = 0
        self.high_priority_tasks_received = 0
        return self._get_state()

    def step(self, action):
        # Generar una tarea
        task = self.generate_task()
        self.high_priority_tasks_received += 1 if task["priority"] >= 8 else 0

        # Asignar la tarea según la acción
        if action < len(self.fog_nodes):  # Asignar a un Fog Node
            if self.fog_nodes[action].allocate_resources(task["resources"]):
                self.high_priority_tasks_served += 1 if task["priority"] >= 8 else 0
                reward = 1 if task["priority"] >= 8 else 0
            else:
                reward = -1  # Penalización por no poder asignar
        else:  # Rechazar la tarea
            reward = -1

        next_state = self._get_state()
        done = False  # La simulación continúa
        return next_state, reward, done, {}

    def generate_task(self):
        # Generar una tarea con recursos y prioridad aleatorios
        resources = np.random.randint(1, 5)  # Recursos requeridos
        priority = np.random.randint(1, 11)  # Prioridad (1 = baja, 10 = alta)
        return {"resources": resources, "priority": priority}

    def _get_state(self):
        # Obtener el estado actual (recursos ocupados en cada FN)
        state = [node.occupied_resources for node in self.fog_nodes]
        return np.array(state)

    def calculate_go_scores(self):
        if self.high_priority_tasks_received == 0:
            return 0
        return self.high_priority_tasks_served / self.high_priority_tasks_received

    def calculate_resource_utilization(self):
        total_resources = sum(node.resource_capacity for node in self.fog_nodes)
        occupied_resources = sum(node.occupied_resources for node in self.fog_nodes)
        if total_resources == 0:
            return 0
        return (occupied_resources / total_resources) * 100
