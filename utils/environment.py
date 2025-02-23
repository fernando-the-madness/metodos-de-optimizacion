import numpy as np

class FogRANEnvironment:
    def __init__(self, num_fog_nodes=7, max_resources=7):
        self.num_fog_nodes = num_fog_nodes
        self.max_resources = max_resources
        self.reset()

    def reset(self):
        # Inicializa el estado de los fog nodes
        self.fog_nodes = [{"resources_used": 0, "tasks": []} for _ in range(self.num_fog_nodes)]
        self.current_task = self._generate_task()
        return self._get_state()

    def _generate_task(self):
        # Genera una tarea con utilidad, recursos y tiempo de retención
        utility = np.random.randint(1, 11)  # Utilidad entre 1 y 10
        resources = np.random.randint(1, 5)  # Recursos entre 1 y 4
        holding_time = np.random.choice([5, 10, 15, 20, 25, 30])  # Tiempo de retención
        return {"utility": utility, "resources": resources, "holding_time": holding_time}

    def _get_state(self):
        # Devuelve el estado actual del sistema
        state = []
        for node in self.fog_nodes:
            state.append(node["resources_used"])
            state.append(len(node["tasks"]))
        state.extend([self.current_task["utility"], self.current_task["resources"], self.current_task["holding_time"]])
        return np.array(state)

    def step(self, action):
        # Ejecuta una acción (asignar tarea a un fog node o rechazarla)
        reward = 0
        if action < self.num_fog_nodes:  # Asignar a un fog node
            node = self.fog_nodes[action]
            if node["resources_used"] + self.current_task["resources"] <= self.max_resources:
                node["resources_used"] += self.current_task["resources"]
                node["tasks"].append(self.current_task)
                reward = self._calculate_reward(action)
            else:
                reward = -10  # Penalización por sobrecarga
        else:  # Rechazar la tarea
            reward = -5 if self.current_task["utility"] >= 8 else -1  # Penalización por rechazo

        self.current_task = self._generate_task()
        next_state = self._get_state()
        done = False  # El entorno no termina
        return next_state, reward, done

    def _calculate_reward(self, action):
        # Calcula la recompensa basada en la utilidad y la carga de la tarea
        utility = self.current_task["utility"]
        load = self.current_task["resources"] * self.current_task["holding_time"]
        if utility >= 8:  # Tarea de alta prioridad
            return 24 - load  # Recompensa alta
        else:
            return 3 - load  # Recompensa baja