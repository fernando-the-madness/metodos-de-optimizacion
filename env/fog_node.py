# env/fog_node.py
class FogNode:
    def __init__(self, node_id, resource_capacity):
        self.node_id = node_id
        self.resource_capacity = resource_capacity
        self.occupied_resources = 0  # Recursos ocupados actualmente

    def allocate_resources(self, resources):
        if self.occupied_resources + resources <= self.resource_capacity:
            self.occupied_resources += resources
            return True
        return False  # No hay suficientes recursos

    def release_resources(self, resources):
        self.occupied_resources -= resources
