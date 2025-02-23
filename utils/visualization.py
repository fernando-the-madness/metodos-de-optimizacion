# utils/visualization.py
import matplotlib.pyplot as plt

def plot_go_scores(go_scores):
    """
    Grafica el Grade of Service (GoS) a lo largo del tiempo.
    
    Parámetros:
        go_scores (list): Lista de valores de GoS para cada episodio.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(go_scores, label="GoS", color="blue")
    plt.xlabel("Episodio")
    plt.ylabel("GoS")
    plt.title("Grade of Service (GoS)")
    plt.legend()
    plt.grid()

def plot_resource_utilization(resource_utilization):
    """
    Grafica la utilización de recursos a lo largo del tiempo.
    
    Parámetros:
        resource_utilization (list): Lista de valores de utilización de recursos para cada episodio.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(resource_utilization, label="Utilización de Recursos", color="green")
    plt.xlabel("Episodio")
    plt.ylabel("Utilización (%)")
    plt.title("Utilización de Recursos")
    plt.legend()
    plt.grid()
