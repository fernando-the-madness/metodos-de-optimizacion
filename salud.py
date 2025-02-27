import streamlit as st
from ortools.sat.python import cp_model
import pulp

# Título de la aplicación
st.title("Optimización de Programación de Citas en Hospitales")

# Entradas del usuario
st.sidebar.header("Parámetros de Entrada")
num_pacientes = st.sidebar.number_input("Número de Pacientes", min_value=1, value=10)
num_doctores = st.sidebar.number_input("Número de Doctores", min_value=1, value=3)
duracion_cita = st.sidebar.number_input("Duración de la Cita (minutos)", min_value=1, value=30)
horario_inicio = st.sidebar.number_input("Hora de Inicio (horas)", min_value=0, value=8)
horario_fin = st.sidebar.number_input("Hora de Fin (horas)", min_value=1, value=17)

# Convertir horas a minutos
horario_inicio_min = horario_inicio * 60
horario_fin_min = horario_fin * 60

# Crear el modelo de optimización
model = cp_model.CpModel()

# Variables de decisión
# x[i][j] = 1 si el paciente i es asignado al doctor j, 0 en caso contrario
x = {}
for i in range(num_pacientes):
    for j in range(num_doctores):
        x[i, j] = model.NewBoolVar(f'x_{i}_{j}')

# Restricciones
# Cada paciente debe ser asignado a un solo doctor
for i in range(num_pacientes):
    model.Add(sum(x[i, j] for j in range(num_doctores)) == 1)

# Cada doctor no puede tener más de un paciente al mismo tiempo
for j in range(num_doctores):
    for i1 in range(num_pacientes):
        for i2 in range(i1 + 1, num_pacientes):
            model.Add(x[i1, j] + x[i2, j] <= 1)

# Función objetivo: Minimizar el tiempo total de espera
model.Minimize(sum(x[i, j] * (i * duracion_cita) for i in range(num_pacientes) for j in range(num_doctores)))

# Resolver el modelo
solver = cp_model.CpSolver()
status = solver.Solve(model)

# Mostrar resultados
st.header("Resultados de la Programación de Citas")
if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
    st.write("Asignación óptima encontrada:")
    for j in range(num_doctores):
        st.write(f"Doctor {j+1}:")
        for i in range(num_pacientes):
            if solver.Value(x[i, j]) == 1:
                st.write(f"  Paciente {i+1} a las {horario_inicio + (i * duracion_cita) // 60}:{(i * duracion_cita) % 60:02d}")
else:
    st.write("No se encontró una solución óptima.")