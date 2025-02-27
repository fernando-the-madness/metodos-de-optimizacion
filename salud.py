import streamlit as st
from ortools.sat.python import cp_model

st.title('Optimizador de Citas en Hospital')
st.sidebar.header('Parámetros de Entrada')

# Entrada de datos
num_medicos = st.sidebar.number_input('Número de médicos', min_value=1, value=3)
duracion_cita = st.sidebar.number_input('Duración de cada cita (minutos)', min_value=10, value=30)
pacientes = st.sidebar.text_area('Lista de pacientes (uno por línea)').splitlines()

disponibilidad = []
for i in range(num_medicos):
    disponibilidad.append(st.sidebar.text_input(f'Disponibilidad Médico {i+1} (ej. 9-12, 14-17)'))

if st.button('Optimizar Citas'):
    # Procesamiento de disponibilidad
    horarios = []
    for disp in disponibilidad:
        rangos = disp.split(',')
        horas = []
        for r in rangos:
            inicio, fin = map(int, r.split('-'))
            horas.extend(list(range(inicio, fin)))
        horarios.append(horas)

    # Crear modelo
    model = cp_model.CpModel()
    asignaciones = {}
    for i, paciente in enumerate(pacientes):
        for j in range(num_medicos):
            for h in horarios[j]:
                asignaciones[(i, j, h)] = model.NewBoolVar(f'paciente_{i}_medico_{j}_hora_{h}')

    # Restricciones
    for i in range(len(pacientes)):
        model.Add(sum(asignaciones[(i, j, h)] for j in range(num_medicos) for h in horarios[j]) == 1)

    # Optimización
    model.Maximize(sum(asignaciones[(i, j, h)] for i in range(len(pacientes)) for j in range(num_medicos) for h in horarios[j]))

    # Resolver modelo
    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    # Mostrar resultados
    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        st.success('Citas optimizadas con éxito')
        for i, paciente in enumerate(pacientes):
            for j in range(num_medicos):
                for h in horarios[j]:
                    if solver.Value(asignaciones[(i, j, h)]) == 1:
                        st.write(f'{paciente} - Médico {j+1} - Hora: {h}:00')
    else:
        st.error('No se encontró una solución factible.')
