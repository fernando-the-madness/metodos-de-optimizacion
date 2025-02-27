# Función para verificar datos y período
def verificar_datos(data, acciones, fecha_inicio, fecha_fin):
    # Verificar si los datos están vacíos
    if data.empty:
        st.error("No se pudieron descargar datos para las acciones seleccionadas. Verifica los símbolos y el período.")
        return False
    
    # Verificar si la columna 'Adj Close' está presente
    if 'Adj Close' not in data.columns:
        st.error("Los datos descargados no contienen precios ajustados ('Adj Close').")
        return False
    
    # Verificar si hay suficientes datos (al menos 30 días)
    if len(data) < 30:
        st.error(f"No hay suficientes datos para el período seleccionado ({len(data)} días). Por favor, selecciona un período más largo.")
        return False
    
    # Verificar si hay valores faltantes
    if data.isnull().values.any():
        st.warning("Algunos datos están faltantes. Eliminando filas con valores NaN.")
        data = data.dropna()
        if len(data) < 30:
            st.error("Después de eliminar valores faltantes, no hay suficientes datos para el análisis.")
            return False
    
    # Verificar el rango de fechas disponible
    fecha_minima = data.index.min()
    fecha_maxima = data.index.max()
    
    if fecha_inicio < fecha_minima or fecha_fin > fecha_maxima:
        st.warning(f"El período seleccionado está fuera del rango de datos disponibles. Datos disponibles desde {fecha_minima.date()} hasta {fecha_maxima.date()}.")
        return False
    
    # Mostrar información sobre el período y las acciones con datos
    st.success(f"Datos disponibles desde **{fecha_minima.date()}** hasta **{fecha_maxima.date()}**.")
    
    # Verificar qué acciones tienen datos completos
    acciones_con_datos = data['Adj Close'].columns[data['Adj Close'].isnull().sum() == 0]
    if len(acciones_con_datos) < len(acciones):
        st.warning(f"Algunas acciones no tienen datos completos en el período seleccionado. Acciones con datos completos: {', '.join(acciones_con_datos)}.")
    else:
        st.success(f"Todas las acciones seleccionadas tienen datos completos en el período seleccionado: {', '.join(acciones)}.")
    
    return True

# Función principal para mostrar resultados
def mostrar_resultados(acciones, fecha_inicio, fecha_fin, aversion_riesgo):
    # Validar símbolos
    acciones = validar_simbolos(acciones)
    if len(acciones) < 2:
        st.error("Se necesitan al menos 2 acciones válidas para continuar.")
        return
    
    # Obtener los datos históricos
    try:
        # Descargar precios históricos
        with st.spinner('Descargando datos históricos de precios...'):
            data = yf.download(acciones, start=fecha_inicio, end=fecha_fin)
        
        # Verificar datos y período
        if not verificar_datos(data, acciones, fecha_inicio, fecha_fin):
            return
        
        # Manejar caso de una sola acción
        if len(acciones) == 1:
            data = pd.DataFrame(data['Adj Close'], columns=acciones)
        else:
            data = data['Adj Close']
        
        # Mostrar datos de precios
        with st.expander("Ver datos de precios", expanded=False):
            st.dataframe(data)
        
        # Calcular rendimientos
        rendimientos = calcular_rendimientos(data)
        
        # Resto del código para la optimización y visualización...
        
    except Exception as e:
        st.error(f"Error durante el análisis: {str(e)}")
        st.error("Por favor, verifica que los símbolos de las acciones sean válidos y que haya datos disponibles para el periodo seleccionado.")
