import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from pulp import *
import folium
from streamlit_folium import folium_static
import base64
from io import BytesIO

# Configuración de la página
st.set_page_config(
    page_title="Optimizador de Cadena de Suministro para PyMEs",
    page_icon="🏭",
    layout="wide"
)

# Estilos CSS personalizados
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #0D47A1;
    }
    .info-box {
        background-color: #E3F2FD;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .success-box {
        background-color: #E8F5E9;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Título y descripción
st.markdown("<h1 class='main-header'>Optimizador de Cadena de Suministro para PyMEs</h1>", unsafe_allow_html=True)
st.markdown("""
<div class='info-box'>
Esta aplicación te ayuda a optimizar tu cadena de suministro, reduciendo costos y mejorando la eficiencia operativa.
Podrás optimizar inventarios, rutas de transporte y programación de producción.
</div>
""", unsafe_allow_html=True)

# Funciones auxiliares
def generate_synthetic_data(num_products, num_warehouses, num_retail_locations, num_periods):
    """Genera datos sintéticos para la aplicación"""
    np.random.seed(42)
    
    # Productos
    products = {f"Producto_{i}": np.random.randint(5, 50) for i in range(1, num_products + 1)}
    
    # Almacenes
    warehouses = {}
    for i in range(1, num_warehouses + 1):
        warehouse_name = f"Almacén_{i}"
        lat = np.random.uniform(19.0, 25.0)  # Latitudes en México
        lon = np.random.uniform(-106.0, -99.0)  # Longitudes en México
        capacity = np.random.randint(1000, 10000)
        storage_cost = np.random.uniform(0.5, 2.0)
        warehouses[warehouse_name] = {
            "latitude": lat,
            "longitude": lon,
            "capacity": capacity,
            "storage_cost": storage_cost
        }
    
    # Tiendas minoristas
    retail_locations = {}
    for i in range(1, num_retail_locations + 1):
        retail_name = f"Tienda_{i}"
        lat = np.random.uniform(19.0, 25.0)
        lon = np.random.uniform(-106.0, -99.0)
        retail_locations[retail_name] = {
            "latitude": lat,
            "longitude": lon
        }
    
    # Demanda por tienda y producto
    demand = {}
    for period in range(1, num_periods + 1):
        demand[period] = {}
        for retail in retail_locations:
            demand[period][retail] = {}
            for product in products:
                # Demanda base con tendencia y estacionalidad simple
                base_demand = np.random.randint(10, 100)
                trend = period * np.random.uniform(0.05, 0.15)
                seasonality = np.sin(period / 12 * 2 * np.pi) * 10
                demand[period][retail][product] = max(5, int(base_demand + trend + seasonality))
    
    # Costos de producción
    production_costs = {product: np.random.uniform(5, 20) for product in products}
    
    # Costos de transporte entre almacenes y tiendas
    transport_costs = {}
    for warehouse in warehouses:
        transport_costs[warehouse] = {}
        w_lat, w_lon = warehouses[warehouse]["latitude"], warehouses[warehouse]["longitude"]
        for retail in retail_locations:
            r_lat, r_lon = retail_locations[retail]["latitude"], retail_locations[retail]["longitude"]
            # Costo basado en la distancia euclidiana
            distance = np.sqrt((w_lat - r_lat)**2 + (w_lon - r_lon)**2)
            transport_costs[warehouse][retail] = distance * np.random.uniform(5, 10)
    
    # Tiempos de entrega entre almacenes y tiendas (en días)
    lead_times = {}
    for warehouse in warehouses:
        lead_times[warehouse] = {}
        for retail in retail_locations:
            # Tiempo basado en la distancia pero con variabilidad
            distance = transport_costs[warehouse][retail]
            lead_times[warehouse][retail] = max(1, int(distance * 0.1 + np.random.randint(0, 3)))
    
    return {
        "products": products,
        "warehouses": warehouses,
        "retail_locations": retail_locations,
        "demand": demand,
        "production_costs": production_costs,
        "transport_costs": transport_costs,
        "lead_times": lead_times
    }

def optimize_inventory(data, selected_period, safety_stock_factor=0.2):
    """Optimiza los niveles de inventario para cada producto y almacén"""
    products = data["products"]
    warehouses = data["warehouses"]
    retail_locations = data["retail_locations"]
    demand = data["demand"]
    production_costs = data["production_costs"]
    transport_costs = data["transport_costs"]
    
    # Problema de optimización
    model = LpProblem("Inventory_Optimization", LpMinimize)
    
    # Variables de decisión
    # x[i][j][k] = cantidad del producto i enviado desde el almacén j a la tienda k
    x = {}
    for product in products:
        x[product] = {}
        for warehouse in warehouses:
            x[product][warehouse] = {}
            for retail in retail_locations:
                x[product][warehouse][retail] = LpVariable(f"x_{product}_{warehouse}_{retail}", lowBound=0, cat=LpInteger)
    
    # y[i][j] = cantidad del producto i almacenado en el almacén j
    y = {}
    for product in products:
        y[product] = {}
        for warehouse in warehouses:
            y[product][warehouse] = LpVariable(f"y_{product}_{warehouse}", lowBound=0, cat=LpInteger)
    
    # Función objetivo: minimizar costos totales
    # Costo de almacenamiento + Costo de transporte + Costo de producción
    objective = LpAffineExpression()
    
    # Costos de almacenamiento
    for product in products:
        for warehouse in warehouses:
            objective += y[product][warehouse] * warehouses[warehouse]["storage_cost"]
    
    # Costos de transporte
    for product in products:
        for warehouse in warehouses:
            for retail in retail_locations:
                objective += x[product][warehouse][retail] * transport_costs[warehouse][retail]
    
    # Costos de producción (simplificado)
    for product in products:
        for warehouse in warehouses:
            for retail in retail_locations:
                objective += x[product][warehouse][retail] * production_costs[product]
    
    model += objective
    
    # Restricciones
    # 1. Satisfacer la demanda
    for product in products:
        for retail in retail_locations:
            model += lpSum([x[product][warehouse][retail] for warehouse in warehouses]) >= demand[selected_period][retail][product], f"Demand_{product}_{retail}"
    
    # 2. Capacidad del almacén
    for warehouse in warehouses:
        model += lpSum([y[product][warehouse] * products[product] for product in products]) <= warehouses[warehouse]["capacity"], f"Capacity_{warehouse}"
    
    # 3. Balance de inventario
    for product in products:
        for warehouse in warehouses:
            total_outgoing = lpSum([x[product][warehouse][retail] for retail in retail_locations])
            safety_stock = safety_stock_factor * lpSum([demand[selected_period][retail][product] for retail in retail_locations])
            model += y[product][warehouse] >= total_outgoing + safety_stock, f"Balance_{product}_{warehouse}"
    
    # Resolver el modelo
    model.solve(PULP_CBC_CMD(msg=False))
    
    # Resultados
    results = {
        "status": LpStatus[model.status],
        "total_cost": value(model.objective),
        "inventory_levels": {},
        "shipments": {}
    }
    
    if model.status == LpStatusOptimal:
        for product in products:
            results["inventory_levels"][product] = {}
            results["shipments"][product] = {}
            for warehouse in warehouses:
                results["inventory_levels"][product][warehouse] = value(y[product][warehouse])
                results["shipments"][product][warehouse] = {}
                for retail in retail_locations:
                    results["shipments"][product][warehouse][retail] = value(x[product][warehouse][retail])
    
    return results

def optimize_transport(data, selected_period):
    """Optimiza las rutas de transporte entre almacenes y tiendas"""
    warehouses = data["warehouses"]
    retail_locations = data["retail_locations"]
    products = data["products"]
    demand = data["demand"]
    transport_costs = data["transport_costs"]
    
    # Problema de optimización
    model = LpProblem("Transport_Optimization", LpMinimize)
    
    # Variables de decisión
    # z[i][j] = 1 si se establece una ruta entre el almacén i y la tienda j, 0 en caso contrario
    z = {}
    for warehouse in warehouses:
        z[warehouse] = {}
        for retail in retail_locations:
            z[warehouse][retail] = LpVariable(f"z_{warehouse}_{retail}", cat=LpBinary)
    
    # Cantidad total transportada entre almacenes y tiendas
    q = {}
    for warehouse in warehouses:
        q[warehouse] = {}
        for retail in retail_locations:
            q[warehouse][retail] = LpVariable(f"q_{warehouse}_{retail}", lowBound=0)
    
    # Función objetivo: minimizar costos de transporte
    objective = LpAffineExpression()
    for warehouse in warehouses:
        for retail in retail_locations:
            objective += z[warehouse][retail] * transport_costs[warehouse][retail]
    
    model += objective
    
    # Restricciones
    # 1. Cada tienda debe recibir productos de al menos un almacén
    for retail in retail_locations:
        model += lpSum([z[warehouse][retail] for warehouse in warehouses]) >= 1, f"Supply_{retail}"
    
    # 2. Capacidad máxima por ruta (simplificado)
    M = 10000  # Un número grande
    for warehouse in warehouses:
        for retail in retail_locations:
            # Si z[warehouse][retail] = 0, entonces q[warehouse][retail] = 0
            model += q[warehouse][retail] <= M * z[warehouse][retail], f"Capacity_{warehouse}_{retail}"
    
    # 3. Demanda total por tienda
    for retail in retail_locations:
        total_demand = sum([demand[selected_period][retail][product] * products[product] for product in products])
        model += lpSum([q[warehouse][retail] for warehouse in warehouses]) >= total_demand, f"Demand_{retail}"
    
    # Resolver el modelo
    model.solve(PULP_CBC_CMD(msg=False))
    
    # Resultados
    results = {
        "status": LpStatus[model.status],
        "total_cost": value(model.objective),
        "routes": {}
    }
    
    if model.status == LpStatusOptimal:
        for warehouse in warehouses:
            results["routes"][warehouse] = {}
            for retail in retail_locations:
                if value(z[warehouse][retail]) > 0.5:  # Si la ruta está activa
                    results["routes"][warehouse][retail] = {
                        "active": True,
                        "quantity": value(q[warehouse][retail]),
                        "cost": transport_costs[warehouse][retail]
                    }
                else:
                    results["routes"][warehouse][retail] = {
                        "active": False,
                        "quantity": 0,
                        "cost": transport_costs[warehouse][retail]
                    }
    
    return results

def optimize_production(data, selected_period, max_production_capacity=1000):
    """Optimiza la programación de producción para satisfacer la demanda"""
    products = data["products"]
    retail_locations = data["retail_locations"]
    demand = data["demand"]
    production_costs = data["production_costs"]
    
    # Problema de optimización
    model = LpProblem("Production_Optimization", LpMinimize)
    
    # Variables de decisión
    # p[i] = cantidad producida del producto i
    p = {}
    for product in products:
        p[product] = LpVariable(f"p_{product}", lowBound=0, cat=LpInteger)
    
    # Función objetivo: minimizar costos de producción
    objective = LpAffineExpression()
    for product in products:
        objective += p[product] * production_costs[product]
    
    model += objective
    
    # Restricciones
    # 1. Satisfacer la demanda
    for product in products:
        total_demand = sum([demand[selected_period][retail][product] for retail in retail_locations])
        model += p[product] >= total_demand, f"Demand_{product}"
    
    # 2. Capacidad máxima de producción
    model += lpSum([p[product] for product in products]) <= max_production_capacity, "Max_Production"
    
    # Resolver el modelo
    model.solve(PULP_CBC_CMD(msg=False))
    
    # Resultados
    results = {
        "status": LpStatus[model.status],
        "total_cost": value(model.objective),
        "production": {}
    }
    
    if model.status == LpStatusOptimal:
        for product in products:
            results["production"][product] = value(p[product])
    
    return results

def run_simulation(data, selected_period, demand_change=0, lead_time_change=0):
    """Ejecuta una simulación con cambios en la demanda o tiempos de entrega"""
    # Copia profunda de los datos
    import copy
    sim_data = copy.deepcopy(data)
    
    # Modificar la demanda si es necesario
    if demand_change != 0:
        for retail in sim_data["retail_locations"]:
            for product in sim_data["products"]:
                current_demand = sim_data["demand"][selected_period][retail][product]
                sim_data["demand"][selected_period][retail][product] = int(current_demand * (1 + demand_change/100))
    
    # Modificar los tiempos de entrega si es necesario
    if lead_time_change != 0:
        for warehouse in sim_data["warehouses"]:
            for retail in sim_data["retail_locations"]:
                current_lead_time = sim_data["lead_times"][warehouse][retail]
                sim_data["lead_times"][warehouse][retail] = max(1, int(current_lead_time * (1 + lead_time_change/100)))
    
    # Ejecutar optimizaciones
    inventory_results = optimize_inventory(sim_data, selected_period)
    transport_results = optimize_transport(sim_data, selected_period)
    production_results = optimize_production(sim_data, selected_period)
    
    return {
        "inventory": inventory_results,
        "transport": transport_results,
        "production": production_results,
        "modified_data": sim_data
    }

def plot_map(data, transport_results):
    """Genera un mapa con almacenes, tiendas y rutas optimizadas"""
    warehouses = data["warehouses"]
    retail_locations = data["retail_locations"]
    
    # Crear mapa centrado en México
    m = folium.Map(location=[22.0, -102.0], zoom_start=5)
    
    # Añadir almacenes
    for warehouse, info in warehouses.items():
        folium.Marker(
            location=[info["latitude"], info["longitude"]],
            popup=f"{warehouse}<br>Capacidad: {info['capacity']}",
            icon=folium.Icon(color="blue", icon="industry", prefix="fa")
        ).add_to(m)
    
    # Añadir tiendas
    for retail, info in retail_locations.items():
        folium.Marker(
            location=[info["latitude"], info["longitude"]],
            popup=retail,
            icon=folium.Icon(color="green", icon="shopping-cart", prefix="fa")
        ).add_to(m)
    
    # Añadir rutas optimizadas
    for warehouse in transport_results["routes"]:
        for retail in transport_results["routes"][warehouse]:
            route_info = transport_results["routes"][warehouse][retail]
            if route_info["active"]:
                w_lat = warehouses[warehouse]["latitude"]
                w_lon = warehouses[warehouse]["longitude"]
                r_lat = retail_locations[retail]["latitude"]
                r_lon = retail_locations[retail]["longitude"]
                
                folium.PolyLine(
                    locations=[[w_lat, w_lon], [r_lat, r_lon]],
                    color="red",
                    weight=2,
                    opacity=0.7,
                    popup=f"Costo: ${route_info['cost']:.2f}"
                ).add_to(m)
    
    return m

def create_inventory_chart(inventory_results):
    """Crea un gráfico de barras para los niveles de inventario"""
    inventory_data = []
    for product in inventory_results["inventory_levels"]:
        for warehouse in inventory_results["inventory_levels"][product]:
            inventory_data.append({
                "Producto": product,
                "Almacén": warehouse,
                "Nivel de Inventario": inventory_results["inventory_levels"][product][warehouse]
            })
    
    if not inventory_data:
        return None
    
    df = pd.DataFrame(inventory_data)
    fig = px.bar(
        df, 
        x="Almacén", 
        y="Nivel de Inventario", 
        color="Producto",
        title="Niveles de Inventario Optimizados",
        labels={"Nivel de Inventario": "Unidades", "Almacén": ""},
        barmode="group"
    )
    
    return fig

def create_production_chart(production_results):
    """Crea un gráfico de barras para la producción optimizada"""
    production_data = []
    for product, quantity in production_results["production"].items():
        production_data.append({
            "Producto": product,
            "Cantidad": quantity
        })
    
    if not production_data:
        return None
    
    df = pd.DataFrame(production_data)
    fig = px.bar(
        df, 
        x="Producto", 
        y="Cantidad",
        title="Producción Optimizada",
        labels={"Cantidad": "Unidades a Producir", "Producto": ""},
        color="Producto"
    )
    
    return fig

def create_transport_chart(transport_results):
    """Crea un gráfico de barras para los costos de transporte"""
    transport_data = []
    for warehouse in transport_results["routes"]:
        active_routes = 0
        total_cost = 0
        for retail, info in transport_results["routes"][warehouse].items():
            if info["active"]:
                active_routes += 1
                total_cost += info["cost"]
        
        transport_data.append({
            "Almacén": warehouse,
            "Rutas Activas": active_routes,
            "Costo Total": total_cost
        })
    
    if not transport_data:
        return None
    
    df = pd.DataFrame(transport_data)
    fig = px.bar(
        df, 
        x="Almacén", 
        y="Costo Total",
        title="Costos de Transporte por Almacén",
        labels={"Costo Total": "Costo ($)", "Almacén": ""},
        color="Rutas Activas",
        color_continuous_scale="Viridis"
    )
    
    return fig

def create_cost_summary_chart(inventory_results, transport_results, production_results):
    """Crea un gráfico de pastel para el resumen de costos"""
    inventory_cost = inventory_results["total_cost"] if inventory_results["status"] == "Optimal" else 0
    transport_cost = transport_results["total_cost"] if transport_results["status"] == "Optimal" else 0
    production_cost = production_results["total_cost"] if production_results["status"] == "Optimal" else 0
    
    total_cost = inventory_cost + transport_cost + production_cost
    
    if total_cost == 0:
        return None
    
    labels = ['Inventario', 'Transporte', 'Producción']
    values = [inventory_cost, transport_cost, production_cost]
    percentages = [value/total_cost*100 for value in values]
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hovertemplate='%{label}: $%{value:.2f}<br>%{percent}',
        textinfo='label+percent',
        marker=dict(colors=['#1E88E5', '#FFC107', '#4CAF50'])
    )])
    
    fig.update_layout(
        title_text=f'Distribución de Costos Totales: ${total_cost:.2f}',
        showlegend=True
    )
    
    return fig

def download_excel_report(data, results):
    """Genera un informe en Excel para descargar"""
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    
    # Hoja de inventario
    inventory_data = []
    for product in results["inventory"]["inventory_levels"]:
        for warehouse in results["inventory"]["inventory_levels"][product]:
            inventory_data.append({
                "Producto": product,
                "Almacén": warehouse,
                "Nivel de Inventario": results["inventory"]["inventory_levels"][product][warehouse]
            })
    
    if inventory_data:
        inventory_df = pd.DataFrame(inventory_data)
        inventory_df.to_excel(writer, sheet_name='Inventario', index=False)
    
    # Hoja de transporte
    transport_data = []
    for warehouse in results["transport"]["routes"]:
        for retail, info in results["transport"]["routes"][warehouse].items():
            if info["active"]:
                transport_data.append({
                    "Almacén": warehouse,
                    "Tienda": retail,
                    "Costo": info["cost"],
                    "Cantidad": info["quantity"]
                })
    
    if transport_data:
        transport_df = pd.DataFrame(transport_data)
        transport_df.to_excel(writer, sheet_name='Transporte', index=False)
    
    # Hoja de producción
    production_data = []
    for product, quantity in results["production"]["production"].items():
        production_data.append({
            "Producto": product,
            "Cantidad a Producir": quantity,
            "Costo Unitario": data["production_costs"][product],
            "Costo Total": quantity * data["production_costs"][product]
        })
    
    if production_data:
        production_df = pd.DataFrame(production_data)
        production_df.to_excel(writer, sheet_name='Producción', index=False)
    
    # Hoja de resumen de costos
    cost_summary = {
        "Categoría": ["Inventario", "Transporte", "Producción", "Total"],
        "Costo": [
            results["inventory"]["total_cost"],
            results["transport"]["total_cost"],
            results["production"]["total_cost"],
            results["inventory"]["total_cost"] + results["transport"]["total_cost"] + results["production"]["total_cost"]
        ]
    }
    cost_df = pd.DataFrame(cost_summary)
    cost_df.to_excel(writer, sheet_name='Resumen', index=False)
    
    writer.close()
    
    return output.getvalue()

# Menú principal
menu = st.sidebar.selectbox(
    "Menú Principal",
    ["Inicio", "Datos de Entrada", "Optimización", "Simulación", "Visualización", "Reportes"]
)

# Inicializar variables de sesión
if 'data' not in st.session_state:
    st.session_state.data = None
if 'results' not in st.session_state:
    st.session_state.results = None
if 'simulation_results' not in st.session_state:
    st.session_state.simulation_results = None

# Página de inicio
if menu == "Inicio":
    st.markdown("<h2 class='sub-header'>Bienvenido al Optimizador de Cadena de Suministro</h2>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class='info-box'>
    <h3>¿Qué puedes hacer con esta aplicación?</h3>
    <ul>
        <li><b>Optimizar inventarios:</b> Determina los niveles óptimos de inventario para reducir costos de almacenamiento.</li>
        <li><b>Optimizar transporte:</b> Encuentra las rutas más eficientes entre almacenes y tiendas.</li>
        <li><b>Planificar la producción:</b> Programa la fabricación de productos para satisfacer la demanda.</li>
        <li><b>Simular escenarios:</b> Analiza qué pasaría si cambia la demanda o hay retrasos en el transporte.</li>
        <li><b>Visualizar resultados:</b> Observa gráficamente el impacto de tus decisiones.</li>
        <li><b>Generar reportes:</b> Descarga informes detallados para compartir con tu equipo.</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<h3 class='sub-header'>¿Cómo empezar?</h3>", unsafe_allow_html=True)
    st.markdown("""
    1. Ve a "Datos de Entrada" para cargar o generar los datos de tu cadena de suministro.
    2. Utiliza la sección de "Optimización" para encontrar la configuración óptima.
    3. Explora diferentes escenarios en la sección de "Simulación".
    4. Visualiza los resultados gráficamente en "Visualización".
    5. Genera reportes detallados en "Reportes".
    """)
    
    st.image("https://raw.githubusercontent.com/streamlit/example-app-interactive-table/master/streamlit-screenshot.png", caption="Ejemplo de visualización", use_column_width=True)
    
    st.markdown("<h3 class='sub-header'>Beneficios de optimizar tu cadena de suministro</h3>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        ### 📉 Reducción de Costos
        - Menor inventario
        - Rutas eficientes
        - Producción optimizada
        """)
    with col2:
        st.markdown("""
        ### ⏱️ Mejor Servicio
        - Entregas más rápidas
        - Mayor disponibilidad
        - Menos roturas de stock
        """)
    with col3:
        st.markdown("""
        ### 📊 Mayor Visibilidad
        - Datos en tiempo real
        - Decisiones informadas
        - Planificación eficaz
        """)

# Datos de Entrada
elif menu == "Datos de Entrada":
    st.markdown("<h2 class='sub-header'>Datos de Entrada</h2>", unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["Generar Datos de Ejemplo", "Cargar Datos"])
    
    with tab1:
        st.markdown("Genera datos sintéticos para probar la aplicación")
        
        col1, col2 = st.columns(2)
        with col1:
            num_products = st.slider("Número de productos", 2, 10, 5)
            num_warehouses = st.slider("Número de almacenes", 2, 8, 3)
        with col2:
            num_retail = st.slider("Número de tiendas", 2, 15, 5)
            num_periods = st.slider("Número de períodos", 1, 12, 6)
        
        if st.button("Generar Datos de Ejemplo"):
            with st.spinner("Generando datos..."):
                st.session_state.data = generate_synthetic_data(num_products, num_warehouses, num_retail, num_periods)
                st.session_state.results = None
                st.session_state.simulation_results = None
                st.success("Datos generados correctamente")
    
    with tab2:
        st.markdown("Carga tus propios datos (Funcionalidad en desarrollo)")
        
        # Esta sección se implementaría con st.file_uploader para permitir cargar archivos Excel o CSV
        st.info("Esta funcionalidad estará disponible en próximas versiones")
    
    if st.session_state.data:
        st.markdown("<h3 class='sub-header'>Resumen de Datos</h3>", unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Productos", len(st.session_state.data["products"]))
        with col2:
            st.metric("Almacenes", len(st.session_state.data["warehouses"]))
        with col3:
            st.metric("Tiendas", len(st.session_state.data["retail_locations"]))
        
        # Previsualización de productos
                    # Previsualización de productos
        st.markdown("<h4>Productos y sus dimensiones (volumen unitario)</h4>", unsafe_allow_html=True)
        
        products_df = pd.DataFrame({
            "Producto": list(st.session_state.data["products"].keys()),
            "Volumen Unitario": list(st.session_state.data["products"].values())
        })
        st.dataframe(products_df)
        
        # Previsualización de almacenes
        st.markdown("<h4>Almacenes y sus capacidades</h4>", unsafe_allow_html=True)
        warehouses_data = []
        for warehouse, info in st.session_state.data["warehouses"].items():
            warehouses_data.append({
                "Almacén": warehouse,
                "Capacidad": info["capacity"],
                "Costo de Almacenamiento": f"${info['storage_cost']:.2f} por unidad"
            })
        warehouses_df = pd.DataFrame(warehouses_data)
        st.dataframe(warehouses_df)
        
        # Mapa de ubicaciones
        st.markdown("<h4>Mapa de almacenes y tiendas</h4>", unsafe_allow_html=True)
        locations_map = folium.Map(location=[22.0, -102.0], zoom_start=5)
        
        # Añadir almacenes al mapa
        for warehouse, info in st.session_state.data["warehouses"].items():
            folium.Marker(
                location=[info["latitude"], info["longitude"]],
                popup=f"{warehouse}<br>Capacidad: {info['capacity']}",
                icon=folium.Icon(color="blue", icon="industry", prefix="fa")
            ).add_to(locations_map)
        
        # Añadir tiendas al mapa
        for retail, info in st.session_state.data["retail_locations"].items():
            folium.Marker(
                location=[info["latitude"], info["longitude"]],
                popup=retail,
                icon=folium.Icon(color="green", icon="shopping-cart", prefix="fa")
            ).add_to(locations_map)
        
        folium_static(locations_map)

# Optimización
elif menu == "Optimización":
    st.markdown("<h2 class='sub-header'>Optimización de la Cadena de Suministro</h2>", unsafe_allow_html=True)
    
    if not st.session_state.data:
        st.warning("No hay datos disponibles. Por favor, genera o carga datos en la sección 'Datos de Entrada'.")
    else:
        st.markdown("""
        <div class='info-box'>
        Selecciona el período para el que deseas optimizar la cadena de suministro y ajusta los parámetros según tus necesidades.
        </div>
        """, unsafe_allow_html=True)
        
        # Selección de período
        periods = list(range(1, len(st.session_state.data["demand"]) + 1))
        selected_period = st.selectbox("Selecciona el período", periods)
        
        # Parámetros de optimización
        st.markdown("<h3 class='sub-header'>Parámetros de Optimización</h3>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        
        with col1:
            safety_stock = st.slider("Factor de stock de seguridad (%)", 5, 50, 20) / 100
        
        with col2:
            max_production = st.slider("Capacidad máxima de producción", 500, 5000, 1000)
        
        # Botón para ejecutar la optimización
        if st.button("Ejecutar Optimización"):
            with st.spinner("Optimizando cadena de suministro..."):
                # Optimización de inventario
                inventory_results = optimize_inventory(st.session_state.data, selected_period, safety_stock)
                
                # Optimización de transporte
                transport_results = optimize_transport(st.session_state.data, selected_period)
                
                # Optimización de producción
                production_results = optimize_production(st.session_state.data, selected_period, max_production)
                
                # Guardar resultados en la sesión
                st.session_state.results = {
                    "inventory": inventory_results,
                    "transport": transport_results,
                    "production": production_results,
                    "period": selected_period,
                    "parameters": {
                        "safety_stock": safety_stock,
                        "max_production": max_production
                    }
                }
                
                st.success("Optimización completada correctamente")
        
        # Mostrar resultados si están disponibles
        if st.session_state.results:
            st.markdown("<h3 class='sub-header'>Resultados de la Optimización</h3>", unsafe_allow_html=True)
            
            tab1, tab2, tab3, tab4 = st.tabs(["Resumen", "Inventario", "Transporte", "Producción"])
            
            with tab1:
                col1, col2, col3 = st.columns(3)
                with col1:
                    inventory_cost = st.session_state.results["inventory"]["total_cost"]
                    st.metric("Costo de Inventario", f"${inventory_cost:.2f}")
                with col2:
                    transport_cost = st.session_state.results["transport"]["total_cost"]
                    st.metric("Costo de Transporte", f"${transport_cost:.2f}")
                with col3:
                    production_cost = st.session_state.results["production"]["total_cost"]
                    st.metric("Costo de Producción", f"${production_cost:.2f}")
                
                total_cost = inventory_cost + transport_cost + production_cost
                st.metric("Costo Total", f"${total_cost:.2f}")
                
                fig = create_cost_summary_chart(
                    st.session_state.results["inventory"], 
                    st.session_state.results["transport"], 
                    st.session_state.results["production"]
                )
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
            
            with tab2:
                st.markdown("### Niveles óptimos de inventario")
                fig = create_inventory_chart(st.session_state.results["inventory"])
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                
                # Detalles por producto y almacén
                inventory_details = []
                for product in st.session_state.results["inventory"]["inventory_levels"]:
                    for warehouse, level in st.session_state.results["inventory"]["inventory_levels"][product].items():
                        storage_cost = st.session_state.data["warehouses"][warehouse]["storage_cost"]
                        inventory_details.append({
                            "Producto": product,
                            "Almacén": warehouse,
                            "Nivel de Inventario": level,
                            "Costo Unitario de Almacenamiento": f"${storage_cost:.2f}",
                            "Costo Total": f"${level * storage_cost:.2f}"
                        })
                
                if inventory_details:
                    st.dataframe(pd.DataFrame(inventory_details))
            
            with tab3:
                st.markdown("### Rutas de transporte optimizadas")
                transport_map = plot_map(st.session_state.data, st.session_state.results["transport"])
                folium_static(transport_map)
                
                fig = create_transport_chart(st.session_state.results["transport"])
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                
                # Detalles de rutas activas
                route_details = []
                for warehouse in st.session_state.results["transport"]["routes"]:
                    for retail, info in st.session_state.results["transport"]["routes"][warehouse].items():
                        if info["active"]:
                            route_details.append({
                                "Origen": warehouse,
                                "Destino": retail,
                                "Costo": f"${info['cost']:.2f}",
                                "Cantidad": f"{info['quantity']:.1f} unidades",
                                "Tiempo Estimado": f"{st.session_state.data['lead_times'][warehouse][retail]} días"
                            })
                
                if route_details:
                    st.dataframe(pd.DataFrame(route_details))
            
            with tab4:
                st.markdown("### Programación de producción")
                fig = create_production_chart(st.session_state.results["production"])
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                
                # Detalles de producción
                production_details = []
                for product, quantity in st.session_state.results["production"]["production"].items():
                    unit_cost = st.session_state.data["production_costs"][product]
                    production_details.append({
                        "Producto": product,
                        "Cantidad a Producir": quantity,
                        "Costo Unitario": f"${unit_cost:.2f}",
                        "Costo Total": f"${quantity * unit_cost:.2f}"
                    })
                
                if production_details:
                    st.dataframe(pd.DataFrame(production_details))

# Simulación
elif menu == "Simulación":
    st.markdown("<h2 class='sub-header'>Simulación de Escenarios</h2>", unsafe_allow_html=True)
    
    if not st.session_state.data:
        st.warning("No hay datos disponibles. Por favor, genera o carga datos en la sección 'Datos de Entrada'.")
    else:
        st.markdown("""
        <div class='info-box'>
        Simula diferentes escenarios modificando la demanda o los tiempos de entrega para ver cómo afectan a tu cadena de suministro.
        </div>
        """, unsafe_allow_html=True)
        
        # Selección de período
        periods = list(range(1, len(st.session_state.data["demand"]) + 1))
        selected_period = st.selectbox("Selecciona el período para la simulación", periods)
        
        # Parámetros de simulación
        st.markdown("<h3 class='sub-header'>Parámetros de Simulación</h3>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        
        with col1:
            demand_change = st.slider("Cambio en la demanda (%)", -50, 100, 0)
            st.caption("Un valor positivo indica un aumento, uno negativo una disminución")
        
        with col2:
            lead_time_change = st.slider("Cambio en tiempos de entrega (%)", -50, 100, 0)
            st.caption("Un valor positivo indica retrasos, uno negativo entregas más rápidas")
        
        # Botón para ejecutar la simulación
        if st.button("Ejecutar Simulación"):
            with st.spinner("Simulando escenario..."):
                simulation_results = run_simulation(
                    st.session_state.data, 
                    selected_period, 
                    demand_change, 
                    lead_time_change
                )
                
                # Guardar resultados de la simulación
                st.session_state.simulation_results = simulation_results
                
                st.success("Simulación completada correctamente")
        
        # Mostrar resultados si están disponibles
        if st.session_state.simulation_results:
            st.markdown("<h3 class='sub-header'>Resultados de la Simulación</h3>", unsafe_allow_html=True)
            
            # Comparativa de costos si hay resultados de optimización
            if st.session_state.results and st.session_state.results["period"] == selected_period:
                # Costos originales
                original_inventory = st.session_state.results["inventory"]["total_cost"]
                original_transport = st.session_state.results["transport"]["total_cost"]
                original_production = st.session_state.results["production"]["total_cost"]
                original_total = original_inventory + original_transport + original_production
                
                # Costos simulados
                sim_inventory = st.session_state.simulation_results["inventory"]["total_cost"]
                sim_transport = st.session_state.simulation_results["transport"]["total_cost"]
                sim_production = st.session_state.simulation_results["production"]["total_cost"]
                sim_total = sim_inventory + sim_transport + sim_production
                
                # Diferencias porcentuales
                inventory_diff = (sim_inventory - original_inventory) / original_inventory * 100 if original_inventory > 0 else 0
                transport_diff = (sim_transport - original_transport) / original_transport * 100 if original_transport > 0 else 0
                production_diff = (sim_production - original_production) / original_production * 100 if original_production > 0 else 0
                total_diff = (sim_total - original_total) / original_total * 100 if original_total > 0 else 0
                
                st.markdown("<h4>Comparativa de Costos</h4>", unsafe_allow_html=True)
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Inventario", f"${sim_inventory:.2f}", f"{inventory_diff:.1f}%")
                
                with col2:
                    st.metric("Transporte", f"${sim_transport:.2f}", f"{transport_diff:.1f}%")
                
                with col3:
                    st.metric("Producción", f"${sim_production:.2f}", f"{production_diff:.1f}%")
                
                with col4:
                    st.metric("Total", f"${sim_total:.2f}", f"{total_diff:.1f}%")
                
                # Gráfico comparativo
                categories = ['Inventario', 'Transporte', 'Producción', 'Total']
                original_values = [original_inventory, original_transport, original_production, original_total]
                sim_values = [sim_inventory, sim_transport, sim_production, sim_total]
                
                fig = go.Figure(data=[
                    go.Bar(name='Original', x=categories, y=original_values),
                    go.Bar(name='Simulación', x=categories, y=sim_values)
                ])
                
                fig.update_layout(
                    title='Comparativa de Costos: Original vs. Simulación',
                    barmode='group',
                    yaxis_title='Costo ($)'
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Detalles de la simulación
            st.markdown("<h4>Detalles de la Simulación</h4>", unsafe_allow_html=True)
            
            tab1, tab2, tab3 = st.tabs(["Inventario", "Transporte", "Producción"])
            
            with tab1:
                fig = create_inventory_chart(st.session_state.simulation_results["inventory"])
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
            
            with tab2:
                sim_transport_map = plot_map(st.session_state.simulation_results["modified_data"], 
                                           st.session_state.simulation_results["transport"])
                folium_static(sim_transport_map)
            
            with tab3:
                fig = create_production_chart(st.session_state.simulation_results["production"])
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
            
            # Análisis de impacto
            st.markdown("<h4>Análisis de Impacto</h4>", unsafe_allow_html=True)
            
            impact_text = ""
            if demand_change != 0:
                impact_text += f"**Impacto del cambio de demanda ({demand_change}%):**\n\n"
                if demand_change > 0:
                    impact_text += "- El aumento de la demanda requiere mayor inventario y producción\n"
                    impact_text += "- Se observa un incremento en los costos totales\n"
                else:
                    impact_text += "- La disminución de la demanda permite reducir inventarios\n"
                    impact_text += "- Se observa una reducción en los costos de producción\n"
            
            if lead_time_change != 0:
                impact_text += f"\n**Impacto del cambio en tiempos de entrega ({lead_time_change}%):**\n\n"
                if lead_time_change > 0:
                    impact_text += "- Retrasos en las entregas requieren mayores inventarios de seguridad\n"
                    impact_text += "- Posible redistribución de rutas de transporte\n"
                else:
                    impact_text += "- Entregas más rápidas permiten reducir inventarios de seguridad\n"
                    impact_text += "- Mejor capacidad de respuesta a la demanda\n"
            
            if impact_text:
                st.markdown(impact_text)
            else:
                st.info("No se han realizado cambios en la simulación con respecto al escenario original.")

# Visualización
elif menu == "Visualización":
    st.markdown("<h2 class='sub-header'>Visualización de Resultados</h2>", unsafe_allow_html=True)
    
    if not st.session_state.results:
        st.warning("No hay resultados disponibles. Por favor, ejecuta una optimización primero.")
    else:
        st.markdown("""
        <div class='info-box'>
        Visualiza los resultados de la optimización desde diferentes perspectivas.
        </div>
        """, unsafe_allow_html=True)
        
        viz_option = st.radio(
            "Selecciona qué deseas visualizar",
            ["Mapa de la Cadena de Suministro", "Análisis de Costos", "Inventarios", "Producción"]
        )
        
        if viz_option == "Mapa de la Cadena de Suministro":
            st.markdown("<h3 class='sub-header'>Mapa de la Cadena de Suministro Optimizada</h3>", unsafe_allow_html=True)
            
            transport_map = plot_map(st.session_state.data, st.session_state.results["transport"])
            folium_static(transport_map)
            
            # Información adicional sobre rutas
            st.markdown("<h4>Rutas de Transporte Activas</h4>", unsafe_allow_html=True)
            
            route_data = []
            for warehouse in st.session_state.results["transport"]["routes"]:
                for retail, info in st.session_state.results["transport"]["routes"][warehouse].items():
                    if info["active"]:
                        route_data.append({
                            "Origen": warehouse,
                            "Destino": retail,
                            "Costo": f"${info['cost']:.2f}",
                            "Tiempo de Entrega": f"{st.session_state.data['lead_times'][warehouse][retail]} días"
                        })
            
            if route_data:
                route_df = pd.DataFrame(route_data)
                st.dataframe(route_df)
        
        elif viz_option == "Análisis de Costos":
            st.markdown("<h3 class='sub-header'>Análisis Detallado de Costos</h3>", unsafe_allow_html=True)
            
            # Resumen de costos
            inventory_cost = st.session_state.results["inventory"]["total_cost"]
            transport_cost = st.session_state.results["transport"]["total_cost"]
            production_cost = st.session_state.results["production"]["total_cost"]
            total_cost = inventory_cost + transport_cost + production_cost
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Costo de Inventario", f"${inventory_cost:.2f}", f"{inventory_cost/total_cost*100:.1f}%")
            with col2:
                st.metric("Costo de Transporte", f"${transport_cost:.2f}", f"{transport_cost/total_cost*100:.1f}%")
            with col3:
                st.metric("Costo de Producción", f"${production_cost:.2f}", f"{production_cost/total_cost*100:.1f}%")
            with col4:
                st.metric("Costo Total", f"${total_cost:.2f}")
            
            # Gráfico de pastel
            fig = create_cost_summary_chart(
                st.session_state.results["inventory"], 
                st.session_state.results["transport"], 
                st.session_state.results["production"]
            )
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            
            # Costos desglosados por producto
            st.markdown("<h4>Costos por Producto</h4>", unsafe_allow_html=True)
            
            product_costs = {}
            for product in st.session_state.data["products"]:
                # Costos de producción
                prod_qty = st.session_state.results["production"]["production"].get(product, 0)
                prod_cost = prod_qty * st.session_state.data["production_costs"][product]
                
                # Costos de inventario (simplificado)
                inv_cost = 0
                for warehouse in st.session_state.data["warehouses"]:
                    inv_level = st.session_state.results["inventory"]["inventory_levels"].get(product, {}).get(warehouse, 0)
                    inv_cost += inv_level * st.session_state.data["warehouses"][warehouse]["storage_cost"]
                
                product_costs[product] = {
                    "Producción": prod_cost,
                    "Inventario": inv_cost,
                    "Total": prod_cost + inv_cost
                }
            
            product_cost_data = []
            for product, costs in product_costs.items():
                product_cost_data.append({
                    "Producto": product,
                    "Costo de Producción": f"${costs['Producción']:.2f}",
                    "Costo de Inventario": f"${costs['Inventario']:.2f}",
                    "Costo Total": f"${costs['Total']:.2f}"
                })
            
            if product_cost_data:
                product_cost_df = pd.DataFrame(product_cost_data)
                st.dataframe(product_cost_df)
            
            # Gráfico de barras por producto
            fig = px.bar(
                pd.DataFrame([
                    {"Producto": prod, "Costo": cost["Total"], "Tipo": "Total"} for prod, cost in product_costs.items()
                ] + [
                    {"Producto": prod, "Costo": cost["Producción"], "Tipo": "Producción"} for prod, cost in product_costs.items()
                ] + [
                    {"Producto": prod, "Costo": cost["Inventario"], "Tipo": "Inventario"} for prod, cost in product_costs.items()
                ]),
                x="Producto",
                y="Costo",
                color="Tipo",
                barmode="group",
                title="Costos por Producto"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        elif viz_option == "Inventarios":
            st.markdown("<h3 class='sub-header'>Análisis de Inventarios</h3>", unsafe_allow_html=True)
            
            fig = create_inventory_chart(st.session_state.results["inventory"])
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            
            # Utilización de capacidad
            st.markdown("<h4>Utilización de Capacidad de Almacenes</h4>", unsafe_allow_html=True)
            
            warehouse_usage = {}
            for warehouse in st.session_state.data["warehouses"]:
                total_volume = 0
                for product in st.session_state.data["products"]:
                    inv_level = st.session_state.results["inventory"]["inventory_levels"].get(product, {}).get(warehouse, 0)
                    total_volume += inv_level * st.session_state.data["products"][product]
                
                capacity = st.session_state.data["warehouses"][warehouse]["capacity"]
                usage_pct = total_volume / capacity * 100 if capacity > 0 else 0
                
                warehouse_usage[warehouse] = {
                    "Volumen Total": total_volume,
                    "Capacidad": capacity,
                    "Utilización (%)": usage_pct
                }
            
            usage_data = []
            for warehouse, usage in warehouse_usage.items():
                usage_data.append({
                    "Almacén": warehouse,
                    "Volumen Ocupado": f"{usage['Volumen Total']:.2f}",
                    "Capacidad Total": usage["Capacidad"],
                    "Utilización": f"{usage['Utilización (%)']:.1f}%"
                })
            
            if usage_data:
                usage_df = pd.DataFrame(usage_data)
                st.dataframe(usage_df)
            
            # Gráfico de utilización
            fig = px.bar(
                pd.DataFrame([
                    {"Almacén": wh, "Porcentaje": usage["Utilización (%)"], "Tipo": "Utilizado"} 
                    for wh, usage in warehouse_usage.items()
                ] + [
                    {"Almacén": wh, "Porcentaje": 100 - usage["Utilización (%)"], "Tipo": "Disponible"} 
                    for wh, usage in warehouse_usage.items()
                ]),
                x="Almacén",
                y="Porcentaje",
                color="Tipo",
                title="Utilización de Capacidad de Almacenes (%)",
                barmode="stack"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        elif viz_option == "Producción":
            st.markdown("<h3 class='sub-header'>Análisis de Producción</h3>", unsafe_allow_html=True)
            
            fig = create_production_chart(st.session_state.results["production"])
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            
            # Comparativa de producción vs demanda
            st.markdown("<h4>Producción vs. Demanda</h4>", unsafe_allow_html=True)
            
            production_vs_demand = []
            for product in st.session_state.data["products"]:
                production = st.session_state.results["production"]["production"].get(product, 0)
                
                # Calcular demanda total para este producto en todas las tiendas
                total_demand = 0
                for retail in st.session_state.data["retail_locations"]:
                    total_demand += st.session_state.data["demand"][st.session_state.results["period"]][retail][product]
                
                # Calcular la diferencia
                diff = production - total_demand
                diff_pct = diff / total_demand * 100 if total_demand > 0 else 0
                
                production_vs_demand.append({
                    "Producto": product,
                    "Producción": production,
                    "Demanda": total_demand,
                    "Diferencia": diff,
                    "Diferencia (%)": f"{diff_pct:.1f}%"
                })
            
            if production_vs_demand:
                pvd_df = pd.DataFrame(production_vs_demand)
                st.dataframe(pvd_df)
            
            # Gráfico comparativo
            fig = px.bar(
                pd.DataFrame([
                    {"Producto": item["Producto"], "Valor": item["Producción"], "Tipo": "Producción"} 
                    for item in production_vs_demand
                ] + [
                    {"Producto": item["Producto"], "Valor": item["Demanda"], "Tipo": "Demanda"} 
                    for item in production_vs_demand
                ]),
                x="Producto",
                y="Valor",
                color="Tipo",
                barmode="group",
                title="Producción vs. Demanda por Producto"
            )
            st.plotly_chart(fig, use_container_width=True)

# Reportes
elif menu == "Reportes":
    st.markdown("<h2 class='sub-header'>Generación de Reportes</h2>", unsafe_allow_html=True)
    
    if not st.session_state.results:
        st.warning("No hay resultados disponibles. Por favor, ejecuta una optimización primero.")
    else:
        st.markdown("""
        <div class='info-box'>
        Genera reportes detallados sobre tu cadena de suministro optimizada que puedes descargar y compartir.
        </div>
        """, unsafe_allow_html=True)
        
        report_type = st.radio(
            "Selecciona el tipo de reporte",
            ["Reporte Completo", "Reporte de Inventario", "Reporte de Transporte", "Reporte de Producción"]
        )
        
        if report_type == "Reporte Completo":
            # Generar Excel con toda la información
            if st.button("Generar Reporte Completo (Excel)"):
                with st.spinner("Generando reporte..."):
                    excel_data = download_excel_report(st.session_state.data, st.session_state.results)
                    
                    b64 = base64.b64encode(excel_data).decode()
                    href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="cadena_suministro_optimizada.xlsx">Descargar Reporte Excel</a>'
                    st.markdown(href, unsafe_allow_html=True)
            
            # Vista previa del reporte
            st.markdown("<h3 class='sub-header'>Vista Previa del Reporte</h3>", unsafe_allow_html=True)
            
            # Resumen de costos
            inventory_cost = st.session_state.results["inventory"]["total_cost"]
            transport_cost = st.session_state.results["transport"]["total_cost"]
            production_cost = st.session_state.results["production"]["total_cost"]
            total_cost = inventory_cost + transport_cost + production_cost
            
            st.markdown(f"""
            <div class='success-box'>
            <h4>Resumen de Costos</h4>
            <div class='success-box'>
            <h4>Resumen de Costos</h4>
            <ul>
                <li>Costo de Inventario: ${inventory_cost:.2f} ({inventory_cost/total_cost*100:.1f}%)</li>
                <li>Costo de Transporte: ${transport_cost:.2f} ({transport_cost/total_cost*100:.1f}%)</li>
                <li>Costo de Producción: ${production_cost:.2f} ({production_cost/total_cost*100:.1f}%)</li>
                <li>Costo Total: ${total_cost:.2f}</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
            
            # Gráficas principales
            st.markdown("<h4>Visualizaciones Principales</h4>", unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            
            with col1:
                fig = create_cost_summary_chart(
                    st.session_state.results["inventory"], 
                    st.session_state.results["transport"], 
                    st.session_state.results["production"]
                )
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = create_inventory_chart(st.session_state.results["inventory"])
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
        
        elif report_type == "Reporte de Inventario":
            # Generar reporte específico de inventario
            if st.button("Generar Reporte de Inventario (PDF)"):
                with st.spinner("Generando reporte..."):
                    pdf_data = download_inventory_report(st.session_state.data, st.session_state.results)
                    
                    b64 = base64.b64encode(pdf_data).decode()
                    href = f'<a href="data:application/pdf;base64,{b64}" download="reporte_inventario.pdf">Descargar Reporte PDF</a>'
                    st.markdown(href, unsafe_allow_html=True)
            
            # Vista previa del reporte
            st.markdown("<h3 class='sub-header'>Vista Previa del Reporte de Inventario</h3>", unsafe_allow_html=True)
            
            # Mostrar gráfico de inventario
            fig = create_inventory_chart(st.session_state.results["inventory"])
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            
            # Detalles de inventario
            inventory_details = []
            for product in st.session_state.results["inventory"]["inventory_levels"]:
                for warehouse, level in st.session_state.results["inventory"]["inventory_levels"][product].items():
                    storage_cost = st.session_state.data["warehouses"][warehouse]["storage_cost"]
                    inventory_details.append({
                        "Producto": product,
                        "Almacén": warehouse,
                        "Nivel de Inventario": level,
                        "Volumen Unitario": st.session_state.data["products"][product],
                        "Volumen Total": level * st.session_state.data["products"][product],
                        "Costo Unitario": f"${storage_cost:.2f}",
                        "Costo Total": f"${level * storage_cost:.2f}"
                    })
            
            if inventory_details:
                st.dataframe(pd.DataFrame(inventory_details))
        
        elif report_type == "Reporte de Transporte":
            # Generar reporte específico de transporte
            if st.button("Generar Reporte de Transporte (PDF)"):
                with st.spinner("Generando reporte..."):
                    pdf_data = download_transport_report(st.session_state.data, st.session_state.results)
                    
                    b64 = base64.b64encode(pdf_data).decode()
                    href = f'<a href="data:application/pdf;base64,{b64}" download="reporte_transporte.pdf">Descargar Reporte PDF</a>'
                    st.markdown(href, unsafe_allow_html=True)
            
            # Vista previa del reporte
            st.markdown("<h3 class='sub-header'>Vista Previa del Reporte de Transporte</h3>", unsafe_allow_html=True)
            
            # Mapa de rutas
            transport_map = plot_map(st.session_state.data, st.session_state.results["transport"])
            folium_static(transport_map)
            
            # Detalles de rutas
            route_details = []
            for warehouse in st.session_state.results["transport"]["routes"]:
                for retail, info in st.session_state.results["transport"]["routes"][warehouse].items():
                    if info["active"]:
                        route_details.append({
                            "Origen": warehouse,
                            "Destino": retail,
                            "Distancia (km)": haversine(
                                (st.session_state.data["warehouses"][warehouse]["latitude"], 
                                 st.session_state.data["warehouses"][warehouse]["longitude"]),
                                (st.session_state.data["retail_locations"][retail]["latitude"], 
                                 st.session_state.data["retail_locations"][retail]["longitude"])
                            ),
                            "Tiempo de Entrega": f"{st.session_state.data['lead_times'][warehouse][retail]} días",
                            "Costo": f"${info['cost']:.2f}",
                            "Volumen Transportado": f"{info['quantity']:.1f}"
                        })
            
            if route_details:
                st.dataframe(pd.DataFrame(route_details))
        
        elif report_type == "Reporte de Producción":
            # Generar reporte específico de producción
            if st.button("Generar Reporte de Producción (PDF)"):
                with st.spinner("Generando reporte..."):
                    pdf_data = download_production_report(st.session_state.data, st.session_state.results)
                    
                    b64 = base64.b64encode(pdf_data).decode()
                    href = f'<a href="data:application/pdf;base64,{b64}" download="reporte_produccion.pdf">Descargar Reporte PDF</a>'
                    st.markdown(href, unsafe_allow_html=True)
            
            # Vista previa del reporte
            st.markdown("<h3 class='sub-header'>Vista Previa del Reporte de Producción</h3>", unsafe_allow_html=True)
            
            # Mostrar gráfico de producción
            fig = create_production_chart(st.session_state.results["production"])
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            
            # Detalles de producción
            production_details = []
            for product, quantity in st.session_state.results["production"]["production"].items():
                unit_cost = st.session_state.data["production_costs"][product]
                production_details.append({
                    "Producto": product,
                    "Cantidad a Producir": quantity,
                    "Volumen Unitario": st.session_state.data["products"][product],
                    "Volumen Total": quantity * st.session_state.data["products"][product],
                    "Costo Unitario": f"${unit_cost:.2f}",
                    "Costo Total": f"${quantity * unit_cost:.2f}"
                })
            
            if production_details:
                st.dataframe(pd.DataFrame(production_details))

# Configuración
elif menu == "Configuración":
    st.markdown("<h2 class='sub-header'>Configuración de la Aplicación</h2>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class='info-box'>
    Personaliza la configuración de la aplicación según tus preferencias.
    </div>
    """, unsafe_allow_html=True)
    
    # Guardamos el tema actual
    current_theme = st.session_state.get("theme", "light")
    
    # Opciones de tema
    st.markdown("<h3 class='sub-header'>Tema de la Aplicación</h3>", unsafe_allow_html=True)
    theme = st.radio("Selecciona el tema", ["Claro", "Oscuro"], index=0 if current_theme == "light" else 1)
    
    if theme == "Claro":
        if current_theme != "light":
            st.session_state.theme = "light"
            st.success("Tema actualizado a claro. Por favor, recarga la página para aplicar los cambios.")
    else:
        if current_theme != "dark":
            st.session_state.theme = "dark"
            st.success("Tema actualizado a oscuro. Por favor, recarga la página para aplicar los cambios.")
    
    # Idioma
    st.markdown("<h3 class='sub-header'>Idioma</h3>", unsafe_allow_html=True)
    language = st.selectbox("Selecciona el idioma", ["Español", "English"], index=0)
    
    if language != "Español":
        st.warning("English language support coming soon. Currently only Spanish is supported.")
    
    # Configuración de visualización
    st.markdown("<h3 class='sub-header'>Visualización</h3>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    
    with col1:
        map_style = st.selectbox("Estilo de mapa", ["OpenStreetMap", "Stamen Terrain", "Stamen Toner"])
    
    with col2:
        chart_theme = st.selectbox("Tema de gráficas", ["plotly", "plotly_white", "plotly_dark", "ggplot2", "seaborn"])
    
    # Guardar configuración
    if st.button("Guardar Configuración"):
        st.session_state.map_style = map_style
        st.session_state.chart_theme = chart_theme
        st.success("Configuración guardada correctamente")
    
    # Reiniciar aplicación
    st.markdown("<h3 class='sub-header'>Reiniciar Aplicación</h3>", unsafe_allow_html=True)
    st.warning("Esta acción eliminará todos los datos y resultados actuales.")
    
    if st.button("Reiniciar Aplicación"):
        # Limpiamos la sesión pero mantenemos configuraciones
        theme = st.session_state.get("theme", "light")
        map_style = st.session_state.get("map_style", "OpenStreetMap")
        chart_theme = st.session_state.get("chart_theme", "plotly")
        
        st.session_state.clear()
        
        # Restauramos configuraciones
        st.session_state.theme = theme
        st.session_state.map_style = map_style
        st.session_state.chart_theme = chart_theme
        
        st.success("Aplicación reiniciada correctamente")

# CSS personalizado
st.markdown("""
<style>
    .sub-header {
        color: #1E88E5;
        padding-bottom: 10px;
        border-bottom: 1px solid #e0e0e0;
        margin-bottom: 20px;
    }
    
    .info-box {
        background-color: #e1f5fe;
        border-left: 4px solid #03a9f4;
        padding: 15px;
        margin-bottom: 20px;
        border-radius: 4px;
    }
    
    .success-box {
        background-color: #e8f5e9;
        border-left: 4px solid #4caf50;
        padding: 15px;
        margin-bottom: 20px;
        border-radius: 4px;
    }
    
    /* Estilo dark mode */
    .dark-mode .sub-header {
        color: #64b5f6;
        border-bottom: 1px solid #424242;
    }
    
    .dark-mode .info-box {
        background-color: #0d47a1;
        border-left: 4px solid #2196f3;
    }
    
    .dark-mode .success-box {
        background-color: #1b5e20;
        border-left: 4px solid #4caf50;
    }
    
    /* Aplica el tema oscuro si está activado */
    body.dark {
        color: white;
        background-color: #121212;
    }
    
    body.dark .stButton>button {
        color: white;
        background-color: #2979ff;
        border: none;
    }
    
    body.dark .stTextInput>div>div>input {
        color: white;
        background-color: #333;
    }
</style>

<script>
    if (localStorage.getItem('theme') === 'dark') {
        document.body.classList.add('dark');
        document.querySelectorAll('.stApp').forEach(el => {
            el.classList.add('dark-mode');
        });
    }
</script>
""", unsafe_allow_html=True)