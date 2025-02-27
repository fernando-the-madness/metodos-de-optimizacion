# Inside the "Simular Rendimientos" button handler:
if st.button("Simular Rendimientos"):
    with st.spinner('Ejecutando simulación Monte Carlo...'):
        # Parámetros de la simulación
        dias_por_anio = 252
        total_dias = dias_por_anio * horizonte
        
        # Definir frecuencia de rebalanceo (mensual por defecto)
        dias_por_mes = 21
        periodo_rebalanceo = dias_por_mes  # Rebalanceo mensual
        
        # Media y covarianza de los rendimientos diarios (usando rendimientos logarítmicos)
        media_diaria = rendimientos.mean()
        cov_diaria = rendimientos.cov()
        
        # Calcular rendimiento esperado y volatilidad de la cartera
        rendimiento_cartera_anual, volatilidad_cartera_anual = calcular_estadisticas_cartera(pesos_optimos, rendimientos)
        rendimiento_cartera_diario = rendimiento_cartera_anual / dias_por_anio
        volatilidad_cartera_diaria = volatilidad_cartera_anual / np.sqrt(dias_por_anio)
        
        # Inicializar la matriz de resultados
        resultados_simulacion = np.zeros((num_simulaciones, total_dias + 1))
        resultados_simulacion[:, 0] = capital_inicial
        
        # Crear fechas futuras para el índice
        fechas_futuras = pd.date_range(start=dt.date.today(), periods=total_dias + 1, freq='B')
        
        # Ejecutar simulaciones más eficientemente
        np.random.seed(42)
        
        # Para cada simulación
        for sim in range(num_simulaciones):
            capital_actual = capital_inicial
            pesos_actuales = pesos_optimos.copy()
            
            # Para cada día de trading
            for t in range(1, total_dias + 1):
                # Generar rendimientos aleatorios correlacionados
                Z = np.random.multivariate_normal(media_diaria, cov_diaria, 1)[0]
                
                # Aplicar rendimientos a cada activo en la cartera
                valores_activos = capital_actual * pesos_actuales
                nuevos_valores = valores_activos * np.exp(Z)  # Usar exponencial para convertir log-returns
                capital_actual = np.sum(nuevos_valores)
                
                # Actualizar pesos después de aplicar rendimientos (drift)
                if len(pesos_actuales) > 1:  # Solo si hay más de un activo
                    pesos_actuales = nuevos_valores / capital_actual
                
                # Rebalancear periódicamente a los pesos objetivo
                if t % periodo_rebalanceo == 0:
                    pesos_actuales = pesos_optimos.copy()
                
                # Guardar el capital actual
                resultados_simulacion[sim, t] = capital_actual
        
        # Convertir a DataFrame para visualización
        df_resultados = pd.DataFrame(resultados_simulacion, columns=fechas_futuras)
        
        # Calcular estadísticas
        df_final = df_resultados[fechas_futuras[-1]]
        percentiles = np.percentile(df_final, [5, 25, 50, 75, 95])
        capital_final_medio = df_final.mean()
        capital_final_mediano = percentiles[2]
        
        # Calcular métricas de riesgo adicionales
        # Value at Risk (VaR) al 95% de confianza
        var_95 = capital_inicial - np.percentile(df_final, 5)
        var_95_pct = (var_95 / capital_inicial) * 100
        
        # Cálculo de drawdowns para cada simulación
        max_drawdowns = []
        for sim in range(num_simulaciones):
            # Obtener la serie de capital para esta simulación
            capital_serie = df_resultados.iloc[sim]
            # Calcular el máximo acumulado
            running_max = np.maximum.accumulate(capital_serie)
            # Calcular drawdowns como porcentaje
            drawdowns = (running_max - capital_serie) / running_max * 100
            # Guardar el máximo drawdown
            max_drawdowns.append(drawdowns.max())
        
        max_drawdown_medio = np.mean(max_drawdowns)
        max_drawdown_extremo = np.percentile(max_drawdowns, 95)
        
        # Gráfico de simulación mejorado
        fig_sim = go.Figure()
        
        # Añadir área para el rango entre percentiles 25 y 75
        fig_sim.add_trace(go.Scatter(
            x=fechas_futuras.tolist() + fechas_futuras.tolist()[::-1],
            y=np.percentile(df_resultados, 75, axis=0).tolist() + np.percentile(df_resultados, 25, axis=0).tolist()[::-1],
            fill='toself',
            fillcolor='rgba(0, 100, 80, 0.2)',
            line=dict(color='rgba(255, 255, 255, 0)'),
            name='Rango 25-75%'
        ))
        
        # Añadir área para el rango entre percentiles 5 y 95
        fig_sim.add_trace(go.Scatter(
            x=fechas_futuras.tolist() + fechas_futuras.tolist()[::-1],
            y=np.percentile(df_resultados, 95, axis=0).tolist() + np.percentile(df_resultados, 5, axis=0).tolist()[::-1],
            fill='toself',
            fillcolor='rgba(0, 100, 80, 0.1)',
            line=dict(color='rgba(255, 255, 255, 0)'),
            name='Rango 5-95%'
        ))
        
        # Añadir algunas simulaciones individuales para referencia
        for i in range(min(10, num_simulaciones)):  # Mostrar solo 10 trayectorias
            fig_sim.add_trace(go.Scatter(
                x=fechas_futuras,
                y=df_resultados.iloc[i],
                mode='lines',
                line=dict(width=1, color='rgba(100, 100, 200, 0.3)'),
                showlegend=False
            ))
        
        # Añadir la mediana como línea principal
        fig_sim.add_trace(go.Scatter(
            x=fechas_futuras,
            y=np.percentile(df_resultados, 50, axis=0),
            mode='lines',
            line=dict(width=3, color='blue'),
            name='Mediana'
        ))
        
        # Líneas de referencia para percentiles clave
        fig_sim.add_trace(go.Scatter(
            x=fechas_futuras,
            y=np.percentile(df_resultados, 75, axis=0),
            mode='lines',
            line=dict(width=2, color='green'),
            name='Percentil 75'
        ))
        
        fig_sim.add_trace(go.Scatter(
            x=fechas_futuras,
            y=np.percentile(df_resultados, 25, axis=0),
            mode='lines',
            line=dict(width=2, color='orange'),
            name='Percentil 25'
        ))
        
        fig_sim.add_trace(go.Scatter(
            x=fechas_futuras,
            y=np.percentile(df_resultados, 5, axis=0),
            mode='lines',
            line=dict(width=1.5, color='red', dash='dot'),
            name='Percentil 5'
        ))
        
        fig_sim.add_trace(go.Scatter(
            x=fechas_futuras,
            y=np.percentile(df_resultados, 95, axis=0),
            mode='lines',
            line=dict(width=1.5, color='darkgreen', dash='dot'),
            name='Percentil 95'
        ))
        
        # Línea de capital inicial
        fig_sim.add_trace(go.Scatter(
            x=fechas_futuras,
            y=[capital_inicial] * len(fechas_futuras),
            mode='lines',
            line=dict(width=1.5, color='rgba(255, 0, 0, 0.5)', dash='dash'),
            name='Capital Inicial'
        ))
        
        # Configuración del gráfico
        fig_sim.update_layout(
            title=f'Simulación Monte Carlo: Evolución de Capital (${capital_inicial:,} por {horizonte} años)',
            xaxis_title='Fecha',
            yaxis_title='Capital ($)',
            height=600,
            template='plotly_dark',
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # Añadir anotación sobre rebalanceo
        fig_sim.add_annotation(
            x=fechas_futuras[len(fechas_futuras)//2],
            y=capital_inicial * 0.4,
            text=f"Rebalanceo mensual a pesos óptimos",
            showarrow=False,
            font=dict(size=12, color="white"),
            bgcolor="rgba(50, 50, 50, 0.7)",
            bordercolor="white",
            borderwidth=1,
            borderpad=4
        )
        
        st.plotly_chart(fig_sim, use_container_width=True)
        
        # Mostrar estadísticas de la simulación
        st.subheader("Resultados de la Simulación")
        
        # Primera fila: Estadísticas básicas
        col_sim1, col_sim2, col_sim3 = st.columns(3)
        
        with col_sim1:
            st.metric("Capital Inicial", f"${capital_inicial:,}")
            st.metric("Capital Final (Mediana)", f"${capital_final_mediano:,.2f}")
        
        with col_sim2:
            rendimiento_total = (capital_final_mediano/capital_inicial - 1) * 100
            rendimiento_anual = ((capital_final_mediano/capital_inicial)**(1/horizonte) - 1) * 100
            st.metric("Rendimiento Total (Mediano)", f"{rendimiento_total:.2f}%")
            st.metric("Rendimiento Anual (Mediano)", f"{rendimiento_anual:.2f}%")
        
        with col_sim3:
            st.metric("Mejor Escenario (P95)", f"${percentiles[4]:,.2f}")
            st.metric("Peor Escenario (P5)", f"${percentiles[0]:,.2f}")
        
        # Segunda fila: Métricas de riesgo
        st.subheader("Métricas de Riesgo")
        col_risk1, col_risk2, col_risk3 = st.columns(3)
        
        with col_risk1:
            st.metric("Value at Risk (95%)", f"${var_95:,.2f}")
            st.metric("VaR como % del capital", f"{var_95_pct:.2f}%")
        
        with col_risk2:
            st.metric("Drawdown Máximo Medio", f"{max_drawdown_medio:.2f}%")
            st.metric("Drawdown Extremo (P95)", f"{max_drawdown_extremo:.2f}%")
        
        with col_risk3:
            prob_positivo = (df_final > capital_inicial).mean() * 100
            prob_doble = (df_final > capital_inicial * 2).mean() * 100
            st.metric("Prob. Rendimiento Positivo", f"{prob_positivo:.2f}%")
            st.metric("Prob. Duplicar Capital", f"{prob_doble:.2f}%")
        
        # Gráficos adicionales
        
        # 1. Histograma de resultados finales mejorado
        fig_hist = px.histogram(
            df_final,
            nbins=50,
            title="Distribución de Capital Final",
            labels={'value': 'Capital Final ($)', 'count': 'Frecuencia'},
            color_discrete_sequence=['green'],
            marginal="box"  # Añadir boxplot en el margen
        )
        
        fig_hist.add_vline(x=capital_inicial, line_dash="dash", line_color="red", 
                          annotation_text="Capital Inicial")
        fig_hist.add_vline(x=capital_final_mediano, line_dash="solid", line_color="blue", 
                          annotation_text="Mediana")
        
        fig_hist.update_layout(
            template='plotly_dark',
            height=400
        )
        
        st.plotly_chart(fig_hist, use_container_width=True)
        
        # 2. Gráfico de distribución de drawdowns
        fig_dd = px.histogram(
            max_drawdowns,
            nbins=40,
            title="Distribución de Drawdowns Máximos",
            labels={'value': 'Drawdown Máximo (%)', 'count': 'Frecuencia'},
            color_discrete_sequence=['orange']
        )
        
        fig_dd.add_vline(x=max_drawdown_medio, line_dash="solid", line_color="red", 
                        annotation_text="Drawdown Medio")
        
        fig_dd.update_layout(
            template='plotly_dark',
            height=350
        )
        
        # 3. Tabla de probabilidades mejorada
        st.subheader("Probabilidades de Escenarios")
        
        escenarios = [
            "Rendimiento Positivo",
            "Rendimiento > Inflación (3%)",
            "Rendimiento > Bonos (5%)",
            "25% de Ganancia",
            "50% de Ganancia",
            "Duplicar Capital",
            "Triplicar Capital",
            "Pérdida > 10%",
            "Pérdida > 25%"
        ]
        
        # Suponiendo una inflación del 3% anual y rendimiento de bonos del 5% anual
        inflacion_acum = (1 + 0.03) ** horizonte - 1
        bonos_acum = (1 + 0.05) ** horizonte - 1
        
        probabilidades = [
            (df_final > capital_inicial).mean() * 100,  # Positivo
            (df_final > capital_inicial * (1 + inflacion_acum)).mean() * 100,  # > Inflación
            (df_final > capital_inicial * (1 + bonos_acum)).mean() * 100,  # > Bonos
            (df_final > capital_inicial * 1.25).mean() * 100,  # +25%
            (df_final > capital_inicial * 1.5).mean() * 100,  # +50%
            (df_final > capital_inicial * 2).mean() * 100,  # x2
            (df_final > capital_inicial * 3).mean() * 100,  # x3
            (df_final < capital_inicial * 0.9).mean() * 100,  # -10%
            (df_final < capital_inicial * 0.75).mean() * 100  # -25%
        ]
        
        df_prob = pd.DataFrame({
            'Escenario': escenarios,
            'Probabilidad (%)': probabilidades
        })
        
        col_prob1, col_prob2 = st.columns([3, 2])
        
        with col_prob1:
            # Graficar probabilidades
            fig_prob = px.bar(
                df_prob,
                x='Escenario',
                y='Probabilidad (%)',
                title="Probabilidades de Diferentes Escenarios",
                color='Probabilidad (%)',
                color_continuous_scale='RdYlGn'
            )
            
            fig_prob.update_layout(
                template='plotly_dark',
                height=350,
                xaxis={'categoryorder':'total descending'}
            )
            
            st.plotly_chart(fig_prob, use_container_width=True)
            
        with col_prob2:
            st.dataframe(df_prob.set_index('Escenario').style.format({
                'Probabilidad (%)': '{:.2f}'
            }), height=350)
        
        # Añadir notas explicativas
        st.markdown("""
        ### Notas sobre la Simulación
        
        - **Metodología**: Se utilizó un modelo de Movimiento Browniano Geométrico multivariado con correlaciones para simular los rendimientos diarios.
        - **Rebalanceo**: La cartera se rebalancea mensualmente para mantener los pesos óptimos.
        - **Value at Risk (VaR)**: Representa la pérdida máxima esperada con un 95% de confianza.
        - **Drawdown**: Representa la caída desde un máximo previo, medida clave del riesgo de la inversión.
        
        **Recuerde**: Las simulaciones son estimaciones basadas en datos históricos y supuestos estadísticos, no predicciones precisas del futuro.
        """)
