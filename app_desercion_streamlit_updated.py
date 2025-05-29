import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import os

# Configuración de la página
st.set_page_config(
    page_title="Sistema de Alerta Temprana - Deserción Estudiantil",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .university-header {
        display: flex;
        align-items: center;
        justify-content: center;
        margin-bottom: 2rem;
        padding: 1rem;
        background-color: #f8f9fa;
        border-radius: 10px;
        border: 2px solid #e9ecef;
    }
    .university-logo {
        margin-right: 2rem;
    }
    .university-info {
        text-align: left;
    }
    .author-credits {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2196f3;
        margin-top: 1rem;
        text-align: center;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .risk-critical {
        background-color: #ffebee;
        border-left: 4px solid #f44336;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .risk-high {
        background-color: #fff3e0;
        border-left: 4px solid #ff9800;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .risk-medium {
        background-color: #fffde7;
        border-left: 4px solid #ffeb3b;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .risk-low {
        background-color: #e8f5e8;
        border-left: 4px solid #4caf50;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .footer-credits {
        margin-top: 3rem;
        padding: 2rem;
        background-color: #f8f9fa;
        border-radius: 10px;
        text-align: center;
        border-top: 3px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

# Función para cargar modelos
@st.cache_resource
def cargar_modelos():
    """Carga todos los modelos y metadatos guardados"""
    try:
        # Verificar que existen los archivos
        archivos_necesarios = [
            'modelo_xgboost_desercion.pkl',
            'modelo_randomforest_desercion.pkl',
            'umbrales_optimos_desercion.pkl',
            'label_encoders_desercion.pkl',
            'feature_names_desercion.pkl',
            'metadatos_desercion.pkl'
        ]
        
        archivos_faltantes = [archivo for archivo in archivos_necesarios if not os.path.exists(archivo)]
        
        if archivos_faltantes:
            st.error(f"Archivos faltantes: {', '.join(archivos_faltantes)}")
            return None
        
        # Cargar modelos y metadatos
        modelo_xgb = joblib.load('modelo_xgboost_desercion.pkl')
        modelo_rf = joblib.load('modelo_randomforest_desercion.pkl')
        umbrales = joblib.load('umbrales_optimos_desercion.pkl')
        encoders = joblib.load('label_encoders_desercion.pkl')
        feature_names = joblib.load('feature_names_desercion.pkl')
        metadatos = joblib.load('metadatos_desercion.pkl')
        
        # Cargar scaler si existe
        scaler = None
        if os.path.exists('scaler_desercion.pkl'):
            scaler = joblib.load('scaler_desercion.pkl')
        
        return {
            'xgboost': modelo_xgb,
            'randomforest': modelo_rf,
            'umbrales': umbrales,
            'encoders': encoders,
            'feature_names': feature_names,
            'metadatos': metadatos,
            'scaler': scaler
        }
        
    except Exception as e:
        st.error(f"Error al cargar modelos: {e}")
        return None

# Función para codificar variables categóricas
def codificar_variables_categoricas(datos_estudiante, encoders):
    """Codifica las variables categóricas usando los encoders guardados"""
    
    datos_codificados = datos_estudiante.copy()
    
    # Mapeo de variables categóricas
    mapeos = {
        'FACULTAD': {
            'Ingeniería': 'INGENIERIA',
            'Medicina': 'MEDICINA', 
            'Derecho': 'DERECHO',
            'Administración': 'ADMINISTRACION',
            'Psicología': 'PSICOLOGIA',
            'Educación': 'EDUCACION',
            'Ciencias': 'CIENCIAS'
        },
        'SEXO': {'Masculino': 'M', 'Femenino': 'F'},
        'TIPO DEL COLEGIO': {'Público': 'PUBLICO', 'Privado': 'PRIVADO'},
        'ALMUERZOS ': {'Sí': 'SI', 'No': 'NO'},
        'REFRIGERIO': {'Sí': 'SI', 'No': 'NO'}
    }
    
    # Aplicar encoders
    for variable, encoder in encoders.items():
        if variable in datos_estudiante:
            valor_original = datos_estudiante[variable]
            
            # Aplicar mapeo si existe
            if variable in mapeos and valor_original in mapeos[variable]:
                valor_mapeado = mapeos[variable][valor_original]
            else:
                valor_mapeado = valor_original
            
            try:
                # Intentar transformar con el encoder
                valor_codificado = encoder.transform([str(valor_mapeado)])[0]
                datos_codificados[f'{variable}_encoded'] = valor_codificado
            except ValueError:
                # Si el valor no existe en el encoder, usar el más común (0)
                datos_codificados[f'{variable}_encoded'] = 0
                st.warning(f"Valor '{valor_original}' no reconocido para {variable}. Usando valor por defecto.")
    
    return datos_codificados

# Función para hacer predicción
def predecir_desercion(datos_estudiante, modelos_cargados, modelo_seleccionado):
    """Realiza predicción de deserción"""
    
    try:
        # Seleccionar modelo
        if modelo_seleccionado == "XGBoost":
            modelo = modelos_cargados['xgboost']
            umbral = modelos_cargados['umbrales']['xgboost']
        else:
            modelo = modelos_cargados['randomforest']
            umbral = modelos_cargados['umbrales']['randomforest']
        
        # Codificar variables categóricas
        datos_codificados = codificar_variables_categoricas(datos_estudiante, modelos_cargados['encoders'])
        
        # Preparar features en el orden correcto
        feature_names = modelos_cargados['feature_names']
        
        # Crear DataFrame con todas las features
        X_pred = pd.DataFrame([datos_codificados])
        
        # Asegurar que tenemos todas las features necesarias
        for feature in feature_names:
            if feature not in X_pred.columns:
                X_pred[feature] = 0  # Valor por defecto
        
        # Seleccionar solo las features del modelo
        X_pred = X_pred[feature_names]
        
        # Aplicar scaler si existe
        if modelos_cargados['scaler'] is not None:
            metadatos = modelos_cargados['metadatos']
            if 'features_numericas' in metadatos:
                features_numericas = metadatos['features_numericas']
                features_existentes = [f for f in features_numericas if f in X_pred.columns]
                if features_existentes:
                    X_pred[features_existentes] = modelos_cargados['scaler'].transform(X_pred[features_existentes])
        
        # Realizar predicción
        probabilidad = modelo.predict_proba(X_pred)[:, 1][0]
        prediccion = 1 if probabilidad >= umbral else 0
        
        # Categorizar riesgo
        if probabilidad >= 0.7:
            categoria = 'CRÍTICO'
            color = '#f44336'
            emoji = '🔴'
            accion = 'INTERVENCIÓN INMEDIATA'
        elif probabilidad >= 0.5:
            categoria = 'ALTO'
            color = '#ff9800'
            emoji = '🟠'
            accion = 'SEGUIMIENTO INTENSIVO'
        elif probabilidad >= umbral:
            categoria = 'MEDIO'
            color = '#ffeb3b'
            emoji = '🟡'
            accion = 'MONITOREO CERCANO'
        else:
            categoria = 'BAJO'
            color = '#4caf50'
            emoji = '🟢'
            accion = 'SEGUIMIENTO REGULAR'
        
        return {
            'probabilidad': probabilidad,
            'prediccion': prediccion,
            'categoria': categoria,
            'color': color,
            'emoji': emoji,
            'accion': accion,
            'umbral': umbral,
            'modelo_usado': modelo_seleccionado
        }
        
    except Exception as e:
        st.error(f"Error en predicción: {e}")
        return None

# Función para crear gráfico de gauge
def crear_gauge_riesgo(probabilidad, umbral, categoria, color):
    """Crea gráfico de gauge para mostrar nivel de riesgo"""
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = probabilidad,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Probabilidad de Deserción", 'font': {'size': 20}},
        delta = {'reference': umbral, 'increasing': {'color': "red"}, 'decreasing': {'color': "green"}},
        gauge = {
            'axis': {'range': [0, 1], 'tickformat': '.0%'},
            'bar': {'color': color, 'thickness': 0.3},
            'steps': [
                {'range': [0, umbral], 'color': "lightgreen"},
                {'range': [umbral, 0.5], 'color': "yellow"},
                {'range': [0.5, 0.7], 'color': "orange"},
                {'range': [0.7, 1], 'color': "red"}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': umbral
            }
        }
    ))
    
    fig.update_layout(
        height=400,
        font={'color': "darkblue", 'family': "Arial"},
        paper_bgcolor="white"
    )
    
    return fig

# APLICACIÓN PRINCIPAL
def main():
    # Header con logo de la universidad
    st.markdown("""
    <div class="university-header">
        <div class="university-logo">
            <img src="https://hebbkx1anhila5yf.public.blob.vercel-storage.com/unimag-Aun-300x141-xQs7GlGRdQWcUBVZfOdh9z9xUJUFQ6.png" width="200" alt="Universidad del Magdalena">
        </div>
        <div class="university-info">
            <h2 style="color: #1f77b4; margin: 0;">Universidad del Magdalena</h2>
            <p style="margin: 5px 0; color: #666;">AÚN+ incluyente e innovadora</p>
            <p style="margin: 0; color: #666; font-weight: bold;">ACREDITADA POR ALTA CALIDAD</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Título principal
    st.markdown('<h1 class="main-header">🎓 Sistema de Alerta Temprana de Deserción Estudiantil</h1>', unsafe_allow_html=True)
    
    # Créditos del autor
    st.markdown("""
    <div class="author-credits">
        <h4 style="margin: 0; color: #1976d2;">👨‍💻 Desarrollado por</h4>
        <p style="margin: 5px 0; font-size: 1.1em; font-weight: bold;">Yeison De La Torre</p>
        <p style="margin: 0; color: #666; font-style: italic;">Con fines académicos - Universidad del Magdalena</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Cargar modelos
    modelos_cargados = cargar_modelos()
    
    if modelos_cargados is None:
        st.error("No se pudieron cargar los modelos. Verifica que los archivos .pkl estén en el directorio.")
        st.stop()
    
    # Sidebar con información del modelo
    with st.sidebar:
        # Logo pequeño en sidebar
        st.image("https://hebbkx1anhila5yf.public.blob.vercel-storage.com/unimag-Aun-300x141-xQs7GlGRdQWcUBVZfOdh9z9xUJUFQ6.png", width=150)
        
        st.header("📊 Información del Modelo")
        
        metadatos = modelos_cargados['metadatos']
        
        st.metric("Modelo Recomendado", metadatos['modelo_recomendado'])
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("AUC XGBoost", f"{metadatos['auc_xgboost']:.3f}")
            st.metric("Recall XGBoost", f"{metadatos['recall_xgboost']:.3f}")
        
        with col2:
            st.metric("AUC Random Forest", f"{metadatos['auc_randomforest']:.3f}")
            st.metric("Recall Random Forest", f"{metadatos['recall_randomforest']:.3f}")
        
        st.metric("Total Features", metadatos['total_features'])
        
        st.markdown("---")
        st.markdown("**Estados de Deserción:**")
        for estado in metadatos['estados_desercion']:
            st.markdown(f"• {estado}")
        
        st.markdown("**Estados Activos:**")
        for estado in metadatos['estados_activos']:
            st.markdown(f"• {estado}")
        
        # Información del desarrollador en sidebar
        st.markdown("---")
        st.markdown("**👨‍💻 Desarrollador:**")
        st.markdown("Yeison De La Torre")
        st.markdown("*Universidad del Magdalena*")
    
    # Selección de modelo
    st.subheader("🔧 Configuración del Modelo")
    modelo_seleccionado = st.radio(
        "Seleccione el modelo a utilizar:",
        ["XGBoost", "Random Forest"],
        index=0 if metadatos['modelo_recomendado'] == "XGBoost" else 1,
        horizontal=True
    )
    
    umbral_actual = modelos_cargados['umbrales']['xgboost'] if modelo_seleccionado == "XGBoost" else modelos_cargados['umbrales']['randomforest']
    st.info(f"Umbral óptimo para {modelo_seleccionado}: {umbral_actual:.3f}")
    
    st.markdown("---")
    
    # Formulario de entrada de datos
    st.subheader("📝 Datos del Estudiante")
    
    with st.form("formulario_estudiante"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**📚 Información Académica**")
            promedio_acumulado = st.number_input(
                "Promedio Acumulado", 
                min_value=1.0, 
                max_value=5.0, 
                value=3.5, 
                step=0.1,
                help="Promedio acumulado del estudiante (1.0 - 5.0)"
            )
            
            promedio_semestre = st.number_input(
                "Promedio del Semestre", 
                min_value=1.0, 
                max_value=5.0, 
                value=3.5, 
                step=0.1,
                help="Promedio del semestre actual"
            )
            
            creditos_aprobados = st.number_input(
                "Créditos Aprobados", 
                min_value=0, 
                max_value=200, 
                value=60,
                help="Total de créditos aprobados"
            )
            
            puntaje_icfes = st.number_input(
                "Puntaje ICFES", 
                min_value=100, 
                max_value=500, 
                value=250,
                help="Puntaje en las pruebas ICFES"
            )
        
        with col2:
            st.markdown("**👤 Información Personal**")
            facultad = st.selectbox(
                "Facultad", 
                options=["Ingeniería", "Medicina", "Derecho", "Administración", "Psicología", "Educación", "Ciencias"],
                help="Facultad a la que pertenece el estudiante"
            )
            
            sexo = st.selectbox(
                "Sexo", 
                options=["Masculino", "Femenino"],
                help="Sexo del estudiante"
            )
            
            estrato = st.selectbox(
                "Estrato Socioeconómico", 
                options=[1, 2, 3, 4, 5, 6],
                index=2,
                help="Estrato socioeconómico del estudiante"
            )
            
            mpio_residencia = st.text_input(
                "Municipio de Residencia", 
                value="Bogotá",
                help="Municipio donde reside el estudiante"
            )
        
        with col3:
            st.markdown("**🏫 Información Educativa**")
            tipo_colegio = st.selectbox(
                "Tipo de Colegio", 
                options=["Público", "Privado"],
                help="Tipo de colegio de procedencia"
            )
            
            nivel_edu_madre = st.selectbox(
                "Nivel Educativo de la Madre", 
                options=["Primaria", "Secundaria", "Técnico", "Universitario", "Posgrado"],
                index=2,
                help="Máximo nivel educativo alcanzado por la madre"
            )
            
            almuerzos = st.selectbox(
                "Recibe Almuerzos", 
                options=["Sí", "No"],
                help="¿El estudiante recibe subsidio de almuerzos?"
            )
            
            refrigerio = st.selectbox(
                "Recibe Refrigerio", 
                options=["Sí", "No"],
                help="¿El estudiante recibe subsidio de refrigerio?"
            )
        
        # Información temporal
        st.markdown("**📅 Información Temporal**")
        col_temp1, col_temp2 = st.columns(2)
        
        with col_temp1:
            periodo_year = st.number_input(
                "Año del Período", 
                min_value=2014, 
                max_value=2030, 
                value=2024,
                help="Año del período académico"
            )
        
        with col_temp2:
            periodo_sem = st.selectbox(
                "Semestre", 
                options=[1, 2],
                help="Semestre del período académico"
            )
        
        # Calcular PERIODO_SEQ (basado en tu código)
        periodo_seq = (periodo_year - 2014) * 2 + periodo_sem - 1
        
        st.info(f"Período secuencial calculado: {periodo_seq}")
        
        # Botón de predicción
        submitted = st.form_submit_button(
            "🔮 Predecir Riesgo de Deserción", 
            type="primary",
            use_container_width=True
        )
    
    # Procesar predicción
    if submitted:
        # Preparar datos del estudiante
        datos_estudiante = {
            'PROMEDIO ACUMULADO': promedio_acumulado,
            'ESTRATO': estrato,
            'creditos aprobados': creditos_aprobados,
            'PUNTAJE ICFES': puntaje_icfes,
            'promedio al semestre': promedio_semestre,
            'PERIODO_SEQ': periodo_seq,
            'FACULTAD': facultad,
            'SEXO': sexo,
            'MPIO RESIDENCIA': mpio_residencia,
            'TIPO DEL COLEGIO': tipo_colegio,
            'NIVEL EDU DE LA MADRE': nivel_edu_madre,
            'ALMUERZOS ': almuerzos,
            'REFRIGERIO': refrigerio
        }
        
        # Realizar predicción
        resultado = predecir_desercion(datos_estudiante, modelos_cargados, modelo_seleccionado)
        
        if resultado:
            st.markdown("---")
            st.subheader("📊 Resultado de la Predicción")
            
            # Mostrar resultado principal
            col_res1, col_res2, col_res3 = st.columns([1, 1, 2])
            
            with col_res1:
                st.metric(
                    "Probabilidad de Deserción", 
                    f"{resultado['probabilidad']:.1%}",
                    delta=f"{(resultado['probabilidad'] - resultado['umbral']):.1%}"
                )
            
            with col_res2:
                st.metric(
                    "Predicción", 
                    "DESERTOR" if resultado['prediccion'] == 1 else "ACTIVO"
                )
            
            with col_res3:
                # Crear tarjeta de riesgo con estilo
                categoria = resultado['categoria']
                emoji = resultado['emoji']
                accion = resultado['accion']
                
                if categoria == 'CRÍTICO':
                    st.markdown(f"""
                    <div class="risk-critical">
                        <h3>{emoji} Riesgo {categoria}</h3>
                        <p><strong>Acción:</strong> {accion}</p>
                    </div>
                    """, unsafe_allow_html=True)
                elif categoria == 'ALTO':
                    st.markdown(f"""
                    <div class="risk-high">
                        <h3>{emoji} Riesgo {categoria}</h3>
                        <p><strong>Acción:</strong> {accion}</p>
                    </div>
                    """, unsafe_allow_html=True)
                elif categoria == 'MEDIO':
                    st.markdown(f"""
                    <div class="risk-medium">
                        <h3>{emoji} Riesgo {categoria}</h3>
                        <p><strong>Acción:</strong> {accion}</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="risk-low">
                        <h3>{emoji} Riesgo {categoria}</h3>
                        <p><strong>Acción:</strong> {accion}</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Gráfico de gauge
            st.subheader("📈 Visualización del Riesgo")
            fig_gauge = crear_gauge_riesgo(
                resultado['probabilidad'], 
                resultado['umbral'], 
                resultado['categoria'], 
                resultado['color']
            )
            st.plotly_chart(fig_gauge, use_container_width=True)
            
            # Información adicional
            with st.expander("ℹ️ Información Adicional"):
                col_info1, col_info2 = st.columns(2)
                
                with col_info1:
                    st.write("**Detalles de la Predicción:**")
                    st.write(f"• Modelo utilizado: {resultado['modelo_usado']}")
                    st.write(f"• Umbral de decisión: {resultado['umbral']:.3f}")
                    st.write(f"• Probabilidad calculada: {resultado['probabilidad']:.4f}")
                    st.write(f"• Categoría de riesgo: {resultado['categoria']}")
                
                with col_info2:
                    st.write("**Recomendaciones por Categoría:**")
                    st.write("🔴 **CRÍTICO**: Contacto inmediato, plan de retención urgente")
                    st.write("🟠 **ALTO**: Seguimiento semanal, apoyo académico")
                    st.write("🟡 **MEDIO**: Monitoreo quincenal, recursos adicionales")
                    st.write("🟢 **BAJO**: Seguimiento regular, mantener motivación")
            
            # Guardar resultado en session state para histórico
            if 'historico_predicciones' not in st.session_state:
                st.session_state.historico_predicciones = []
            
            prediccion_actual = {
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'facultad': facultad,
                'promedio': promedio_acumulado,
                'probabilidad': resultado['probabilidad'],
                'categoria': resultado['categoria'],
                'modelo': resultado['modelo_usado']
            }
            
            st.session_state.historico_predicciones.append(prediccion_actual)
    
    # Mostrar histórico si existe
    if 'historico_predicciones' in st.session_state and st.session_state.historico_predicciones:
        st.markdown("---")
        st.subheader("📋 Histórico de Predicciones")
        
        df_historico = pd.DataFrame(st.session_state.historico_predicciones)
        
        # Mostrar tabla
        st.dataframe(
            df_historico.tail(10),  # Últimas 10 predicciones
            use_container_width=True
        )
        
        # Botón para limpiar histórico
        if st.button("🗑️ Limpiar Histórico"):
            st.session_state.historico_predicciones = []
            st.rerun()
    
    # Footer con créditos completos
    st.markdown("""
    <div class="footer-credits">
        <h3 style="color: #1976d2; margin-bottom: 1rem;">🎓 Sistema de Alerta Temprana de Deserción Estudiantil</h3>
        <div style="display: flex; justify-content: center; align-items: center; margin-bottom: 1rem;">
            <img src="https://hebbkx1anhila5yf.public.blob.vercel-storage.com/unimag-Aun-300x141-xQs7GlGRdQWcUBVZfOdh9z9xUJUFQ6.png" width="120" style="margin-right: 1rem;">
            <div style="text-align: left;">
                <p style="margin: 0; font-weight: bold; color: #1976d2;">Universidad del Magdalena</p>
                <p style="margin: 0; color: #666;">AÚN+ incluyente e innovadora</p>
                <p style="margin: 0; color: #666; font-size: 0.9em;">ACREDITADA POR ALTA CALIDAD</p>
            </div>
        </div>
        <hr style="border: 1px solid #e0e0e0; margin: 1rem 0;">
        <p style="margin: 0.5rem 0; font-weight: bold; color: #1976d2;">👨‍💻 Desarrollado por: Yeison De La Torre</p>
        <p style="margin: 0; color: #666; font-style: italic;">Proyecto desarrollado con fines académicos</p>
        <p style="margin: 0.5rem 0; color: #666; font-size: 0.9em;">
            Utilizando Machine Learning para la predicción temprana de deserción estudiantil
        </p>
        <p style="margin: 0; color: #999; font-size: 0.8em;">
            © 2024 - Universidad del Magdalena - Todos los derechos reservados
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
