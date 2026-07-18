import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import warnings
import logging
import os
import bcrypt
from dotenv import load_dotenv
import requests
warnings.filterwarnings("ignore", category=FutureWarning)
logging.getLogger("balance").setLevel(logging.ERROR)
try:
    os.environ["API_URL"] = st.secrets["API_URL"]
    os.environ["DATABASE_URL"] = st.secrets["DATABASE_URL"]
    os.environ["API_KEY"] = st.secrets["API_KEY"]
    os.environ["USUARIO"] = st.secrets["USUARIO"]
    os.environ["CONTRASENA_HASH"] = st.secrets["CONTRASENA_HASH"]
except Exception:
    load_dotenv()
from carga import cargar_datos
from tracking import calcular_tracking
from limpieza import limpiar, normalizar
from imputacion import imputar
from ventanas import crear_ventanas
from ponderacion import obtener_targets_desde_censo, ponderar, POBLACIONES, generar_reporte
from estadistica import calcular_intervalos, test_hipotesis
from base_datos import registrar_metricas, get_db, CorridaDB, MetricaDB

# ==========================================
# 0. CONFIGURACIÓN DE PÁGINA
# ==========================================

st.set_page_config(page_title="Sistema de Ponderación Muestral", page_icon="📊", layout="wide")

# ==========================================
# 1. LOGIN
# ==========================================

if 'autenticado' not in st.session_state:
    st.session_state.autenticado = False
if not st.session_state.autenticado:
    st.title("🔐 Iniciar sesión")
    st.markdown("Ingrese sus credenciales para acceder al sistema.")
    usuario    = st.text_input("Usuario")
    contrasena = st.text_input("Contraseña", type="password")
    if st.button("Ingresar"):
        hash_guardado = os.getenv("CONTRASENA_HASH").encode()
        if usuario == os.getenv("USUARIO") and bcrypt.checkpw(contrasena.encode(), hash_guardado):
            st.session_state.autenticado = True
            st.rerun()
        else:
            st.error("Usuario o contraseña incorrectos.")
    st.stop()

# ==========================================
# 2. NAVEGACIÓN SIDEBAR
# ==========================================

st.sidebar.title("📊 Sistema de Ponderación")
st.sidebar.markdown(f"Usuario: **{os.getenv('USUARIO')}**")
st.sidebar.divider()
seccion = st.sidebar.radio(
    "Navegación",
    options=["🏠 Inicio", "📂 Carga de encuesta", "📊 Dashboard analítico", "📋 Historial de corridas"]
)
st.sidebar.divider()
if st.sidebar.button("Cerrar sesión"):
    st.session_state.autenticado = False
    st.rerun()

# ==========================================
# 3. SECCIÓN: INICIO
# ==========================================

if seccion == "🏠 Inicio":
    st.title("📊 Sistema de Ponderación Muestral mediante Raking")
    st.markdown("""
    Esta es una aplicación de ponderación muestral construida 100% en Python.
    Permite a los usuarios cargar sus encuestas, elegir la población target y elegir la ventana temporal
    para trackear la evolución de la imagen del candidato a través del tiempo.
    """)
    st.divider()
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("**📂 Carga de encuesta**\n\nCargue su archivo CSV, elija la población objetivo y la ventana temporal.")
    with col2:
        st.info("**⚙️ Procesamiento automático**\n\nEl sistema limpia, imputa y pondera la encuesta automáticamente.")
    with col3:
        st.info("**📊 Dashboard analítico**\n\nVisualice los resultados del tracking y el reporte de calibración.")

# ==========================================
# 4. SECCIÓN: CARGA DE ENCUESTA
# ==========================================

elif seccion == "📂 Carga de encuesta":
    st.title("📂 Carga de encuesta")
    st.divider()
    col1, col2, col3 = st.columns(3)
    with col1:
        archivo = st.file_uploader("Cargar encuesta", type=["csv", "xlsx"])
    with col2:
        poblacion = st.selectbox(
            "Población objetivo",
            options=list(POBLACIONES.keys()),
            format_func=lambda x: POBLACIONES[x]
        )
    with col3:
        tipo_track = st.selectbox(
            "Ventana temporal",
            options=["d", "s", "m"],
            format_func=lambda x: {"d": "Diaria", "s": "Semanal", "m": "Mensual"}[x]
        )
    st.divider()
    procesar = st.button("Procesar encuesta", type="primary", use_container_width=True)
    if procesar:
        if archivo is None:
            st.error("Por favor cargue un archivo de encuesta.")
            st.stop()
        with st.spinner("Cargando datos..."):
            df, error = cargar_datos(archivo)
            if error:
                st.error(error)
                st.stop()
            st.success(f"Archivo cargado: {len(df)} registros")
        with st.spinner("Limpiando y normalizando..."):
            df = limpiar(df)
            df, hay_municipios_bsas = normalizar(df, poblacion)
            if hay_municipios_bsas:
                st.info("Municipios bonaerenses detectados. Recodificado en GBA/interior.")
        with st.spinner("Imputando valores faltantes..."):
            df = imputar(df)
        with st.spinner("Creando ventanas temporales..."):
            df = crear_ventanas(df)
        with st.spinner("Ponderando con raking..."):
            try:
                targets, df = obtener_targets_desde_censo(poblacion, hay_municipios_bsas, df)
            except RuntimeError as e:
                st.error(str(e))
                st.stop()
            df, target_df, vars_rake, advertencias = ponderar(df, targets)
            for adv in advertencias:
                st.warning(adv)
        # Guardar resultados en session_state para el Dashboard
        st.session_state.df = df
        st.session_state.targets = targets
        st.session_state.target_df = target_df
        st.session_state.vars_rake = vars_rake
        st.session_state.poblacion = poblacion
        st.session_state.tipo_track = tipo_track
        st.session_state.procesado = True
        # Registrar en base de datos
        adjusted_total = generar_reporte(df, target_df, vars_rake)
        pesos_w = adjusted_total.weights().df['weight']
        deff = float(1 + (pesos_w.var() / pesos_w.mean()**2))
        ess = float((pesos_w.sum()**2) / (pesos_w**2).sum())
        essp = float(ess / len(pesos_w))

        resp_corrida = requests.post(
            f"{os.getenv('API_URL')}/corridas",
            headers={"x-api-key": os.getenv("API_KEY")},
            json={
                "poblacion": poblacion,
                "n_registros": len(df),
                "variables_calib": ", ".join(vars_rake)
            },
            timeout=60
        )
        corrida_id = resp_corrida.json()["corrida_id"]
        requests.post(
            f"{os.getenv('API_URL')}/metricas",
            headers={"x-api-key": os.getenv("API_KEY")},
            json={
                "corrida_id": corrida_id,
                "deff": deff,
                "ess": ess,
                "essp": essp,
                "peso_max": float(pesos_w.max()),
                "peso_min": float(pesos_w.min())
            },
            timeout=60
        )
        st.success(f"Ponderación completada. Corrida registrada (id={corrida_id})")
        st.info("Vaya al **Dashboard analítico** en el menú lateral para ver los resultados.")
# ==========================================
# 5. SECCIÓN: DASHBOARD
# ==========================================

elif seccion == "📊 Dashboard analítico":
    st.title("📊 Dashboard analítico")
    if 'procesado' not in st.session_state or not st.session_state.procesado:
        st.warning("No hay datos procesados todavía. Vaya a **Carga de encuesta** primero.")
        st.stop()
    df = st.session_state.df
    targets = st.session_state.targets
    target_df = st.session_state.target_df
    vars_rake = st.session_state.vars_rake
    poblacion = st.session_state.poblacion
    tipo_track = st.session_state.tipo_track
    if tipo_track == "d":
        peso_col    = 'peso_d'
        ventana_col = 'Ventana_D'
    elif tipo_track == "s":
        peso_col    = 'peso_s'
        ventana_col = 'Ventana_S'
    else:
        peso_col    = 'peso_m'
        ventana_col = 'Ventana_M'
    # KPIs
    st.divider()
    col_k1, col_k2, col_k3 = st.columns(3)
    with col_k1:
        st.metric("Registros procesados", len(df))
    with col_k2:
        st.metric("Población objetivo", POBLACIONES[poblacion])
    with col_k3:
        st.metric("Ventana", {"d": "Diaria", "s": "Semanal", "m": "Mensual"}[tipo_track])
    st.divider()
     # Tracking
    st.subheader("Tracking electoral")
    tracking_imagen, tracking_voto = calcular_tracking(df, ventana_col, peso_col)
    tracking_imagen_plot = tracking_imagen.copy()
    tracking_imagen_plot[ventana_col] = tracking_imagen_plot[ventana_col].astype(str)
    fig1 = px.line(
        tracking_imagen_plot,
        x=ventana_col,
        y='trackeo',
        markers=True,
        title='Evolución de la imagen del candidato',
        labels={ventana_col: 'Ventana', 'trackeo': 'Imagen promedio'}
    )
    st.plotly_chart(fig1, use_container_width=True)
    tv = tracking_voto.copy()
    tv[ventana_col] = tv[ventana_col].astype(str)
    cols_voto = [col for col in tv.columns if col.startswith('Vota_')]
    tv_melted = tv.melt(id_vars=ventana_col, value_vars=cols_voto, var_name='Candidato', value_name='Intención de voto (%)')
    fig2 = px.line(
        tv_melted,
        x=ventana_col,
        y='Intención de voto (%)',
        color='Candidato',
        markers=True,
        title='Tracking de intención de voto',
        labels={ventana_col: 'Ventana'}
    )
    st.plotly_chart(fig2, use_container_width=True)
    st.divider()
    # Reporte de calibración
    st.subheader("📊 Reporte de calibración")
    adjusted_total = generar_reporte(df, target_df, vars_rake)
    tab1, tab2, tab3 = st.tabs(["Resumen", "Muestra vs Población", "Monitoreo de pesos"])
    with tab1:
        st.text(str(adjusted_total.summary()))
    with tab2:
        fig, axes = plt.subplots(1, len(vars_rake), figsize=(4 * len(vars_rake), 4))
        if len(vars_rake) == 1:
            axes = [axes]
        for ax, var in zip(axes, vars_rake):
            sample_props = df[var].value_counts(normalize=True).sort_index()
            target_props = pd.Series(targets[var]).sort_index()
            x = range(len(target_props))
            ax.bar([i - 0.2 for i in x], sample_props.reindex(target_props.index).fillna(0),
                   width=0.4, label='Muestra', color='salmon', alpha=0.8)
            ax.bar([i + 0.2 for i in x], target_props,
                   width=0.4, label='Población', color='steelblue', alpha=0.8)
            ax.set_title(var)
            ax.set_xticks(list(x))
            ax.set_xticklabels(target_props.index, rotation=45, ha='right')
            ax.legend()
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    with tab3:
        pesos_summary = adjusted_total.weights().summary()
        st.dataframe(pesos_summary.set_index('var'))
    # Intervalos de confianza
    st.divider()
    st.subheader("Intervalos de confianza (95%)")
    ic_img, ic_vot = calcular_intervalos(df, tipo_track)
    st.dataframe(ic_img, use_container_width=True)
    st.dataframe(ic_vot, use_container_width=True)
    st.divider()
    st.subheader("Test de hipótesis — cambio en la imagen")
    tstat, pval, rechaza_h0 = test_hipotesis(df)
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        st.metric("Estadístico", round(float(tstat), 4))
    with col_t2:
        st.metric("p-value", round(pval, 4))
    if rechaza_h0:
        st.warning("Se RECHAZA H0: la imagen cambió significativamente.")
    else:
        st.info("NO se rechaza H0: no hay evidencia de cambio significativo.")
# ==========================================
# 6. SECCIÓN: HISTORIAL DE CORRIDAS
# ========================================== 
elif seccion == "📋 Historial de corridas":
    st.title("📋 Historial de corridas")
    st.divider()
    try:
        db = next(get_db())
        corridas = db.query(CorridaDB).order_by(CorridaDB.id.desc()).limit(20).all()
        db.close()
        if not corridas:
            st.info("No hay corridas registradas todavía.")
        else:
            for corrida in corridas:
                with st.expander(f"Corrida #{corrida.id} — {corrida.fecha_hora} — {corrida.poblacion}"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Población:**", corrida.poblacion)
                        st.write("**Registros:**", corrida.n_registros)
                    with col2:
                        st.write("**Fecha:**", corrida.fecha_hora)
                        st.write("**Variables:**", corrida.variables_calib)
                    db2 = next(get_db())
                    metrica = db2.query(MetricaDB).filter(MetricaDB.corrida_id == corrida.id).first()
                    db2.close()
                    if metrica:
                        st.divider()
                        col_m1, col_m2, col_m3 = st.columns(3)
                        with col_m1:
                            st.metric("Deff", round(metrica.deff, 3))
                        with col_m2:
                            st.metric("ESS", round(metrica.ess, 1))
                        with col_m3:
                            st.metric("ESSP", round(metrica.essp, 3))
    except Exception as e:
        st.info("Procese una encuesta primero para ver el historial.")
