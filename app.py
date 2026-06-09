import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import logging
warnings.filterwarnings("ignore", category=FutureWarning)
logging.getLogger("balance").setLevel(logging.ERROR)
from balance import Sample
from limpieza    import limpiar, normalizar, GBA_PARTIDOS
from imputacion  import imputar
from ventanas    import crear_ventanas
from ponderacion import obtener_targets_desde_censo, ponderar, POBLACIONES
from estadistica import calcular_intervalos, test_hipotesis
from base_datos  import registrar_corrida, registrar_metricas, get_db

# ==========================================
# 1. CONFIGURACIÓN DE PÁGINA Y TEXTOS
# ==========================================

st.set_page_config(page_title="Sistema de Ponderación Electoral", page_icon="📊")
st.title("📊 Sistema de Ponderación Electoral mediante Raking")
st.markdown("""
Esta es una aplicación de ponderación muestral construida 100% en Python.
Permite a los usuarios cargar sus encuestas, elegir la población target y elegir la ventana temporal para trackear la evolución de la imagen del candidato a través del tiempo.
""")
st.divider()

# ==========================================
# 2. INTERACTIVIDAD
# ==========================================

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

# ==========================================
# 3. PIPELINE
# ==========================================

if procesar:
    if archivo is None:
        st.error("Por favor cargue un archivo de encuesta.")
        st.stop()

    with st.spinner("Cargando datos..."):
        if archivo.name.endswith(".csv"):
            df = pd.read_csv(archivo, encoding="utf-8")
        else:
            df = pd.read_excel(archivo)
        df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
        columnas_requeridas = [
            "fecha", "encuesta", "estrato", "sexo", "edad",
            "nivel_educativo", "cantidad_de_integrantes_en_el_hogar",
            "imagen_del_candidato", "voto", "voto_anterior"
        ]
        faltantes = [col for col in columnas_requeridas if col not in df.columns]
        if faltantes:
            st.error(f"Faltan columnas: {faltantes}")
            st.stop()
        df['fecha'] = pd.to_datetime(df['fecha'])
        df = df.rename(columns={"cantidad_de_integrantes_en_el_hogar": "integrantes_hogar"})
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
        df, target_df, vars_rake = ponderar(df, targets)

    st.success("Ponderación completada.")
    st.divider()

    # ==========================================
    # 4. RESULTADOS
    # ==========================================

    if tipo_track == "d":
        peso_col    = 'peso_d'
        ventana_col = 'Ventana_D'
    elif tipo_track == "s":
        peso_col    = 'peso_s'
        ventana_col = 'Ventana_S'
    else:
        peso_col    = 'peso_m'
        ventana_col = 'Ventana_M'

    col_k1, col_k2, col_k3 = st.columns(3)
    with col_k1:
        st.metric("Registros procesados", len(df))
    with col_k2:
        st.metric("Población objetivo", POBLACIONES[poblacion])
    with col_k3:
        st.metric("Ventana", {"d": "Diaria", "s": "Semanal", "m": "Mensual"}[tipo_track])

    st.divider()

    # Tracking imagen
    st.subheader("Tracking electoral")
    tracking_imagen = (
        df.groupby(ventana_col)
        .apply(lambda g: np.average(g['imagen_del_candidato'], weights=g[peso_col]))
        .reset_index(name='trackeo')
    )
    tracking_imagen_plot = tracking_imagen.copy()
    tracking_imagen_plot[ventana_col] = tracking_imagen_plot[ventana_col].astype(str)

    fig1, ax1 = plt.subplots(figsize=(10, 4))
    ax1.plot(tracking_imagen_plot[ventana_col], tracking_imagen_plot['trackeo'], marker='o')
    ax1.set_xlabel('Ventana')
    ax1.set_ylabel('Imagen promedio')
    ax1.set_title('Evolución de la imagen del candidato')
    n = len(tracking_imagen_plot)
    ticks = list(range(0, n, max(1, n // 10)))
    ax1.set_xticks(ticks)
    ax1.set_xticklabels([tracking_imagen_plot[ventana_col].iloc[i] for i in ticks], rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(fig1)
    plt.close()

    # Tracking voto
    candidatos = df['voto'].unique().tolist()
    for c in candidatos:
        df[f'vota_{c}'] = (df['voto'] == c).astype(int)

    tracking_voto = (
        df.groupby(ventana_col)
        .apply(lambda g: pd.Series({
            f"Vota_{c}": np.average(g[f'vota_{c}'], weights=g[peso_col]) * 100
            for c in candidatos
        }))
        .reset_index()
    )
    tv = tracking_voto.copy()
    tv[ventana_col] = tv[ventana_col].astype(str)
    cols_voto = [col for col in tv.columns if col.startswith('Vota_')]

    fig2, ax2 = plt.subplots(figsize=(10, 4))
    for col in cols_voto:
        ax2.plot(tv[ventana_col], tv[col], marker='o', label=col)
    ax2.set_xlabel('Ventana')
    ax2.set_ylabel('Intención de voto (%)')
    ax2.set_title('Tracking de intención de voto')
    ax2.legend(title="Candidato")
    ax2.grid(alpha=0.3)
    n2 = len(tv)
    ticks2 = list(range(0, n2, max(1, n2 // 10)))
    ax2.set_xticks(ticks2)
    ax2.set_xticklabels([tv[ventana_col].iloc[i] for i in ticks2], rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()

    st.divider()

    # Reporte de calibración
    st.subheader("📊 Reporte de calibración (total muestra)")
    sample_df_total = df[vars_rake].copy().reset_index(drop=True)
    sample_df_total.insert(0, '_id', range(len(sample_df_total)))
    t_df_total = target_df.copy()
    if '_id' not in t_df_total.columns:
        t_df_total.insert(0, '_id', range(len(t_df_total)))
    s_total = Sample.from_frame(sample_df_total, id_column='_id', outcome_columns=[])
    t_total = Sample.from_frame(t_df_total,      id_column='_id', outcome_columns=[])
    adjusted_total = s_total.set_target(t_total).adjust(method='rake').trim(ratio=3)

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

    # Test de hipótesis
    st.divider()
    st.subheader("Test de hipótesis — cambio en la imagen")
    tstat, pval = test_hipotesis(df)

    col_t1, col_t2 = st.columns(2)
    with col_t1:
        st.metric("Estadístico", round(float(tstat), 4))
    with col_t2:
        st.metric("p-value", round(pval, 4))

    alpha = 0.05
    if pval < alpha:
        st.warning("Se RECHAZA H0: la imagen cambió significativamente.")
    else:
        st.info("NO se rechaza H0: no hay evidencia de cambio significativo.")

    # ==========================================
    # 5. BASE DE DATOS
    # ==========================================

    pesos_w = adjusted_total.weights().df['weight']
    deff    = 1 + (pesos_w.var() / pesos_w.mean()**2)
    ess     = (pesos_w.sum()**2) / (pesos_w**2).sum()
    essp    = ess / len(pesos_w)

    db = next(get_db())
    corrida_id = registrar_corrida(db, poblacion, len(df), vars_rake)
    registrar_metricas(db, corrida_id, deff, ess, essp, pesos_w.max(), pesos_w.min())
    db.close()

    st.divider()
    st.success(f"Corrida registrada en la base de datos (id={corrida_id})")
