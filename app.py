import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import warnings
import logging
warnings.filterwarnings("ignore", category=FutureWarning)
logging.getLogger("balance").setLevel(logging.ERROR)
from base_datos import registrar_corrida, registrar_metricas, get_db
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import ttest_ind, mannwhitneyu, levene
from balance.weighting_methods.rake import rake, prepare_marginal_dist_for_raking
from balance import Sample

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
# 2. CLASIFICACIÓN
# ==========================================

POBLACIONES = {
    "nacional": "Total Argentina",
    "gba": "Gran Buenos Aires (39 partidos)",
    "interior_buenos_aires": "Provincia de Buenos Aires sin GBA",
    "pampeana": "Región Pampeana",
    "noa": "Región NOA",
    "nea": "Región NEA",
    "cuyo": "Región Cuyo",
    "patagonia": "Región Patagonia",
    "caba": "Ciudad Autónoma de Buenos Aires",
    "buenos_aires": "Provincia de Buenos Aires",
    "catamarca": "Catamarca",
    "cordoba": "Córdoba",
    "corrientes": "Corrientes",
    "chaco": "Chaco",
    "chubut": "Chubut",
    "entre_rios": "Entre Ríos",
    "formosa": "Formosa",
    "jujuy": "Jujuy",
    "la_pampa": "La Pampa",
    "la_rioja": "La Rioja",
    "mendoza": "Mendoza",
    "misiones": "Misiones",
    "neuquen": "Neuquén",
    "rio_negro": "Río Negro",
    "salta": "Salta",
    "san_juan": "San Juan",
    "san_luis": "San Luis",
    "santa_cruz": "Santa Cruz",
    "santa_fe": "Santa Fe",
    "santiago_estero": "Santiago del Estero",
    "tierra_del_fuego": "Tierra del Fuego",
    "tucuman": "Tucumán",
}

GBA_PARTIDOS = {
    "almirante brown", "avellaneda", "berazategui", "berisso",
    "brandsen", "campana", "cañuelas", "ensenada", "escobar",
    "esteban echeverría", "exaltación de la cruz", "ezeiza",
    "florencio varela", "general las heras", "general rodríguez",
    "general san martín", "hurlingham", "ituzaingó", "josé c. paz",
    "la matanza", "lanús", "la plata", "lomas de zamora", "luján",
    "marcos paz", "malvinas argentinas", "merlo", "moreno", "morón",
    "pilar", "presidente perón", "quilmes", "san fernando",
    "san isidro", "san miguel", "san vicente", "tigre",
    "tres de febrero", "vicente lópez"
}

# ==========================================
# 3. INTERACTIVIDAD
# ==========================================

col1, col2, col3 = st.columns(3)

with col1:
    archivo = st.file_uploader("Cargar encuesta", type=["csv","xlsx"])

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
# 4. PIPELINE
# ==========================================

if procesar:
    if archivo is None:
        st.error("Por favor cargue un archivo de encuesta.")
        st.stop()
    with st.spinner("Cargando datos..."):
        # Paso 2: cargar
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
        # Paso 3: limpiar
        df = df[~df[['voto', 'imagen_del_candidato']].isna().all(axis=1)]
        df = df[df['edad'] >= 16]
        df = df[~df['encuesta'].duplicated()]
        df = df.dropna(subset=['fecha', 'estrato', 'nivel_educativo', 'sexo', 'edad'])
        df['nivel_educativo'] = df['nivel_educativo'].astype(str).str.strip().str.lower()
        df['nivel_educativo'] = df['nivel_educativo'].replace({'sin estudios': 'prim'})
        def normalizar_nivel_educativo(x):
            for nivel in ["prim", "sec", "terc", "univ", "pos"]:
                if x.startswith(nivel):
                    return nivel
            return x
        df['nivel_educativo'] = df['nivel_educativo'].apply(normalizar_nivel_educativo)
        df['sexo'] = df['sexo'].where(df['sexo'].isin(['femenino', 'masculino']), np.nan)
        df = df.dropna(subset=['sexo'])
        df['integrantes_hogar'] = df['integrantes_hogar'].fillna('Desconocido')
        # Paso 4: normalizar
        df['estrato'] = df['estrato'].astype(str).str.strip().str.lower()
        df['sexo']    = df['sexo'].astype(str).str.strip().str.lower()
        df['edad_cat'] = pd.cut(df['edad'], bins=[15, 29, 44, 59, 120], labels=['16-29', '30-44', '45-59', '60+'])
        hay_municipios_bsas = False
        for estrato in df['estrato'].unique():
            if estrato in GBA_PARTIDOS:
                hay_municipios_bsas = True
                break
        if hay_municipios_bsas:
            zonas = []
            for estrato in df['estrato']:
                if estrato in GBA_PARTIDOS:
                    zonas.append("gba")
                else:
                    zonas.append("interior")
            df['estrato_bsas'] = zonas
            st.info("Municipios bonaerenses detectados. Recodificado en GBA/interior.")
    with st.spinner("Imputando valores faltantes..."):
        # Paso 5: imputar
        def imputar_categorica(df, variable_objetivo, variables_predictoras):
            df_full = df[df[variable_objetivo].notna()]
            df_miss = df[df[variable_objetivo].isna()]
            if len(df_miss) == 0:
                return df
            X_full = pd.get_dummies(df_full[variables_predictoras], drop_first=True)
            y_full = df_full[variable_objetivo]
            model  = LogisticRegression(multi_class='multinomial', solver='newton-cg', max_iter=2000)
            model.fit(X_full, y_full)
            X_miss = pd.get_dummies(df_miss[variables_predictoras], drop_first=True)
            X_miss = X_miss.reindex(columns=X_full.columns, fill_value=0)
            df.loc[df[variable_objetivo].isna(), variable_objetivo] = model.predict(X_miss)
            return df
        df_eval_img = df[df['imagen_del_candidato'].notna()].copy()
        X_img = pd.get_dummies(df_eval_img[['edad', 'sexo', 'nivel_educativo', 'voto', 'voto_anterior']], drop_first=True)
        y_img = df_eval_img['imagen_del_candidato']
        X_train_img, X_test_img, y_train_img, y_test_img = train_test_split(X_img, y_img, test_size=0.3, random_state=42)
        model_img = LinearRegression()
        model_img.fit(X_train_img, y_train_img)
        y_pred_img = model_img.predict(X_test_img)
        r2_img = r2_score(y_test_img, y_pred_img)
        def imputar_numerica(df, variable_objetivo, variables_predictoras):
            df_full = df[df[variable_objetivo].notna()]
            df_miss = df[df[variable_objetivo].isna()]
            if len(df_miss) == 0:
                return df
            if r2_img > 0.15:
                X_full = pd.get_dummies(df_full[variables_predictoras], drop_first=True)
                y_full = df_full[variable_objetivo]
                model  = LinearRegression()
                model.fit(X_full, y_full)
                X_miss = pd.get_dummies(df_miss[variables_predictoras], drop_first=True)
                X_miss = X_miss.reindex(columns=X_full.columns, fill_value=0)
                df.loc[df[variable_objetivo].isna(), variable_objetivo] = model.predict(X_miss)
            else:
                df[variable_objetivo] = df[variable_objetivo].fillna(df[variable_objetivo].median())
            return df
        df = imputar_categorica(df, 'voto_anterior', ['edad', 'sexo', 'estrato', 'nivel_educativo'])
        df = imputar_categorica(df, 'voto',          ['edad', 'sexo', 'estrato', 'nivel_educativo', 'voto_anterior'])
        df = imputar_numerica(df,   'imagen_del_candidato', ['edad', 'sexo', 'estrato', 'nivel_educativo', 'voto', 'voto_anterior'])
        df['imagen_del_candidato'] = df['imagen_del_candidato'].clip(lower=0, upper=100)
        df['voto']          = df['voto'].astype(str).str.strip().str.lower()
        df['voto_anterior'] = df['voto_anterior'].astype(str).str.strip().str.lower()
    with st.spinner("Ponderando con raking..."):
        # Paso 6: ventanas
        df = df.sort_values('fecha')
        df['Ventana_D'] = df['fecha']
        df['Ventana_S'] = df['fecha'].dt.to_period('W')
        df['Ventana_M'] = df['fecha'].dt.to_period('M')
        # Paso 7: targets desde API
        try:
            API_URL = f"http://localhost:8000/targets/{poblacion}"
            respuesta = requests.get(API_URL, timeout=10)
            respuesta.raise_for_status()
            targets = respuesta.json()["targets"]
            if poblacion == "buenos_aires" and hay_municipios_bsas:
                resp_estrato = requests.get("http://localhost:8000/estrato-bsas", timeout=10)
                resp_estrato.raise_for_status()
                targets["estrato_bsas"] = resp_estrato.json()["estrato"]
            elif poblacion == "nacional":
                df['region'] = df['estrato'].map({
                    'buenos aires': 'Región Metropolitana',
                    'ciudad autónoma de buenos aires': 'Región Metropolitana',
                    'córdoba': 'Región Pampeana',
                    'entre ríos': 'Región Pampeana',
                    'la pampa': 'Región Pampeana',
                    'santa fe': 'Región Pampeana',
                    'catamarca': 'Región NOA',
                    'jujuy': 'Región NOA',
                    'la rioja': 'Región NOA',
                    'salta': 'Región NOA',
                    'santiago del estero': 'Región NOA',
                    'tucumán': 'Región NOA',
                    'chaco': 'Región NEA',
                    'corrientes': 'Región NEA',
                    'formosa': 'Región NEA',
                    'misiones': 'Región NEA',
                    'mendoza': 'Región Cuyo',
                    'san juan': 'Región Cuyo',
                    'san luis': 'Región Cuyo',
                    'chubut': 'Región Patagonia',
                    'neuquén': 'Región Patagonia',
                    'río negro': 'Región Patagonia',
                    'santa cruz': 'Región Patagonia',
                    'tierra del fuego': 'Región Patagonia',
                })
                respuesta_region = requests.get("http://localhost:8000/region-nacional", timeout=10)
                targets["region"] = respuesta_region.json()["region"]
        except requests.exceptions.ConnectionError:
            st.error("No se pudo conectar con la API")
            st.stop()
        # Paso 7: raking
        df['peso_d'] = 1
        df['peso_s'] = 1
        df['peso_m'] = 1
        for var in targets.keys():
            df[var] = df[var].astype(str)
        target_df      = prepare_marginal_dist_for_raking(targets)
        target_weights = pd.Series(1.0, index=target_df.index, name="w_target")
        vars_rake      = list(targets.keys())
        def aplicar_rake(grupo, peso_col):
            try:
                res = rake(sample_df=grupo[vars_rake], sample_weights=grupo[peso_col],
                   target_df=target_df, target_weights=target_weights, variables=vars_rake)
                pesos = res['weight'].values
                promedio = np.mean(pesos)
                grupo[peso_col] = np.clip(pesos, promedio / 3, promedio * 3)
                deff = 1 + (pesos.var() / pesos.mean()**2)
                cv   = pesos.std() / pesos.mean() * 100
                if deff > 2.5:
                    st.warning(f"ADVERTENCIA ventana {grupo.name}: Deff alto ({round(deff, 2)}). Considere ampliar la ventana.")
                if cv > 80:
                    st.warning(f"ADVERTENCIA ventana {grupo.name}: CV alto ({round(cv, 1)}%). La muestra tiene perfiles muy subrepresentados.")
            except ValueError:
                w = grupo[peso_col].fillna(1)
                grupo[peso_col] = w / w.mean()
            return grupo
        df_rake_d = df.groupby('Ventana_D', group_keys=False).apply(lambda g: aplicar_rake(g, 'peso_d'))
        df_rake_s = df.groupby('Ventana_S', group_keys=False).apply(lambda g: aplicar_rake(g, 'peso_s'))
        df_rake_m = df.groupby('Ventana_M', group_keys=False).apply(lambda g: aplicar_rake(g, 'peso_m'))
        df['peso_d'] = df_rake_d['peso_d']
        df['peso_s'] = df_rake_s['peso_s']
        df['peso_m'] = df_rake_m['peso_m']
        def normalizar_pesos(df, peso_col, ventana_col):
            df[peso_col] = df.groupby(ventana_col)[peso_col].transform(lambda w: w / w.sum() * len(w))
            return df
        df = normalizar_pesos(df, 'peso_d', 'Ventana_D')
        df = normalizar_pesos(df, 'peso_s', 'Ventana_S')
        df = normalizar_pesos(df, 'peso_m', 'Ventana_M')
    st.success("Ponderación completada.")
    st.divider()

    # ==========================================
    # 5. RESULTADOS
    # ==========================================
    # Tracking
    st.subheader("Tracking electoral")

    if tipo_track == "d":
        peso_col    = 'peso_d'
        ventana_col = 'Ventana_D'
    elif tipo_track == "s":
        peso_col    = 'peso_s'
        ventana_col = 'Ventana_S'
    else:
        peso_col    = 'peso_m'
        ventana_col = 'Ventana_M'

    tracking_imagen = (
        df.groupby(ventana_col)
        .apply(lambda g: np.average(g['imagen_del_candidato'], weights=g[peso_col]))
        .reset_index(name='trackeo')
    )

    fig1, ax1 = plt.subplots(figsize=(10, 4))
    tracking_imagen_plot = tracking_imagen.copy()
    tracking_imagen_plot[ventana_col] = tracking_imagen_plot[ventana_col].astype(str)
    ax1.plot(tracking_imagen_plot[ventana_col], tracking_imagen_plot['trackeo'], marker='o')
    ax1.set_xlabel('Ventana')
    ax1.set_ylabel('Imagen promedio')
    ax1.set_title('Evolución de la imagen del candidato')
    n_labels = 10
    ticks = list(range(0, len(tracking_imagen_plot), max(1, len(tracking_imagen_plot) // n_labels)))
    ax1.set_xticks(ticks)
    ax1.set_xticklabels([tracking_imagen_plot[ventana_col].iloc[i] for i in ticks], rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(fig1)
    plt.close()

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

    fig2, ax2 = plt.subplots(figsize=(10, 4))
    cols_voto = [col for col in tracking_voto.columns if col.startswith('Vota_')]
    tv = tracking_voto.copy()
    tv[ventana_col] = tv[ventana_col].astype(str)
    for col in cols_voto:
        ax2.plot(tv[ventana_col], tv[col], marker='o', label=col)
    ax2.set_xlabel('Ventana')
    ax2.set_ylabel('Intención de voto (%)')
    ax2.set_title('Tracking de intención de voto')
    ax2.legend(title="Candidato")
    ax2.grid(alpha=0.3)
    n_labels = 10
    ticks = list(range(0, len(tv), max(1, len(tv) // n_labels)))
    ax2.set_xticks(ticks)
    ax2.set_xticklabels([tv[ventana_col].iloc[i] for i in ticks], rotation=45, ha='right')
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
    t_total = Sample.from_frame(t_df_total, id_column='_id', outcome_columns=[])
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
    st.subheader("Intervalos de confianza (95%)")

    def weighted_std(values, weights):
        values  = np.array(values)
        weights = np.array(weights)
        average = np.average(values, weights=weights)
        variance = np.average((values - average)**2, weights=weights)
        return np.sqrt(variance)

    def n_efectivo(weights):
        w = np.array(weights)
        return (w.sum()**2) / ((w**2).sum())

    def margen_error_imagen(g, pc):
        pesos = g[pc]
        n_eff = n_efectivo(pesos)
        media = np.average(g['imagen_del_candidato'], weights=pesos)
        sd_w  = weighted_std(g['imagen_del_candidato'], pesos)
        MOE   = 1.96 * sd_w / np.sqrt(n_eff)
        return pd.Series({'imagen_media': media, 'MOE_95': MOE, 'LI_95': media - MOE, 'LS_95': media + MOE})

    ic_img = df.groupby(ventana_col).apply(lambda g: margen_error_imagen(g, peso_col)).reset_index()
    st.dataframe(ic_img)

    st.divider()

    # Test de hipótesis
    st.subheader("Test de hipótesis — cambio en la imagen")

    primera_ventana = df['Ventana_S'].min()
    ultima_ventana  = df['Ventana_S'].max()
    img_ini = df.loc[df['Ventana_S'] == primera_ventana, 'imagen_del_candidato']
    img_fin = df.loc[df['Ventana_S'] == ultima_ventana,  'imagen_del_candidato']
    n_ini, n_fin = len(img_ini), len(img_fin)
    alpha = 0.05
    normalidad = (n_ini >= 30) and (n_fin >= 30)
    stat_lev, p_lev = levene(img_ini, img_fin, center='mean')
    homocedasticidad = (p_lev >= alpha)

    if normalidad and homocedasticidad:
        tstat, pval = ttest_ind(img_ini, img_fin, equal_var=True)
    elif normalidad and not homocedasticidad:
        tstat, pval = ttest_ind(img_ini, img_fin, equal_var=False)
    else:
        tstat, pval = mannwhitneyu(img_ini, img_fin, alternative='two-sided')

    if pval < alpha:
        st.warning("Se RECHAZA H0: la imagen cambió significativamente. p-value: " + str(round(pval, 4)))
    else:
        st.info("NO se rechaza H0: no hay evidencia de cambio significativo. p-value: " + str(round(pval, 4)))

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
