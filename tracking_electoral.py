# %%
# Primer paso: importar las librerías necesarias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import os
import requests
from pathlib import Path
from sklearn.linear_model import LogisticRegression, LinearRegression
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from balance.weighting_methods.rake import rake, prepare_marginal_dist_for_raking
from balance import Sample
import logging
logging.getLogger("balance").setLevel(logging.ERROR)
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from scipy.stats import ttest_ind, mannwhitneyu, levene

# %%
# Segundo paso: importar el archivo
def cargar_datos(ruta):
    try:
        if not os.path.exists(ruta):
            raise FileNotFoundError(f"No se encontró el archivo: {ruta}")
        _, extension = os.path.splitext(ruta)
        extension = extension.lower().strip()
        if extension == ".csv":
            df = pd.read_csv(ruta, encoding="utf-8")
        elif extension in [".xls", ".xlsx"]:
            df = pd.read_excel(ruta)
            ruta_csv = ruta.replace(extension, ".csv")
            df.to_csv(ruta_csv, index=False, encoding="utf-8")
            print(f"Archivo Excel convertido y guardado como CSV → {ruta_csv}")
        elif extension == ".json":
            df = pd.read_json(ruta)
            ruta_csv = ruta.replace(extension, ".csv")
            df.to_csv(ruta_csv, index=False, encoding="utf-8")
            print(f"Archivo JSON convertido y guardado como CSV → {ruta_csv}")
        elif extension == ".txt":
            df = pd.read_csv(ruta, sep="\t", encoding="utf-8")
            ruta_csv = ruta.replace(extension, ".csv")
            df.to_csv(ruta_csv, index=False, encoding="utf-8")
            print(f"Archivo TXT convertido y guardado como CSV → {ruta_csv}")
        else:
            raise ValueError(f"Formato no soportado: {extension}")
        print(f"Archivo cargado correctamente ({extension}) → {ruta}")
        print(f"  Filas: {len(df)} | Columnas: {len(df.columns)}")
        return df
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except pd.errors.EmptyDataError:
        print("Error: el archivo está vacío.")
    except pd.errors.ParserError:
        print("Error: formato de archivo incorrecto o corrupto.")
    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Error inesperado al cargar el archivo: {e}")
    return None

ruta = "ruta/a/tu/archivo.csv"   # ← cambiar por la ruta real
df = cargar_datos(ruta)

df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
)

columnas_requeridas = [
    "fecha", "encuesta", "estrato", "sexo", "edad",
    "nivel_educativo", "cantidad_de_integrantes_en_el_hogar",
    "imagen_del_candidato", "voto", "voto_anterior"
]
faltantes = [col for col in columnas_requeridas if col not in df.columns]
if faltantes:
    raise ValueError(f"Faltan columnas requeridas en el archivo: {faltantes}")

df['fecha'] = pd.to_datetime(df['fecha'])
df = df.rename(columns={"cantidad_de_integrantes_en_el_hogar": "integrantes_hogar"})

# %%
# Tercer paso: manipular los valores faltantes para las VI
df = df[~df[['voto', 'imagen_del_candidato']].isna().all(axis=1)]
df = df[df['edad'] >= 16]
df = df[~df['encuesta'].duplicated()]
df = df.dropna(subset=['fecha'])
df = df.dropna(subset=['estrato'])
df = df.dropna(subset=['nivel_educativo'])
df['nivel_educativo'] = df['nivel_educativo'].astype(str).str.strip().str.lower()
df['nivel_educativo'] = df['nivel_educativo'].replace({'sin estudios': 'prim'})

def normalizar_nivel_educativo(x):
    niveles_base = ["prim", "sec", "terc", "univ", "pos"]
    for nivel in niveles_base:
        if x.startswith(nivel):
            return nivel
    return x

df['nivel_educativo'] = df['nivel_educativo'].apply(normalizar_nivel_educativo)
df['sexo'] = df['sexo'].where(df['sexo'].isin(['femenino', 'masculino']), np.nan)
df = df.dropna(subset=['sexo'])
df = df.dropna(subset=['edad'])
df['integrantes_hogar'] = df['integrantes_hogar'].fillna('Desconocido')
print("porcentaje de nans previo a la imputación:", df.isna().mean() * 100)

# %%
# Cuarto Paso: normalización y agrupación de variables
df['estrato'] = df['estrato'].astype(str).str.strip().str.lower()
df['sexo']    = df['sexo'].astype(str).str.strip().str.lower()

df['edad_cat'] = pd.cut(
    df['edad'],
    bins=[15, 29, 44, 59, 120],
    labels=['16-29', '30-44', '45-59', '60+']
)

GBA_PARTIDOS = { # Recodificación GBA/interior - solo para encuestas de Buenos Aires.
        "almirante brown", "avellaneda", "berazategui", "berisso", 
        "brandsen", "campana", "cañuelas", "ensenada", "escobar", 
        "esteban echeverría", "exaltación de la cruz", "ezeiza", 
        "florencio varela", "general las heras", "general rodríguez", 
        "general san martín", "lanús", "la plata", "lomas de zamora", "luján", 
        "marcos paz", "malvinas argentinas", "merlo", "moreno", "morón", 
        "pilar", "presidente perón", "quilmes", "san fernando", 
        "san isidro", "san miguel", "san vicente", "tigre", 
        "tres de febrero", "vicente lópez"
        }

# %%
# Quinto Paso: calcular los valores faltantes para las VD
print("Evaluación de modelos de regresión para imputación: logística y lineal\n")
print("EVALUACIÓN VOTO_ANTERIOR\n")
df_eval_va = df[df['voto_anterior'].notna()].copy()
features_va_eval = ['edad', 'sexo', 'nivel_educativo']
X_va = pd.get_dummies(df_eval_va[features_va_eval], drop_first=True)
y_va = df_eval_va['voto_anterior'].astype('category')
y_va_num = y_va.cat.codes
mapeo_va = dict(enumerate(y_va.cat.categories))
X_train_va, X_test_va, y_train_va, y_test_va = train_test_split(
    X_va, y_va_num, test_size=0.3, random_state=42, stratify=y_va_num
)
model_va_eval = LogisticRegression(multi_class='multinomial', solver='newton-cg', max_iter=2000)
model_va_eval.fit(X_train_va, y_train_va)
y_pred_va = model_va_eval.predict(X_test_va)
print("Accuracy:", accuracy_score(y_test_va, y_pred_va))
labels_va = np.unique(y_test_va)
names_va  = [mapeo_va[i] for i in labels_va]
print("\nClassification report:")
print(classification_report(y_test_va, y_pred_va, labels=labels_va, target_names=names_va, zero_division=0))
print("\nMatriz de confusión:")
print(confusion_matrix(y_test_va, y_pred_va, labels=labels_va))

print("\nEVALUACIÓN VOTO\n")
df_eval_v = df[df['voto'].notna()].copy()
features_voto_eval = ['edad', 'sexo', 'nivel_educativo', 'voto_anterior']
X_v = pd.get_dummies(df_eval_v[features_voto_eval], drop_first=True)
y_v = df_eval_v['voto'].astype('category')
y_v_num = y_v.cat.codes
mapeo_v = dict(enumerate(y_v.cat.categories))
X_train_v, X_test_v, y_train_v, y_test_v = train_test_split(
    X_v, y_v_num, test_size=0.3, random_state=42, stratify=y_v_num
)
model_v_eval = LogisticRegression(multi_class='multinomial', solver='newton-cg', max_iter=2000)
model_v_eval.fit(X_train_v, y_train_v)
y_pred_v = model_v_eval.predict(X_test_v)
print("Accuracy:", accuracy_score(y_test_v, y_pred_v))
labels_v = np.unique(y_test_v)
names_v   = [mapeo_v[i] for i in labels_v]
print("\nClassification report:")
print(classification_report(y_test_v, y_pred_v, labels=labels_v, target_names=names_v, zero_division=0))
print("\nMatriz de confusión:")
print(confusion_matrix(y_test_v, y_pred_v, labels=labels_v))

print("\nEVALUACIÓN IMAGEN_DEL_CANDIDATO\n")
df_eval_img = df[df['imagen_del_candidato'].notna()].copy()
features_img_eval = ['edad', 'sexo', 'nivel_educativo', 'voto', 'voto_anterior']
X_img = pd.get_dummies(df_eval_img[features_img_eval], drop_first=True)
y_img = df_eval_img['imagen_del_candidato']
X_train_img, X_test_img, y_train_img, y_test_img = train_test_split(
    X_img, y_img, test_size=0.3, random_state=42
)
model_img_eval = LinearRegression()
model_img_eval.fit(X_train_img, y_train_img)
y_pred_img = model_img_eval.predict(X_test_img)
print("MAE:",  mean_absolute_error(y_test_img, y_pred_img))
print("RMSE:", np.sqrt(mean_squared_error(y_test_img, y_pred_img)))
print("R²:",   r2_score(y_test_img, y_pred_img))

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
    preds  = model.predict(X_miss)
    df.loc[df[variable_objetivo].isna(), variable_objetivo] = preds
    return df

def imputar_numerica(df, variable_objetivo, variables_predictoras):
    if r2_score(y_test_img, y_pred_img) > 0.15:
        print("El modelo de IMAGEN es suficientemente bueno: imputando con regresión lineal")
        df_full = df[df[variable_objetivo].notna()]
        df_miss = df[df[variable_objetivo].isna()]
        if len(df_miss) == 0:
            return df
        X_full = pd.get_dummies(df_full[variables_predictoras], drop_first=True)
        y_full = df_full[variable_objetivo]
        model  = LinearRegression()
        model.fit(X_full, y_full)
        X_miss = pd.get_dummies(df_miss[variables_predictoras], drop_first=True)
        X_miss = X_miss.reindex(columns=X_full.columns, fill_value=0)
        preds  = model.predict(X_miss)
        df.loc[df[variable_objetivo].isna(), variable_objetivo] = preds
        return df
    else:
        print("El modelo de IMAGEN NO es bueno, se usará imputación por mediana")
        df['imagen_del_candidato'] = df['imagen_del_candidato'].fillna(
            df['imagen_del_candidato'].median()
        )
        return df

df = imputar_categorica(df, 'voto_anterior', ['edad', 'sexo', 'estrato', 'nivel_educativo'])
df = imputar_categorica(df, 'voto', ['edad', 'sexo', 'estrato', 'nivel_educativo', 'voto_anterior'])
df = imputar_numerica(df,   'imagen_del_candidato', ['edad', 'sexo', 'estrato', 'nivel_educativo', 'voto', 'voto_anterior'])
df['imagen_del_candidato'] = df['imagen_del_candidato'].clip(lower=0, upper=100)
df['voto']          = df['voto'].astype(str).str.strip().str.lower()
df['voto_anterior'] = df['voto_anterior'].astype(str).str.strip().str.lower()
print("\nporcentaje de nans post imputación:", df.isna().mean() * 100)

# %%
# Sexto Paso: definir la ventana
df = df.sort_values('fecha')
df['Ventana_D'] = df['fecha']
df['Ventana_S'] = df['fecha'].dt.to_period('W')
df['Ventana_M'] = df['fecha'].dt.to_period('M')

# %%
# Séptimo Paso: ponderación con targets desde el Censo 2022
df['peso_d'] = 1
df['peso_s'] = 1
df['peso_m'] = 1

POBLACIONES = {
    # Regiones
    "nacional":              "Total Argentina",
    "gba":                   "Gran Buenos Aires (39 partidos)",
    "interior_buenos_aires": "Provincia de Buenos Aires sin GBA",
    "pampeana":              "Región Pampeana",
    "noa":                   "Región NOA",
    "nea":                   "Región NEA",
    "cuyo":                  "Región Cuyo",
    "patagonia":             "Región Patagonia",
    # 24 provincias
    "caba":                  "Ciudad Autónoma de Buenos Aires",
    "buenos_aires":          "Provincia de Buenos Aires",
    "catamarca":             "Catamarca",
    "cordoba":               "Córdoba",
    "corrientes":            "Corrientes",
    "chaco":                 "Chaco",
    "chubut":                "Chubut",
    "entre_rios":            "Entre Ríos",
    "formosa":               "Formosa",
    "jujuy":                 "Jujuy",
    "la_pampa":              "La Pampa",
    "la_rioja":              "La Rioja",
    "mendoza":               "Mendoza",
    "misiones":              "Misiones",
    "neuquen":               "Neuquén",
    "rio_negro":             "Río Negro",
    "salta":                 "Salta",
    "san_juan":              "San Juan",
    "san_luis":              "San Luis",
    "santa_cruz":            "Santa Cruz",
    "santa_fe":              "Santa Fe",
    "santiago_estero":       "Santiago del Estero",
    "tierra_del_fuego":      "Tierra del Fuego",
    "tucuman":               "Tucumán",
}

def obtener_targets_desde_censo(poblacion):
    API_URL = f"http://localhost:8000/targets/{poblacion}"
    print(f"  Consultando API — {POBLACIONES.get(poblacion, poblacion)}...")
    try:
        respuesta = requests.get(API_URL, timeout=10)
        respuesta.raise_for_status()
        targets = respuesta.json()["targets"]
        if poblacion == "buenos_aires" and hay_municipios_bsas:
            resp_estrato = requests.get(
                "http://localhost:8000/estrato-bsas",
                timeout=10
            )
            resp_estrato.raise_for_status()
            targets["estrato_bsas"] = resp_estrato.json()["estrato"]
            print("  Variable 'estrato_bsas' agregada (GBA/interior).")
        elif poblacion == "nacional": # Crear columna region solo si la encuesta es nacional
            df['region'] = df['estrato'].map({
                'buenos aires':                      'Región Metropolitana',
                'ciudad autónoma de buenos aires':   'Región Metropolitana',
                'córdoba':                           'Región Pampeana',
                'entre ríos':                        'Región Pampeana',
                'la pampa':                          'Región Pampeana',
                'santa fe':                          'Región Pampeana',
                'catamarca':                         'Región NOA',
                'jujuy':                             'Región NOA',
                'la rioja':                          'Región NOA',
                'salta':                             'Región NOA',
                'santiago del estero':               'Región NOA',
                'tucumán':                           'Región NOA',
                'chaco':                             'Región NEA',
                'corrientes':                        'Región NEA',
                'formosa':                           'Región NEA',
                'misiones':                          'Región NEA',
                'mendoza':                           'Región Cuyo',
                'san juan':                          'Región Cuyo',
                'san luis':                          'Región Cuyo',
                'chubut':                            'Región Patagonia',
                'neuquén':                           'Región Patagonia',
                'río negro':                         'Región Patagonia',
                'santa cruz':                        'Región Patagonia',
                'tierra del fuego':                  'Región Patagonia',
            })
            respuesta_region = requests.get("http://localhost:8000/region-nacional", timeout=10)
            targets["region"] = respuesta_region.json()["region"]
            print("Columna region creada para calibración nacional.")
        print(f"  Variables de calibración: {list(targets.keys())}")
        return targets
    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            "No se pudo conectar con la API."
        )
        
def elegir_targets():
    print("Poblaciones disponibles:")
    for clave in POBLACIONES:
        print(clave)
    while True:
        poblacion = input("Escribí la población: ").strip().lower()
        if poblacion in POBLACIONES:
            break
        print("Población no reconocida. Por favor elegí una de la lista.")
    targets = obtener_targets_desde_censo(poblacion)
    return targets, poblacion

targets, poblacion = elegir_targets()
# Verificar si la encuesta tiene municipios bonaerenses
hay_municipios_bsas = False
for estrato in df['estrato'].astype(str).str.lower().str.strip().unique():
    if estrato in GBA_PARTIDOS:
        hay_municipios_bsas = True
        break
if hay_municipios_bsas:
    print("Municipios bonaerenses detectados en 'estrato'. Recodificando en GBA/interior...")
    zonas = []
    for estrato in df['estrato'].astype(str).str.lower().str.strip():
        if estrato in GBA_PARTIDOS:
            zonas.append("gba")
        else:
            zonas.append("interior")
    df['estrato_bsas'] = zonas

if poblacion == "nacional": # Crear columna region solo si la encuesta es nacional
    df['region'] = df['estrato'].map({
        'buenos aires':                      'Región Metropolitana',
        'ciudad autónoma de buenos aires':   'Región Metropolitana',
        'córdoba':                           'Región Pampeana',
        'entre ríos':                        'Región Pampeana',
        'la pampa':                          'Región Pampeana',
        'santa fe':                          'Región Pampeana',
        'catamarca':                         'Región NOA',
        'jujuy':                             'Región NOA',
        'la rioja':                          'Región NOA',
        'salta':                             'Región NOA',
        'santiago del estero':               'Región NOA',
        'tucumán':                           'Región NOA',
        'chaco':                             'Región NEA',
        'corrientes':                        'Región NEA',
        'formosa':                           'Región NEA',
        'misiones':                          'Región NEA',
        'mendoza':                           'Región Cuyo',
        'san juan':                          'Región Cuyo',
        'san luis':                          'Región Cuyo',
        'chubut':                            'Región Patagonia',
        'neuquén':                           'Región Patagonia',
        'río negro':                         'Región Patagonia',
        'santa cruz':                        'Región Patagonia',
        'tierra del fuego':                  'Región Patagonia',
    })
    
respuesta_region = requests.get("http://localhost:8000/region-nacional", timeout=10)
    targets["region"] = respuesta_region.json()["region"]
    print("Columna region creada para calibración nacional.")

for var in targets.keys():
    df[var] = df[var].astype(str)

target_df      = prepare_marginal_dist_for_raking(targets)
target_weights = pd.Series(1.0, index=target_df.index, name="w_target")
vars_rake      = list(targets.keys())

def aplicar_rake_diario(grupo):
    try:
        res = rake(
            sample_df      = grupo[vars_rake],
            sample_weights = grupo['peso_d'],
            target_df      = target_df,
            target_weights = target_weights,
            variables      = vars_rake
        )
        pesos = res['weight'].values
        promedio = np.mean(pesos)
        grupo['peso_d'] = np.clip(pesos, promedio / 3, promedio * 3)
        deff = 1 + (pesos.var() / pesos.mean()**2)
        cv   = pesos.std() / pesos.mean() * 100
        if deff > 2.5:
            print("ADVERTENCIA ventana", grupo.name, ": Deff alto (", round(deff, 2), "). Considere ampliar la ventana.")
        if cv > 80:
            print("ADVERTENCIA ventana", grupo.name, ": CV alto (", round(cv, 1), "%). La muestra tiene perfiles muy subrepresentados.")
    except ValueError as e:
        print("No se pudo hacer raking en ventana", grupo.name, "(peso_d):", e)
        w = grupo['peso_d'].fillna(1)
        grupo['peso_d'] = w / w.mean()
    return grupo

def aplicar_rake_semanal(grupo):
    try:
        res = rake(
            sample_df      = grupo[vars_rake],
            sample_weights = grupo['peso_s'],
            target_df      = target_df,
            target_weights = target_weights,
            variables      = vars_rake
        )
        pesos = res['weight'].values
        promedio = np.mean(pesos)
        grupo['peso_s'] = np.clip(pesos, promedio / 3, promedio * 3)
        deff = 1 + (pesos.var() / pesos.mean()**2)
        cv   = pesos.std() / pesos.mean() * 100
        if deff > 2.5:
            print("ADVERTENCIA ventana", grupo.name, ": Deff alto (", round(deff, 2), "). Considere ampliar la ventana.")
        if cv > 80:
            print("ADVERTENCIA ventana", grupo.name, ": CV alto (", round(cv, 1), "%). La muestra tiene perfiles muy subrepresentados.")
    except ValueError as e:
        print("No se pudo hacer raking en ventana", grupo.name, "(peso_s):", e)
        w = grupo['peso_s'].fillna(1)
        grupo['peso_s'] = w / w.mean()
    return grupo

def aplicar_rake_mensual(grupo):
    try:
        res = rake(
            sample_df      = grupo[vars_rake],
            sample_weights = grupo['peso_m'],
            target_df      = target_df,
            target_weights = target_weights,
            variables      = vars_rake
        )
        pesos = res['weight'].values
        promedio = np.mean(pesos)
        grupo['peso_m'] = np.clip(pesos, promedio / 3, promedio * 3)
        deff = 1 + (pesos.var() / pesos.mean()**2)
        cv   = pesos.std() / pesos.mean() * 100
        if deff > 2.5:
            print("ADVERTENCIA ventana", grupo.name, ": Deff alto (", round(deff, 2), "). Considere ampliar la ventana.")
        if cv > 80:
            print("ADVERTENCIA ventana", grupo.name, ": CV alto (", round(cv, 1), "%). La muestra tiene perfiles muy subrepresentados.")
    except ValueError as e:
        print("No se pudo hacer raking en ventana", grupo.name, "(peso_m):", e)
        w = grupo['peso_m'].fillna(1)
        grupo['peso_m'] = w / w.mean()
    return grupo

df_rake_diario  = df.groupby('Ventana_D', group_keys=False).apply(aplicar_rake_diario)
df_rake_semana  = df.groupby('Ventana_S', group_keys=False).apply(aplicar_rake_semanal)
df_rake_mensual = df.groupby('Ventana_M', group_keys=False).apply(aplicar_rake_mensual)

df['peso_d'] = df_rake_diario['peso_d']
df['peso_s'] = df_rake_semana['peso_s']
df['peso_m'] = df_rake_mensual['peso_m']

def normalizar_pesos(df, peso_col, ventana_col):
    df[peso_col] = df.groupby(ventana_col)[peso_col].transform(
        lambda w: w / w.sum() * len(w)
    )
    return df

df = normalizar_pesos(df, 'peso_d', 'Ventana_D')
df = normalizar_pesos(df, 'peso_s', 'Ventana_S')
df = normalizar_pesos(df, 'peso_m', 'Ventana_M')
df

# %%
# Octavo Paso: TRACKING DIARIO
def tracking_diario():
    tracking_imagen_diario = (
        df.groupby('Ventana_D')
        .apply(lambda g: np.average(g['imagen_del_candidato'], weights=g['peso_d']))
        .reset_index(name='trackeo')
    )
    print(tracking_imagen_diario.round(1))
    plt.figure(figsize=(10, 5))
    plt.plot(tracking_imagen_diario['Ventana_D'], tracking_imagen_diario['trackeo'], marker='o')
    plt.xlabel('Ventana (diaria)', fontsize=10)
    plt.ylabel('Imagen promedio', fontsize=10)
    plt.title('Evolución de la imagen del candidato (ventana diaria)', fontsize=16)
    plt.tight_layout()
    plt.show()

    candidatos = df['voto'].unique().tolist()
    for c in candidatos:
        df[f'vota_{c}'] = (df['voto'] == c).astype(int)
    tracking_voto_diario = (
        df.groupby('Ventana_D')
        .apply(lambda g: pd.Series({
            f"Vota_{c}": np.average(g[f'vota_{c}'], weights=g['peso_d']) * 100
            for c in candidatos
        }))
        .reset_index()
    )
    print(tracking_voto_diario.round(1))
    cols_voto = [col for col in tracking_voto_diario.columns if col.startswith('Vota_')]
    tracking_voto_diario.set_index('Ventana_D')[cols_voto].plot(figsize=(10, 5))
    plt.xlabel('Ventana (diaria)', fontsize=10)
    plt.ylabel('Intención de voto (%)', fontsize=10)
    plt.title('Tracking de intención de voto (ventana diaria)', fontsize=16)
    plt.grid(alpha=0.3)
    plt.legend(title="Candidato")
    plt.tight_layout()
    plt.show()
