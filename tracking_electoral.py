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

ruta = "C:/Users/charo/Downloads/DS4P/encuesta_ficticia_nacional.csv"   # ← cambiar por la ruta real
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
