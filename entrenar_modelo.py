import hashlib
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    r2_score,
    mean_absolute_error,
    mean_squared_error,
)

# ==========================================
# CONFIGURACIÓN
# ==========================================

DIRECTORIO_BASE = os.path.dirname(os.path.abspath(__file__))
RUTA_ENCUESTA = os.path.join(DIRECTORIO_BASE,"encuesta_ficticia_nacional_2.csv")
def construir_ruta(nombre_archivo):
    return os.path.join(DIRECTORIO_BASE, nombre_archivo)
def generar_id_encuesta(df):
    df_hash = df.copy()
    df_hash = df_hash.reindex(sorted(df_hash.columns), axis=1)
    hashes_filas = pd.util.hash_pandas_object(df_hash, index=False).to_numpy()
    hashes_filas.sort()
    return hashlib.sha256(hashes_filas.tobytes()).hexdigest()

# ==========================================
# CARGA Y VALIDACIÓN
# ==========================================

if not os.path.exists(RUTA_ENCUESTA):
    raise FileNotFoundError(f"No se encontró el archivo: {RUTA_ENCUESTA}")
df = pd.read_csv(RUTA_ENCUESTA)
columnas_requeridas = {"edad", "sexo", "estrato", "nivel_educativo", "voto", "voto_anterior", "imagen_del_candidato",}
columnas_faltantes = columnas_requeridas - set(df.columns)
if columnas_faltantes:
    raise ValueError(f"Faltan columnas requeridas: {sorted(columnas_faltantes)}")

# ==========================================
# NORMALIZACIÓN
# ==========================================

columnas_texto = ["sexo", "estrato", "nivel_educativo", "voto", "voto_anterior",]
for columna in columnas_texto:
    mascara_observados = df[columna].notna()
    df.loc[mascara_observados, columna] = (df.loc[mascara_observados, columna].astype(str).str.strip().str.lower())
encuesta_id = generar_id_encuesta(df)

# ==========================================
# MODELO 1: VOTO ANTERIOR
# ==========================================

print("\n=== ENTRENANDO MODELO VOTO_ANTERIOR ===")
features_va = ["edad","sexo","estrato","nivel_educativo"]
columnas_va = features_va + ["voto_anterior"]
df_va = (df[columnas_va].dropna().copy())
if df_va["voto_anterior"].nunique() < 2:
    raise ValueError("voto_anterior necesita al menos dos categorías.")
X_va = pd.get_dummies(df_va[features_va],drop_first=True,)
y_va = df_va["voto_anterior"]
X_train_va, X_test_va, y_train_va, y_test_va = (train_test_split(X_va, y_va, test_size=0.3, random_state=42, stratify=y_va))
modelo_va_evaluacion = LogisticRegression( solver="newton-cg", max_iter=2000)
modelo_va_evaluacion.fit( X_train_va, y_train_va,)
pred_va = modelo_va_evaluacion.predict(X_test_va)
print("Accuracy:", accuracy_score(y_test_va, pred_va))
print("\nClassification report:")
print(classification_report(y_test_va, pred_va, zero_division=0))
print("\nMatriz de confusión:")
print(confusion_matrix(y_test_va, pred_va))
modelo_va = LogisticRegression(solver="newton-cg", max_iter=2000,)
modelo_va.fit(X_va, y_va)
datos_va = df_va.copy()
datos_va["_encuesta_id"] = encuesta_id
joblib.dump(modelo_va, construir_ruta("modelo_voto_anterior.joblib"))
joblib.dump(X_va.columns.tolist(), construir_ruta("features_voto_anterior.joblib"))
joblib.dump(datos_va, construir_ruta("datos_voto_anterior.joblib"))
print(f"Modelo de voto anterior guardado con {len(datos_va)} casos iniciales.")

# ==========================================
# MODELO 2: VOTO ACTUAL
# ==========================================

print("\n=== ENTRENANDO MODELO VOTO ===")
features_voto = ["edad","sexo", "estrato", "nivel_educativo", "voto_anterior"]
columnas_voto = features_voto + ["voto"]
df_voto = (df[columnas_voto].dropna().copy())
if df_voto["voto"].nunique() < 2:
    raise ValueError("voto necesita al menos dos categorías.")
X_voto = pd.get_dummies(df_voto[features_voto], drop_first=True)
y_voto = df_voto["voto"]
X_train_v, X_test_v, y_train_v, y_test_v = (train_test_split(X_voto, y_voto, test_size=0.3, random_state=42, stratify=y_voto))
modelo_voto_evaluacion = LogisticRegression(solver="newton-cg", max_iter=2000)
modelo_voto_evaluacion.fit(X_train_v, y_train_v)
pred_voto = modelo_voto_evaluacion.predict(X_test_v)
print("Accuracy:", accuracy_score(y_test_v, pred_voto))
print("\nClassification report:")
print(classification_report(y_test_v, pred_voto, zero_division=0))
print("\nMatriz de confusión:")
print(confusion_matrix(y_test_v, pred_voto))
modelo_voto = LogisticRegression(solver="newton-cg", max_iter=2000,)
modelo_voto.fit(X_voto, y_voto)
datos_voto = df_voto.copy()
datos_voto["_encuesta_id"] = encuesta_id
joblib.dump(modelo_voto, construir_ruta("modelo_voto.joblib"),)
joblib.dump(X_voto.columns.tolist(), construir_ruta("features_voto.joblib"))
joblib.dump(datos_voto, construir_ruta("datos_voto.joblib"),)
print(f"Modelo de voto guardado con {len(datos_voto)} casos iniciales.")

# ==========================================
# MODELO 3: IMAGEN DEL CANDIDATO
# ==========================================

print("\n=== ENTRENANDO MODELO IMAGEN ===")
features_imagen = ["edad", "sexo", "estrato", "nivel_educativo", "voto", "voto_anterior",]
columnas_imagen = (features_imagen + ["imagen_del_candidato"])
df_imagen = (df[columnas_imagen].dropna().copy())
X_imagen = pd.get_dummies(df_imagen[features_imagen],drop_first=True)
y_imagen = df_imagen["imagen_del_candidato"]
(X_train_img, X_test_img, y_train_img, y_test_img,) = train_test_split(X_imagen, y_imagen, test_size=0.3, random_state=42)
modelo_imagen_evaluacion = LinearRegression()
modelo_imagen_evaluacion.fit(X_train_img, y_train_img)
pred_imagen = modelo_imagen_evaluacion.predict(X_test_img)
print("R²:", r2_score(y_test_img, pred_imagen))
print("MAE:", mean_absolute_error(y_test_img, pred_imagen))
print("RMSE:", np.sqrt(mean_squared_error(y_test_img, pred_imagen)))
modelo_imagen = LinearRegression()
modelo_imagen.fit(X_imagen, y_imagen,)
datos_imagen = df_imagen.copy()
datos_imagen["_encuesta_id"] = encuesta_id
joblib.dump(modelo_imagen, construir_ruta("modelo_imagen.joblib"))
joblib.dump(X_imagen.columns.tolist(), construir_ruta("features_imagen.joblib"))
joblib.dump(datos_imagen, construir_ruta("datos_imagen.joblib"))
print(f"Modelo de imagen guardado con {len(datos_imagen)} casos iniciales.")
print("\nTodos los modelos, features y datos históricos fueron guardados correctamente.")
