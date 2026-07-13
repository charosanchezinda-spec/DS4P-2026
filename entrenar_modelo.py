import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, r2_score, mean_absolute_error, mean_squared_error
import joblib
import os
 
# ==========================================
# ENTRENAMIENTO DEL MODELO PREDICTIVO
# ==========================================
# Este script entrena tres modelos para imputación:
# - modelo_voto_anterior.joblib: predice voto anterior
# - modelo_voto.joblib: predice intención de voto
# - modelo_imagen.joblib: predice imagen del candidato
# ==========================================
 
RUTA_ENCUESTA = "encuesta_ficticia_nacional_2.csv"
if not os.path.exists(RUTA_ENCUESTA):
    raise FileNotFoundError(f"No se encontró el archivo: {RUTA_ENCUESTA}")
df = pd.read_csv(RUTA_ENCUESTA)
df['nivel_educativo'] = df['nivel_educativo'].astype(str).str.strip().str.lower()
df['sexo'] = df['sexo'].astype(str).str.strip().str.lower()
df['voto'] = df['voto'].astype(str).str.strip().str.lower()
df['voto_anterior'] = df['voto_anterior'].astype(str).str.strip().str.lower()
 
# ── Modelo 1: voto_anterior ──────────────────────────────────────────────────
print("\n=== ENTRENANDO MODELO VOTO_ANTERIOR ===")
df_va = df[df['voto_anterior'].notna() & (df['voto_anterior'] != 'nan')].copy()
features_va = ['edad', 'sexo', 'nivel_educativo']
X_va = pd.get_dummies(df_va[features_va], drop_first=True)
y_va = df_va['voto_anterior']
X_train_va, X_test_va, y_train_va, y_test_va = train_test_split(X_va, y_va, test_size=0.3, random_state=42, stratify=y_va)
model_va = LogisticRegression(solver='newton-cg', max_iter=2000)
model_va.fit(X_train_va, y_train_va)
print("Accuracy:", accuracy_score(y_test_va, model_va.predict(X_test_va)))
joblib.dump(model_va, "modelo_voto_anterior.joblib")
joblib.dump(X_train_va.columns.tolist(), "features_voto_anterior.joblib")
print("Guardado: modelo_voto_anterior.joblib")
 
# ── Modelo 2: voto ──────────────────────────────────────────────────────────
print("\n=== ENTRENANDO MODELO VOTO ===")
df_v = df[df['voto'].notna() & (df['voto'] != 'nan')].copy()
features_v = ['edad', 'sexo', 'nivel_educativo']
X_v = pd.get_dummies(df_v[features_v], drop_first=True)
y_v = df_v['voto']
X_train_v, X_test_v, y_train_v, y_test_v = train_test_split(X_v, y_v, test_size=0.3, random_state=42, stratify=y_v)
model_v = LogisticRegression(solver='newton-cg', max_iter=2000)
model_v.fit(X_train_v, y_train_v)
print("Accuracy:", accuracy_score(y_test_v, model_v.predict(X_test_v)))
joblib.dump(model_v, "modelo_voto.joblib")
joblib.dump(X_train_v.columns.tolist(), "features_voto.joblib")
print("Guardado: modelo_voto.joblib")
 
# ── Modelo 3: imagen del candidato ──────────────────────────────────────────
print("\n=== ENTRENANDO MODELO IMAGEN ===")
df_img = df[df['imagen_del_candidato'].notna()].copy()
features_img = ['edad', 'sexo', 'nivel_educativo']
X_img = pd.get_dummies(df_img[features_img], drop_first=True)
y_img = df_img['imagen_del_candidato']
X_train_img, X_test_img, y_train_img, y_test_img = train_test_split(X_img, y_img, test_size=0.3, random_state=42)
model_img = LinearRegression()
model_img.fit(X_train_img, y_train_img)
r2 = r2_score(y_test_img, model_img.predict(X_test_img))
print("R²:", r2)
print("MAE:", mean_absolute_error(y_test_img, model_img.predict(X_test_img)))
joblib.dump(model_img, "modelo_imagen.joblib")
joblib.dump(X_train_img.columns.tolist(), "features_imagen.joblib")
print("Guardado: modelo_imagen.joblib")
 
print("\nTodos los modelos entrenados y guardados correctamente.")
