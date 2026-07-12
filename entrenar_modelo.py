import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# ==========================================
# ENTRENAMIENTO DEL MODELO PREDICTIVO
# ==========================================
# Este script entrena un modelo de clasificación para predecir
# la intención de voto en base al perfil sociodemográfico.
#
# Es un modelo demostrativo entrenado con datos ficticios.
# Puede reentrenarse con encuestas reales para mejorar su precisión.
# La estrategia óptima de features depende de cada dataset.
# ==========================================

RUTA_ENCUESTA = "encuesta_ficticia_nacional_2.csv"
RUTA_MODELO   = "modelo_voto.joblib"
RUTA_FEATURES = "features_voto.joblib"

if not os.path.exists(RUTA_ENCUESTA):
    raise FileNotFoundError(f"No se encontró el archivo: {RUTA_ENCUESTA}")

df = pd.read_csv(RUTA_ENCUESTA)
df = df.dropna(subset=['voto'])
df['nivel_educativo'] = df['nivel_educativo'].astype(str).str.strip().str.lower()
df['sexo']            = df['sexo'].astype(str).str.strip().str.lower()

features = ['edad', 'sexo', 'nivel_educativo']
X = pd.get_dummies(df[features], drop_first=True)
y = df['voto']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

model = LogisticRegression(solver='newton-cg', max_iter=2000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification report:")
print(classification_report(y_test, y_pred, zero_division=0))

joblib.dump(model, RUTA_MODELO)
joblib.dump(X_train.columns.tolist(), RUTA_FEATURES)
print("\nModelo guardado en:", RUTA_MODELO)
print("Features guardadas en:", RUTA_FEATURES)
