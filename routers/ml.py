from fastapi import APIRouter, HTTPException, Depends
from security import verificar_api_key
import pandas as pd
import joblib
import os

router = APIRouter()

def _cargar_modelo(nombre_modelo, nombre_features):
    ruta_modelo = os.path.join(os.path.dirname(__file__), "..", nombre_modelo)
    ruta_features = os.path.join(os.path.dirname(__file__), "..", nombre_features)
    if os.path.exists(ruta_modelo) and os.path.exists(ruta_features):
        return joblib.load(ruta_modelo), joblib.load(ruta_features)
    return None, None

modelo_voto_anterior, features_voto_anterior = _cargar_modelo("modelo_voto_anterior.joblib", "features_voto_anterior.joblib")
modelo_voto, features_voto = _cargar_modelo("modelo_voto.joblib", "features_voto.joblib")
modelo_imagen, features_imagen = _cargar_modelo("modelo_imagen.joblib", "features_imagen.joblib")
 
@router.get("/predecir", dependencies=[Depends(verificar_api_key)])
def predecir(edad: int, sexo: str, estrato: str, nivel_educativo: str, voto_anterior: str, voto: str):
    if modelo_voto is None or modelo_voto_anterior is None or modelo_imagen is None:
        raise HTTPException(
            status_code=503,
            detail="Modelos no disponibles. Corra el proceso de entrenamiento primero."
        )
    datos = pd.DataFrame([{
        "edad": edad,
        "sexo": sexo,
        "estrato": estrato,
        "nivel_educativo": nivel_educativo,
        "voto_anterior": voto_anterior,
        "voto": voto,
    }])
    datos_dummies = pd.get_dummies(datos, drop_first=True)
    X_va = datos_dummies.reindex(columns=features_voto_anterior, fill_value=0)
    prediccion_va = modelo_voto_anterior.predict(X_va)[0]
    X_voto = datos_dummies.reindex(columns=features_voto, fill_value=0)
    prediccion_voto = modelo_voto.predict(X_voto)[0]
    probabilidades = dict(zip(modelo_voto.classes_, modelo_voto.predict_proba(X_voto)[0].tolist()))
    X_img = datos_dummies.reindex(columns=features_imagen, fill_value=0)
    prediccion_imagen = round(float(modelo_imagen.predict(X_img)[0]), 1)
    return {
        "prediccion_voto": prediccion_voto,
        "probabilidades_voto": probabilidades,
        "prediccion_voto_anterior": prediccion_va,
        "prediccion_imagen": prediccion_imagen,
        "nota": "Modelo demostrativo entrenado con datos acumulados. Las versiones se actualizan al procesar nuevas encuestas.",
    }

