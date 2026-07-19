import hashlib
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
# ==========================================
# RUTAS Y CARGA DE ARCHIVOS
# ==========================================
DIRECTORIO_BASE = os.path.dirname(os.path.abspath(__file__))
def _construir_ruta(nombre_archivo):
    return os.path.join(DIRECTORIO_BASE, nombre_archivo)
def _cargar(nombre_archivo):
    ruta = _construir_ruta(nombre_archivo)
    if os.path.exists(ruta):
        return joblib.load(ruta)
    return None
# Modelo, features y datos históricos de voto anterior
_modelo_va = _cargar("modelo_voto_anterior.joblib")
_features_va = _cargar("features_voto_anterior.joblib")
_datos_va = _cargar("datos_voto_anterior.joblib")
# Modelo, features y datos históricos de voto actual
_modelo_v = _cargar("modelo_voto.joblib")
_features_v = _cargar("features_voto.joblib")
_datos_v = _cargar("datos_voto.joblib")
# Modelo, features y datos históricos de imagen
_modelo_img = _cargar("modelo_imagen.joblib")
_features_img = _cargar("features_imagen.joblib")
_datos_img = _cargar("datos_imagen.joblib")

# ==========================================
# IDENTIFICACIÓN DE ENCUESTAS
# ==========================================
def _generar_id_encuesta(df):
    df_hash = df.copy()
    df_hash = df_hash.reindex(sorted(df_hash.columns), axis=1)
    hashes_filas = pd.util.hash_pandas_object(df_hash,index=False).to_numpy()
    hashes_filas = np.sort(hashes_filas)
    return hashlib.sha256(hashes_filas.tobytes()).hexdigest()

# ==========================================
# EVALUACIÓN DE MODELOS
# ==========================================
def evaluar_modelos(df):
    print("Evaluación de modelos para imputación:regresión logística y regresión lineal\n")
    print("EVALUACIÓN VOTO_ANTERIOR\n")
    columnas_va = ["edad", "sexo", "nivel_educativo", "estrato", "voto_anterior",
    ]
    df_eval_va = (df[columnas_va].dropna().copy())
    if (len(df_eval_va) >= 10 and df_eval_va["voto_anterior"].nunique() >= 2):
        features_va_eval = ["edad", "sexo", "nivel_educativo", "estrato"]
        X_va = pd.get_dummies(df_eval_va[features_va_eval],drop_first=True,)
        y_va = df_eval_va["voto_anterior"].astype("category")
        y_va_num = y_va.cat.codes
        mapeo_va = dict(enumerate(y_va.cat.categories))
        try:
            (X_train_va, X_test_va, y_train_va, y_test_va) = train_test_split(X_va, y_va_num, test_size=0.3, random_state=42,stratify=y_va_num,)
            model_va_eval = LogisticRegression(solver="newton-cg",max_iter=2000)
            model_va_eval.fit(X_train_va, y_train_va)
            y_pred_va = model_va_eval.predict(X_test_va)
            print("Accuracy:", accuracy_score(y_test_va, y_pred_va))
            labels_va = np.unique(np.concatenate([y_test_va, y_pred_va]))
            names_va = [str(mapeo_va[i])
                for i in labels_va
            ]
            print("\nClassification report:")
            print(classification_report(y_test_va,y_pred_va, labels=labels_va, target_names=names_va,zero_division=0,))
            print("\nMatriz de confusión:")
            print(confusion_matrix(y_test_va, y_pred_va, labels=labels_va))
        except ValueError as error:
            print("No fue posible evaluar voto_anterior:", error)
    else:
        print("No hay suficientes casos o categorías para evaluar voto_anterior.")

    # ------------------------------------------
    # VOTO ACTUAL
    # ------------------------------------------
    print("\nEVALUACIÓN VOTO\n")
    columnas_voto = ["edad", "sexo", "nivel_educativo", "estrato", "voto_anterior","voto"]
    df_eval_v = (df[columnas_voto].dropna().copy())
    if (len(df_eval_v) >= 10 and df_eval_v["voto"].nunique() >= 2):
        features_voto_eval = ["edad", "sexo", "nivel_educativo", "estrato", "voto_anterior",]
        X_v = pd.get_dummies(df_eval_v[features_voto_eval], drop_first=True)
        y_v = df_eval_v["voto"].astype("category")
        y_v_num = y_v.cat.codes
        mapeo_v = dict(enumerate(y_v.cat.categories))
        try:
            (X_train_v, X_test_v, y_train_v, y_test_v) = train_test_split(X_v, y_v_num, test_size=0.3, random_state=42, stratify=y_v_num)
            model_v_eval = LogisticRegression(solver="newton-cg", max_iter=2000)
            model_v_eval.fit(X_train_v, y_train_v)
            y_pred_v = model_v_eval.predict(X_test_v)
            print("Accuracy:", accuracy_score(y_test_v, y_pred_v))
            labels_v = np.unique(np.concatenate([y_test_v, y_pred_v]))
            names_v = [str(mapeo_v[i])
                for i in labels_v
            ]
            print("\nClassification report:")
            print(classification_report(y_test_v, y_pred_v, labels=labels_v, target_names=names_v, zero_division=0))
            print("\nMatriz de confusión:")
            print(confusion_matrix(y_test_v, y_pred_v, labels=labels_v))
        except ValueError as error:
            print("No fue posible evaluar voto:", error)
    else:
        print("No hay suficientes casos o categorías para evaluar voto.")

    # ------------------------------------------
    # IMAGEN DEL CANDIDATO
    # ------------------------------------------

    print("\nEVALUACIÓN IMAGEN_DEL_CANDIDATO\n")
    columnas_img = ["edad", "sexo", "nivel_educativo", "estrato", "voto", "voto_anterior", "imagen_del_candidato"]
    df_eval_img = (df[columnas_img].dropna().copy())
    if len(df_eval_img) >= 10:
        features_img_eval = ["edad","sexo","nivel_educativo", "estrato", "voto","voto_anterior"]
        X_img = pd.get_dummies(df_eval_img[features_img_eval],drop_first=True)
        y_img = df_eval_img["imagen_del_candidato"]
        (X_train_img, X_test_img, y_train_img, y_test_img) = train_test_split(X_img, y_img, test_size=0.3, random_state=42,)
        model_img_eval = LinearRegression()
        model_img_eval.fit(X_train_img, y_train_img)
        y_pred_img = model_img_eval.predict(X_test_img)
        r2_img = r2_score(y_test_img, y_pred_img)
        print("MAE:", mean_absolute_error(y_test_img, y_pred_img))
        print("RMSE:", np.sqrt(mean_squared_error(y_test_img, y_pred_img)))
        print("R²:", r2_img)
        return r2_img
    print("No hay suficientes casos completos para evaluar imagen_del_candidato.")
    return float("-inf")

# ==========================================
# REENTRENAMIENTO ACUMULATIVO
# ==========================================

def _actualizar_modelo( df_nuevo, variable_objetivo, variables_predictoras, datos_acumulados, nombre_modelo, nombre_features, nombre_datos, encuesta_id, tipo="categorica"):
    columnas_necesarias = (variables_predictoras + [variable_objetivo])
    df_valido = (df_nuevo[columnas_necesarias].dropna().copy())
    if df_valido.empty:
        print(f"No hay casos observados suficientes para actualizar {nombre_modelo}.")
        return None, None, datos_acumulados
    df_valido["_encuesta_id"] = encuesta_id
    encuesta_ya_incorporada = (datos_acumulados is not None and "_encuesta_id" in datos_acumulados.columns and encuesta_id in set(datos_acumulados["_encuesta_id"].dropna()))
    if encuesta_ya_incorporada:
        print(f"La encuesta ya estaba incorporada en {nombre_datos}. No se duplicaron sus casos.")
        df_combinado = datos_acumulados.copy()
    elif datos_acumulados is not None:
        df_combinado = pd.concat([datos_acumulados, df_valido], ignore_index=True)
    else:
        df_combinado = df_valido
    columnas_entrenamiento = [columna
        for columna in df_combinado.columns
        if columna != "_encuesta_id"
    ]
    df_entrenamiento = (df_combinado[columnas_entrenamiento].dropna(subset=variables_predictoras + [variable_objetivo]).copy())
    if df_entrenamiento.empty:
        print(f"No quedaron casos válidos para entrenar {nombre_modelo}.")
        return None, None, df_combinado
    X = pd.get_dummies(df_entrenamiento[variables_predictoras], drop_first=True)
    y = df_entrenamiento[variable_objetivo]
    if tipo == "categorica":
        if y.nunique() < 2:
            print(f"No se pudo actualizar {nombre_modelo}: la variable objetivo tiene una sola categoría.")
            return None, X.columns.tolist(), df_combinado
        modelo = LogisticRegression( solver="newton-cg", max_iter=2000)
    elif tipo == "numerica":
        modelo = LinearRegression()
    else:
        raise ValueError("El tipo de modelo debe ser'categorica' o 'numerica'."
        )
    modelo.fit(X, y)
    features = X.columns.tolist()
    ruta_modelo = _construir_ruta(nombre_modelo)
    ruta_features = _construir_ruta(nombre_features)
    ruta_datos = _construir_ruta(nombre_datos)
    joblib.dump(modelo, ruta_modelo)
    joblib.dump(features, ruta_features)
    joblib.dump(df_combinado, ruta_datos)
    print(f"Modelo {nombre_modelo} actualizado con {len(df_entrenamiento)} casos acumulados.")
    return modelo, features, df_combinado

# ==========================================
# IMPUTACIÓN CATEGÓRICA
# ==========================================

def _imputar_categorica(df, variable_objetivo, variables_predictoras, modelo_base=None, features_base=None):
    mascara_faltantes = df[variable_objetivo].isna()
    if not mascara_faltantes.any():
        return df
    df_miss = df.loc[mascara_faltantes].copy()
    if modelo_base is not None and features_base is not None:
        print(f"Usando el modelo acumulado para imputar {variable_objetivo}.")
        X_miss = pd.get_dummies(df_miss[variables_predictoras], drop_first=True)
        X_miss = X_miss.reindex(columns=features_base, fill_value=0)
        predicciones = modelo_base.predict(X_miss)
    else:
        columnas_necesarias = (variables_predictoras + [variable_objetivo])
        df_full = (df.loc[~mascara_faltantes, columnas_necesarias].dropna().copy())
        if df_full.empty:
            raise ValueError(f"No hay casos completos para imputar {variable_objetivo}.")
        if df_full[variable_objetivo].nunique() < 2:
            raise ValueError(f"No se puede entrenar el modelo para {variable_objetivo}: solo existe una categoría.")
        print(f"No existe un modelo acumulado. Entrenando un modelo temporal para {variable_objetivo}.")
        X_full = pd.get_dummies(df_full[variables_predictoras],drop_first=True)
        y_full = df_full[variable_objetivo]
        modelo_temporal = LogisticRegression(solver="newton-cg", max_iter=2000)
        modelo_temporal.fit(X_full, y_full)
        X_miss = pd.get_dummies(df_miss[variables_predictoras], drop_first=True)
        X_miss = X_miss.reindex(columns=X_full.columns, fill_value=0)
        predicciones = modelo_temporal.predict(X_miss)
    df.loc[mascara_faltantes, variable_objetivo] = predicciones
    return df
# ==========================================
# IMPUTACIÓN NUMÉRICA
# ==========================================

def _imputar_numerica(df, variable_objetivo, variables_predictoras, r2_img, modelo_base=None, features_base=None):
    mascara_faltantes = df[variable_objetivo].isna()
    if not mascara_faltantes.any():
        return df
    if r2_img <= 0.15:
        print("El modelo de IMAGEN no supera el umbral de calidad. Se utilizará la mediana.")
        mediana = df[variable_objetivo].median()
        df[variable_objetivo] = (df[variable_objetivo].fillna(mediana))
        return df
    print("El modelo de IMAGEN supera el umbral. Se utilizará regresión lineal.")
    df_miss = df.loc[mascara_faltantes].copy()
    if modelo_base is not None and features_base is not None:
        print(f"Usando el modelo acumulado para imputar {variable_objetivo}.")
        X_miss = pd.get_dummies(df_miss[variables_predictoras], drop_first=True)
        X_miss = X_miss.reindex(columns=features_base, fill_value=0)
        predicciones = modelo_base.predict(X_miss)
    else:
        columnas_necesarias = (variables_predictoras + [variable_objetivo])
        df_full = (df.loc[~mascara_faltantes, columnas_necesarias].dropna().copy())
        if df_full.empty:
            print("No hay casos completos para regresión. Se utilizará la mediana.")
            mediana = df[variable_objetivo].median()
            df[variable_objetivo] = (df[variable_objetivo].fillna(mediana))
            return df
        X_full = pd.get_dummies(df_full[variables_predictoras], drop_first=True)
        y_full = df_full[variable_objetivo]
        modelo_temporal = LinearRegression()
        modelo_temporal.fit(X_full, y_full)
        X_miss = pd.get_dummies(df_miss[variables_predictoras], drop_first=True)
        X_miss = X_miss.reindex(columns=X_full.columns, fill_value=0)
        predicciones = modelo_temporal.predict(X_miss)
    df.loc[mascara_faltantes, variable_objetivo] = predicciones
    return df

# ==========================================
# FUNCIÓN PRINCIPAL
# ==========================================

def imputar(df):
    global _modelo_va, _features_va, _datos_va
    global _modelo_v, _features_v, _datos_v
    global _modelo_img, _features_img, _datos_img
    df_observado = df.copy(deep=True)
    encuesta_id = _generar_id_encuesta(df_observado)
    r2_img = evaluar_modelos(df_observado)
    # ------------------------------------------
    # IMPUTACIÓN PARA EL TRACKING
    # ------------------------------------------
    df = _imputar_categorica(df=df, variable_objetivo="voto_anterior",variables_predictoras=["edad", "sexo", "estrato", "nivel_educativo",], modelo_base=_modelo_va, features_base=_features_va,)
    df = _imputar_categorica(df=df, variable_objetivo="voto", variables_predictoras=["edad", "sexo", "estrato", "nivel_educativo", "voto_anterior"], modelo_base=_modelo_v, features_base=_features_v)
    df = _imputar_numerica(df=df, variable_objetivo="imagen_del_candidato", variables_predictoras=[ "edad", "sexo", "estrato", "nivel_educativo", "voto", "voto_anterior"], r2_img=r2_img, modelo_base=_modelo_img, features_base=_features_img)
    df["imagen_del_candidato"] = (df["imagen_del_candidato"].clip(lower=0, upper=100))
    df["voto"] = (df["voto"].astype(str).str.strip().str.lower())
    df["voto_anterior"] = (df["voto_anterior"].astype(str).str.strip().str.lower())

    # ------------------------------------------
    # REENTRENAMIENTO ACUMULATIVO
    # ------------------------------------------

    print("\nActualizando modelos con los valores observados de la encuesta nueva...")
    (
        nuevo_modelo_va,
        nuevas_features_va,
        nuevos_datos_va,
    ) = _actualizar_modelo(
        df_nuevo=df_observado,
        variable_objetivo="voto_anterior",
        variables_predictoras=[
            "edad",
            "sexo",
            "estrato",
            "nivel_educativo",
        ],
        datos_acumulados=_datos_va,
        nombre_modelo="modelo_voto_anterior.joblib",
        nombre_features="features_voto_anterior.joblib",
        nombre_datos="datos_voto_anterior.joblib",
        encuesta_id=encuesta_id,
        tipo="categorica",
    )
    if nuevo_modelo_va is not None:
        _modelo_va = nuevo_modelo_va
        _features_va = nuevas_features_va
    _datos_va = nuevos_datos_va
    (
        nuevo_modelo_v,
        nuevas_features_v,
        nuevos_datos_v,
    ) = _actualizar_modelo(
        df_nuevo=df_observado,
        variable_objetivo="voto",
        variables_predictoras=[
            "edad",
            "sexo",
            "estrato",
            "nivel_educativo",
            "voto_anterior",
        ],
        datos_acumulados=_datos_v,
        nombre_modelo="modelo_voto.joblib",
        nombre_features="features_voto.joblib",
        nombre_datos="datos_voto.joblib",
        encuesta_id=encuesta_id,
        tipo="categorica",
    )
    if nuevo_modelo_v is not None:
        _modelo_v = nuevo_modelo_v
        _features_v = nuevas_features_v
    _datos_v = nuevos_datos_v
    (
        nuevo_modelo_img,
        nuevas_features_img,
        nuevos_datos_img,
    ) = _actualizar_modelo(
        df_nuevo=df_observado,
        variable_objetivo="imagen_del_candidato",
        variables_predictoras=[
            "edad",
            "sexo",
            "estrato",
            "nivel_educativo",
            "voto",
            "voto_anterior",
        ],
        datos_acumulados=_datos_img,
        nombre_modelo="modelo_imagen.joblib",
        nombre_features="features_imagen.joblib",
        nombre_datos="datos_imagen.joblib",
        encuesta_id=encuesta_id,
        tipo="numerica",
    )
    if nuevo_modelo_img is not None:
        _modelo_img = nuevo_modelo_img
        _features_img = nuevas_features_img
    _datos_img = nuevos_datos_img
    print("\nPorcentaje de valores faltantes post imputación:")
    print(df.isna().mean() * 100)
    print("\nLos modelos fueron actualizados en disco. Para que el endpoint /predecir use estas versiones, reinicie la API FastAPI.")
    return df
