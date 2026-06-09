import os
import warnings
import logging
from dotenv import load_dotenv

warnings.filterwarnings("ignore", category=FutureWarning)
logging.getLogger("balance").setLevel(logging.ERROR)

load_dotenv()

from carga       import cargar_datos
from limpieza    import limpiar, normalizar
from imputacion  import imputar
from ventanas    import crear_ventanas
from ponderacion import elegir_targets, ponderar
from tracking    import tracking_diario, tracking_semanal, tracking_mensual
from estadistica import calcular_intervalos, test_hipotesis
from base_datos  import registrar_corrida, registrar_metricas, get_db

if __name__ == "__main__":
    # Paso 1: cargar
    ruta = input("Ruta del archivo de encuesta: ").strip()
    df = cargar_datos(ruta)
    if df is None:
        raise SystemExit("No se pudo cargar el archivo.")

    # Paso 2: elegir población (necesaria antes de limpiar para detectar municipios GBA)
    from ponderacion import POBLACIONES
    print("Poblaciones disponibles:")
    for clave in POBLACIONES:
        print(clave)
    while True:
        poblacion = input("Escribí la población: ").strip().lower()
        if poblacion in POBLACIONES:
            break
        print("Población no reconocida. Por favor elegí una de la lista.")

    # Paso 3: limpiar y normalizar
    df = limpiar(df)
    df, hay_municipios_bsas = normalizar(df, poblacion)

    # Paso 4: imputar
    df = imputar(df)

    # Paso 5: crear ventanas
    df = crear_ventanas(df)

    # Paso 6: obtener targets y ponderar
    from ponderacion import obtener_targets_desde_censo
    targets, df = obtener_targets_desde_censo(poblacion, hay_municipios_bsas, df)
    df, target_df, vars_rake = ponderar(df, targets)

    # Paso 7: elegir ventana y generar tracking
    tipo_track = input("Elegí el tipo de tracking (d=diario, s=semanal, m=mensual): ").strip().lower()
    if tipo_track == "d":
        print("Generando TRACKING DIARIO...")
        tracking_diario(df, vars_rake, target_df)
    elif tipo_track == "s":
        print("Generando TRACKING SEMANAL...")
        tracking_semanal(df, vars_rake, target_df)
    elif tipo_track == "m":
        print("Generando TRACKING MENSUAL...")
        tracking_mensual(df, vars_rake, target_df)
    else:
        raise SystemExit("Opción inválida. Elegí d, s o m.")

    # Paso 8: intervalos de confianza y test
    calcular_intervalos(df, tipo_track)
    test_hipotesis(df)

    # Paso 9: registrar en base de datos
    from balance import Sample
    sample_df_total = df[vars_rake].copy().reset_index(drop=True)
    sample_df_total.insert(0, '_id', range(len(sample_df_total)))
    t_df_total = target_df.copy()
    if '_id' not in t_df_total.columns:
        t_df_total.insert(0, '_id', range(len(t_df_total)))
    s_total = Sample.from_frame(sample_df_total, id_column='_id', outcome_columns=[])
    t_total = Sample.from_frame(t_df_total,      id_column='_id', outcome_columns=[])
    adjusted_total = s_total.set_target(t_total).adjust(method='rake').trim(ratio=3)

    pesos_w = adjusted_total.weights().df['weight']
    deff    = 1 + (pesos_w.var() / pesos_w.mean()**2)
    ess     = (pesos_w.sum()**2) / (pesos_w**2).sum()
    essp    = ess / len(pesos_w)

    db = next(get_db())
    corrida_id = registrar_corrida(db, poblacion, len(df), vars_rake)
    registrar_metricas(db, corrida_id, deff, ess, essp, pesos_w.max(), pesos_w.min())
    db.close()
    print(f"Corrida registrada en la base de datos (id={corrida_id})")
