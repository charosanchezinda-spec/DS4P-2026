import pandas as pd
import numpy as np
import requests
from balance.weighting_methods.rake import rake, prepare_marginal_dist_for_raking
from balance import Sample
import os
from dotenv import load_dotenv

load_dotenv()
API_URL = os.getenv("API_URL")
API_KEY = os.getenv("API_KEY")
if not API_URL:
    raise RuntimeError("Falta API_URL en el archivo .env")
if not API_KEY:
    raise RuntimeError("Falta API_KEY en el archivo .env")
HEADERS = {"x-api-key": API_KEY}

try:
    _resp = requests.get(f"{API_URL}/poblaciones", timeout=60)
    _resp.raise_for_status()
    POBLACIONES = _resp.json()["poblaciones"]
except Exception:
    raise RuntimeError("No se pudo obtener el catálogo de poblaciones desde la API.")

def obtener_targets_desde_censo(poblacion, hay_municipios_bsas, df):
    url = f"{API_URL}/targets/{poblacion}"
    print(f"  Consultando API — {POBLACIONES.get(poblacion, poblacion)}...")
    try:
        respuesta = requests.get(url, headers=HEADERS, timeout=60)
        respuesta.raise_for_status()
        targets = respuesta.json()["targets"]
        if poblacion == "buenos_aires" and hay_municipios_bsas:
            resp_estrato = requests.get(f"{API_URL}/estrato-bsas", headers=HEADERS, timeout=10)
            resp_estrato.raise_for_status()
            targets["estrato_bsas"] = resp_estrato.json()["estrato"]
            print("  Variable 'estrato_bsas' agregada (GBA/interior).")
        elif poblacion == "nacional":
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
            respuesta_region = requests.get(f"{API_URL}/region-nacional", headers=HEADERS, timeout=10)
            respuesta_region.raise_for_status()
            targets["region"] = respuesta_region.json()["region"]
            print("Columna region creada para calibración nacional.")
        print(f"  Variables de calibración: {list(targets.keys())}")
        return targets, df
    except requests.exceptions.ConnectionError:
        raise RuntimeError("No se pudo conectar con la API.")

def ponderar(df, targets):
    advertencias = []
    df['peso_d'] = 1
    df['peso_s'] = 1
    df['peso_m'] = 1
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
                advertencias.append(f"ADVERTENCIA ventana {grupo.name}: Deff alto ({round(deff, 2)}). Considere ampliar la ventana.")
            if cv > 80:
                advertencias.append(f"ADVERTENCIA ventana {grupo.name}: CV alto ({round(cv, 1)}%). La muestra tiene perfiles muy subrepresentados.")
        except ValueError as e:
            advertencias.append(f"No se pudo hacer raking en ventana {grupo.name}: {e}")
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
                advertencias.append(f"ADVERTENCIA ventana {grupo.name}: Deff alto ({round(deff, 2)}). Considere ampliar la ventana.")
            if cv > 80:
                advertencias.append(f"ADVERTENCIA ventana {grupo.name}: CV alto ({round(cv, 1)}%). La muestra tiene perfiles muy subrepresentados.")
        except ValueError as e:
            advertencias.append(f"No se pudo hacer raking en ventana {grupo.name}: {e}")
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
                advertencias.append(f"ADVERTENCIA ventana {grupo.name}: Deff alto ({round(deff, 2)}). Considere ampliar la ventana.")
            if cv > 80:
                advertencias.append(f"ADVERTENCIA ventana {grupo.name}: CV alto ({round(cv, 1)}%). La muestra tiene perfiles muy subrepresentados.")
        except ValueError as e:
            advertencias.append(f"No se pudo hacer raking en ventana {grupo.name}: {e}")
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
    return df, target_df, vars_rake, advertencias

def generar_reporte(df, target_df, vars_rake):
    sample_df_total = df[vars_rake].copy().reset_index(drop=True)
    sample_df_total.insert(0, '_id', range(len(sample_df_total)))
    t_df_total = target_df.copy()
    if '_id' not in t_df_total.columns:
        t_df_total.insert(0, '_id', range(len(t_df_total)))
    s_total = Sample.from_frame(sample_df_total, id_column='_id', outcome_columns=[])
    t_total = Sample.from_frame(t_df_total,      id_column='_id', outcome_columns=[])
    adjusted_total = s_total.set_target(t_total).adjust(method='rake').trim(ratio=3)
    print(adjusted_total.summary())
    print(adjusted_total.weights().summary())
    adjusted_total.covars().plot()
    return adjusted_total
