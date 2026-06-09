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


    return grupodef aplicar_rake_diario(grupo):
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
