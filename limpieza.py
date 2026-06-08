import pandas as pd
import numpy as np


GBA_PARTIDOS = {
    "almirante brown", "avellaneda", "berazategui", "berisso",
    "brandsen", "campana", "cañuelas", "ensenada", "escobar",
    "esteban echeverría", "exaltación de la cruz", "ezeiza",
    "florencio varela", "general las heras", "general rodríguez",
    "general san martín", "hurlingham", "ituzaingó", "josé c. paz",
    "la matanza", "lanús", "la plata", "lomas de zamora", "luján",
    "marcos paz", "malvinas argentinas", "merlo", "moreno", "morón",
    "pilar", "presidente perón", "quilmes", "san fernando",
    "san isidro", "san miguel", "san vicente", "tigre",
    "tres de febrero", "vicente lópez"
}


def limpiar(df):
    df = df[~df[['voto', 'imagen_del_candidato']].isna().all(axis=1)]
    df = df[df['edad'] >= 16]
    df = df[~df['encuesta'].duplicated()]
    df = df.dropna(subset=['fecha', 'estrato', 'nivel_educativo', 'sexo', 'edad'])
    df['nivel_educativo'] = df['nivel_educativo'].astype(str).str.strip().str.lower()
    df['nivel_educativo'] = df['nivel_educativo'].replace({'sin estudios': 'prim'})

    def normalizar_nivel_educativo(x):
        for nivel in ["prim", "sec", "terc", "univ", "pos"]:
            if x.startswith(nivel):
                return nivel
        return x

    df['nivel_educativo'] = df['nivel_educativo'].apply(normalizar_nivel_educativo)
    df['sexo'] = df['sexo'].where(df['sexo'].isin(['femenino', 'masculino']), np.nan)
    df = df.dropna(subset=['sexo'])
    df['integrantes_hogar'] = df['integrantes_hogar'].fillna('Desconocido')
    print("Porcentaje de nans previo a la imputación:")
    print(df.isna().mean() * 100)
    return df


def normalizar(df, poblacion):
    df['estrato'] = df['estrato'].astype(str).str.strip().str.lower()
    df['sexo']    = df['sexo'].astype(str).str.strip().str.lower()
    df['edad_cat'] = pd.cut(
        df['edad'],
        bins=[15, 29, 44, 59, 120],
        labels=['16-29', '30-44', '45-59', '60+']
    )

    hay_municipios_bsas = False
    if poblacion == "buenos_aires":
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

    return df, hay_municipios_bsas
