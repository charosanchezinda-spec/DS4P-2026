import numpy as np
import pandas as pd
def calcular_tracking(df, ventana_col, peso_col):
  #Tracking imagen  
  tracking_imagen = (
        df.groupby(ventana_col)
        .apply(lambda g: np.average(g['imagen_del_candidato'], weights=g[peso_col]))
        .reset_index(name='trackeo')
    )
    # Tracking voto
    candidatos = df['voto'].unique().tolist()
    for c in candidatos:
        df[f'vota_{c}'] = (df['voto'] == c).astype(int)
    tracking_voto = (
        df.groupby(ventana_col)
        .apply(lambda g: pd.Series({f"Vota_{c}": np.average(g[f'vota_{c}'], weights=g[peso_col]) * 100
            for c in candidatos
        }))
        .reset_index()
    )
    return tracking_imagen, tracking_voto
