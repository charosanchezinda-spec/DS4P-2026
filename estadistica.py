import numpy as np
import pandas as pd
from scipy.stats import ttest_ind, mannwhitneyu, levene

def weighted_std(values, weights):
    values   = np.array(values)
    weights  = np.array(weights)
    average  = np.average(values, weights=weights)
    variance = np.average((values - average)**2, weights=weights)
    return np.sqrt(variance)

def n_efectivo(weights):
    w = np.array(weights)
    return (w.sum()**2) / ((w**2).sum())

def calcular_intervalos(df, tipo_track):
    candidatos = df['voto'].unique().tolist()

    def margen_error_voto(g, peso_col):
        data  = {}
        pesos = g[peso_col]
        n_eff = n_efectivo(pesos)
        for c in candidatos:
            col = f"vota_{c}"
            if col not in g.columns:
                continue
            p   = np.average(g[col], weights=pesos)
            SE  = np.sqrt(p * (1 - p) / n_eff)
            MOE = 1.96 * SE
            data[f"Vota_{c}"]     = p * 100
            data[f"Vota_{c}_MOE"] = MOE * 100
            data[f"Vota_{c}_LI"]  = (p - MOE) * 100
            data[f"Vota_{c}_LS"]  = (p + MOE) * 100
        return pd.Series(data)

    def margen_error_imagen(g, peso_col):
        pesos   = g[peso_col]
        valores = g['imagen_del_candidato']
        n_eff   = n_efectivo(pesos)
        media   = np.average(valores, weights=pesos)
        sd_w    = weighted_std(valores, pesos)
        SE      = sd_w / np.sqrt(n_eff)
        MOE     = 1.96 * SE
        return pd.Series({
            'imagen_media': media,
            'MOE_95':       MOE,
            'LI_95':        media - MOE,
            'LS_95':        media + MOE
        })

    if tipo_track == "d":
        margen_de_error_img = df.groupby('Ventana_D').apply(lambda g: margen_error_imagen(g, 'peso_d')).reset_index()
        margen_de_error_vot = df.groupby('Ventana_D').apply(lambda g: margen_error_voto(g, 'peso_d')).reset_index()
        print("=== Intervalos de confianza — Imagen (diario) (95%) ===")
        print(margen_de_error_img)
        print("=== Intervalos de confianza — Voto (diario) (95%) ===")
        print(margen_de_error_vot)
    elif tipo_track == "s":
        margen_de_error_img = df.groupby('Ventana_S').apply(lambda g: margen_error_imagen(g, 'peso_s')).reset_index()
        margen_de_error_vot = df.groupby('Ventana_S').apply(lambda g: margen_error_voto(g, 'peso_s')).reset_index()
        print("=== Intervalos de confianza — Imagen (semanal) (95%) ===")
        print(margen_de_error_img)
        print("=== Intervalos de confianza — Voto (semanal) (95%) ===")
        print(margen_de_error_vot)
    else:
        margen_de_error_img = df.groupby('Ventana_M').apply(lambda g: margen_error_imagen(g, 'peso_m')).reset_index()
        margen_de_error_vot = df.groupby('Ventana_M').apply(lambda g: margen_error_voto(g, 'peso_m')).reset_index()
        print("=== Intervalos de confianza — Imagen (mensual) (95%) ===")
        print(margen_de_error_img)
        print("=== Intervalos de confianza — Voto (mensual) (95%) ===")
        print(margen_de_error_vot)

    return margen_de_error_img, margen_de_error_vot

def test_hipotesis(df):
    primera_ventana = df['Ventana_S'].min()
    ultima_ventana  = df['Ventana_S'].max()
    img_ini = df.loc[df['Ventana_S'] == primera_ventana, 'imagen_del_candidato']
    img_fin = df.loc[df['Ventana_S'] == ultima_ventana,  'imagen_del_candidato']
    n_ini, n_fin = len(img_ini), len(img_fin)

    if n_ini == 0 or n_fin == 0:
        raise ValueError("Una de las ventanas no tiene datos de imagen.")

    alpha = 0.05
    print("TEST DE HIPÓTESIS SOBRE CAMBIO EN LA IMAGEN")
    print("Ventana inicial:", primera_ventana, " | n =", n_ini)
    print("Ventana final  :", ultima_ventana,  " | n =", n_fin)

    normalidad       = (n_ini >= 30) and (n_fin >= 30)
    stat_lev, p_lev  = levene(img_ini, img_fin, center='mean')
    homocedasticidad = (p_lev >= alpha)
    print("Test de Levene — p-value:", p_lev)

    if normalidad and homocedasticidad:
        tstat, pval = ttest_ind(img_ini, img_fin, equal_var=True)
    elif normalidad and not homocedasticidad:
        tstat, pval = ttest_ind(img_ini, img_fin, equal_var=False)
    else:
        tstat, pval = mannwhitneyu(img_ini, img_fin, alternative='two-sided')

    print("Estadístico:", tstat)
    print("p-value:", pval)
    if pval < alpha:
        print("Conclusión: Se RECHAZA H0")
    else:
        print("Conclusión: NO se rechaza H0")

    return tstat, pval
