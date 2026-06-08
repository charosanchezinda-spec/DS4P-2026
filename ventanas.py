def crear_ventanas(df):
    df = df.sort_values('fecha')
    df['Ventana_D'] = df['fecha']
    df['Ventana_S'] = df['fecha'].dt.to_period('W')
    df['Ventana_M'] = df['fecha'].dt.to_period('M')
    return df
