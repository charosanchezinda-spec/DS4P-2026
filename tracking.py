# %%
# TRACKING DIARIO
def tracking_diario():
    tracking_imagen_diario = (
        df.groupby('Ventana_D')
        .apply(lambda g: np.average(g['imagen_del_candidato'], weights=g['peso_d']))
        .reset_index(name='trackeo')
    )
    print(tracking_imagen_diario.round(1))
    plt.figure(figsize=(10, 5))
    plt.plot(tracking_imagen_diario['Ventana_D'], tracking_imagen_diario['trackeo'], marker='o')
    plt.xlabel('Ventana (diaria)', fontsize=10)
    plt.ylabel('Imagen promedio', fontsize=10)
    plt.title('Evolución de la imagen del candidato (ventana diaria)', fontsize=16)
    plt.tight_layout()
    plt.show()

    candidatos = df['voto'].unique().tolist()
    for c in candidatos:
        df[f'vota_{c}'] = (df['voto'] == c).astype(int)
    tracking_voto_diario = (
        df.groupby('Ventana_D')
        .apply(lambda g: pd.Series({
            f"Vota_{c}": np.average(g[f'vota_{c}'], weights=g['peso_d']) * 100
            for c in candidatos
        }))
        .reset_index()
    )
    print(tracking_voto_diario.round(1))
    cols_voto = [col for col in tracking_voto_diario.columns if col.startswith('Vota_')]
    tracking_voto_diario.set_index('Ventana_D')[cols_voto].plot(figsize=(10, 5))
    plt.xlabel('Ventana (diaria)', fontsize=10)
    plt.ylabel('Intención de voto (%)', fontsize=10)
    plt.title('Tracking de intención de voto (ventana diaria)', fontsize=16)
    plt.grid(alpha=0.3)
    plt.legend(title="Candidato")
    plt.tight_layout()
    plt.show()

    sample_df_total = df[vars_rake].copy().reset_index(drop=True)
    sample_df_total.insert(0, '_id', range(len(sample_df_total)))
    t_df_total = target_df.copy()
    if '_id' not in t_df_total.columns:
        t_df_total.insert(0, '_id', range(len(t_df_total)))
    s_total = Sample.from_frame(sample_df_total, id_column='_id', outcome_columns=[])
    t_total = Sample.from_frame(t_df_total, id_column='_id', outcome_columns=[])
    adjusted_total = s_total.set_target(t_total).adjust(method='rake').trim(ratio=3)
    print(adjusted_total.summary())
    print(adjusted_total.weights().summary())
    adjusted_total.covars().plot()

# %%
# TRACKING SEMANAL
def tracking_semanal():
    tracking_imagen_semanal = (
        df.groupby('Ventana_S')
        .apply(lambda g: np.average(g['imagen_del_candidato'], weights=g['peso_s']))
        .reset_index(name='trackeo')
    )
    print(tracking_imagen_semanal.round(1))
    plt.figure(figsize=(10, 5))
    tracking_imagen_semanal.set_index('Ventana_S')['trackeo'].plot(marker='o')
    plt.xlabel('Ventana (semanal)', fontsize=10)
    plt.ylabel('Imagen promedio', fontsize=10)
    plt.title('Evolución de la imagen del candidato (ventana semanal)', fontsize=16)
    plt.tight_layout()
    plt.show()

    candidatos = df['voto'].unique().tolist()
    for c in candidatos:
        df[f'vota_{c}'] = (df['voto'] == c).astype(int)
    tracking_voto_semanal = (
        df.groupby('Ventana_S')
        .apply(lambda g: pd.Series({
            f"Vota_{c}": np.average(g[f'vota_{c}'], weights=g['peso_s']) * 100
            for c in candidatos
        }))
        .reset_index()
    )
    print(tracking_voto_semanal.round(1))
    cols_voto = [col for col in tracking_voto_semanal.columns if col.startswith('Vota_')]
    tracking_voto_semanal.set_index('Ventana_S')[cols_voto].plot(figsize=(10, 5))
    plt.xlabel('Ventana (semanal)', fontsize=10)
    plt.ylabel('Intención de voto (%)', fontsize=10)
    plt.title('Tracking de intención de voto (ventana semanal)', fontsize=16)
    plt.grid(alpha=0.3)
    plt.legend(title="Candidato")
    plt.tight_layout()
    plt.show()

    sample_df_total = df[vars_rake].copy().reset_index(drop=True)
    sample_df_total.insert(0, '_id', range(len(sample_df_total)))
    t_df_total = target_df.copy()
    if '_id' not in t_df_total.columns:
        t_df_total.insert(0, '_id', range(len(t_df_total)))
    s_total = Sample.from_frame(sample_df_total, id_column='_id', outcome_columns=[])
    t_total = Sample.from_frame(t_df_total, id_column='_id', outcome_columns=[])
    adjusted_total = s_total.set_target(t_total).adjust(method='rake').trim(ratio=3)
    print(adjusted_total.summary())
    print(adjusted_total.weights().summary())
    adjusted_total.covars().plot()

# %%
# TRACKING MENSUAL
def tracking_mensual():
    tracking_imagen_mensual = (
        df.groupby('Ventana_M')
        .apply(lambda g: np.average(g['imagen_del_candidato'], weights=g['peso_m']))
        .reset_index(name='trackeo')
    )
    print(tracking_imagen_mensual.round(1))
    plt.figure(figsize=(10, 5))
    tracking_imagen_mensual.set_index('Ventana_M')['trackeo'].plot(marker='o')
    plt.xlabel('Ventana (mensual)', fontsize=10)
    plt.ylabel('Imagen promedio', fontsize=10)
    plt.title('Evolución de la imagen del candidato (ventana mensual)', fontsize=16)
    plt.tight_layout()
    plt.show()

    candidatos = df['voto'].unique().tolist()
    for c in candidatos:
        df[f'vota_{c}'] = (df['voto'] == c).astype(int)
    tracking_voto_mensual = (
        df.groupby('Ventana_M')
        .apply(lambda g: pd.Series({
            f"Vota_{c}": np.average(g[f'vota_{c}'], weights=g['peso_m']) * 100
            for c in candidatos
        }))
        .reset_index()
    )
    print(tracking_voto_mensual.round(1))
    cols_voto = [col for col in tracking_voto_mensual.columns if col.startswith('Vota_')]
    tracking_voto_mensual.set_index('Ventana_M')[cols_voto].plot(figsize=(10, 5))
    plt.xlabel('Ventana (mensual)', fontsize=10)
    plt.ylabel('Intención de voto (%)', fontsize=10)
    plt.title('Tracking de intención de voto (ventana mensual)', fontsize=16)
    plt.grid(alpha=0.3)
    plt.legend(title="Candidato")
    plt.tight_layout()
    plt.show()

    sample_df_total = df[vars_rake].copy().reset_index(drop=True)
    sample_df_total.insert(0, '_id', range(len(sample_df_total)))
    t_df_total = target_df.copy()
    if '_id' not in t_df_total.columns:
        t_df_total.insert(0, '_id', range(len(t_df_total)))
    s_total = Sample.from_frame(sample_df_total, id_column='_id', outcome_columns=[])
    t_total = Sample.from_frame(t_df_total, id_column='_id', outcome_columns=[])
    adjusted_total = s_total.set_target(t_total).adjust(method='rake').trim(ratio=3)
    print(adjusted_total.summary())
    print(adjusted_total.weights().summary())
    adjusted_total.covars().plot()

# %%
# Elegir el tipo de trackeo
tipo_track = input("Elegí el tipo de tracking (D = diario, S = semanal, M = mensual): ").strip().lower()
if tipo_track == "d":
    print("Generando TRACKING DIARIO...")
    tracking_diario()
elif tipo_track == "s":
    print("Generando TRACKING SEMANAL...")
    tracking_semanal()
elif tipo_track == "m":
    print("Generando TRACKING MENSUAL...")
    tracking_mensual()
else:
    print("Opción inválida. Elegí D, S o M.")
