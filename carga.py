import pandas as pd
import os
def cargar_datos(archivo):
    try:
        if isinstance(archivo, str):
            nombre = archivo
            if not os.path.exists(archivo):
                raise FileNotFoundError(f"No se encontró el archivo: {archivo}")
        else:
            nombre = archivo.name
        _, extension = os.path.splitext(nombre)
        extension = extension.lower().strip()
        if extension == ".csv":
            df = pd.read_csv(archivo, encoding="utf-8")
        elif extension in [".xls", ".xlsx"]:
            df = pd.read_excel(archivo)
        else:
            raise ValueError(f"Formato no soportado: {extension}")
        df.columns = (
            df.columns
            .str.strip()
            .str.lower()
            .str.replace(" ", "_")
        )
        columnas_requeridas = ["fecha", "encuesta", "estrato", "sexo", "edad","nivel_educativo", "cantidad_de_integrantes_en_el_hogar","imagen_del_candidato", "voto", "voto_anterior"]
        faltantes = [col for col in columnas_requeridas if col not in df.columns]
        if faltantes:
            raise ValueError(f"Faltan columnas requeridas en el archivo: {faltantes}")
        df['fecha'] = pd.to_datetime(df['fecha'])
        df = df.rename(columns={"cantidad_de_integrantes_en_el_hogar": "integrantes_hogar"})
        print(f"Archivo cargado correctamente ({extension})")
        print(f"Filas: {len(df)} | Columnas: {len(df.columns)}")
        return df, None
    except FileNotFoundError as e:
        return None, str(e)
    except pd.errors.EmptyDataError:
        return None, "El archivo está vacío."
    except pd.errors.ParserError:
        return None, "Formato de archivo incorrecto o corrupto."
    except ValueError as e:
        return None, str(e)
    except Exception as e:
        return None, f"Error inesperado al cargar el archivo: {e}"
