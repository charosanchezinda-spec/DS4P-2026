from fastapi import FastAPI, HTTPException, Header, Depends
from sqlalchemy.orm import Session
from schemas import CorrridaCreate
from base_datos import get_db, CorrridaDB, MetricaDB, registrar_corrida
import joblib
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("API_KEY")
if not API_KEY:
    raise RuntimeError("Falta la clave de seguridad API_KEY en el archivo .env")

def verificar_api_key(x_api_key: str = Header(...)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="No autorizado.")

class CensoRepository:
    _POBLACIONES = {
        "nacional": "Total Argentina",
        "gba": "Gran Buenos Aires (39 partidos)",
        "interior_buenos_aires": "Provincia de Buenos Aires sin GBA",
        "pampeana": "Región Pampeana",
        "noa": "Región NOA",
        "nea": "Región NEA",
        "cuyo": "Región Cuyo",
        "patagonia": "Región Patagonia",
        "caba": "Ciudad Autónoma de Buenos Aires",
        "buenos_aires": "Provincia de Buenos Aires",
        "catamarca": "Catamarca",
        "cordoba": "Córdoba",
        "corrientes": "Corrientes",
        "chaco": "Chaco",
        "chubut": "Chubut",
        "entre_rios": "Entre Ríos",
        "formosa": "Formosa",
        "jujuy": "Jujuy",
        "la_pampa": "La Pampa",
        "la_rioja": "La Rioja",
        "mendoza": "Mendoza",
        "misiones": "Misiones",
        "neuquen": "Neuquén",
        "rio_negro": "Río Negro",
        "salta": "Salta",
        "san_juan": "San Juan",
        "san_luis": "San Luis",
        "santa_cruz": "Santa Cruz",
        "santa_fe": "Santa Fe",
        "santiago_estero": "Santiago del Estero",
        "tierra_del_fuego": "Tierra del Fuego",
        "tucuman": "Tucumán",
    }

    _SEXO = {
        "nacional": {"femenino": 0.519, "masculino": 0.481},
        "gba": {"femenino": 0.521, "masculino": 0.479},
        "interior_buenos_aires": {"femenino": 0.516, "masculino": 0.484},
        "pampeana": {"femenino": 0.517, "masculino": 0.483},
        "noa": {"femenino": 0.514, "masculino": 0.486},
        "nea": {"femenino": 0.513, "masculino": 0.487},
        "cuyo": {"femenino": 0.516, "masculino": 0.484},
        "patagonia": {"femenino": 0.510, "masculino": 0.490},
        "caba": {"femenino": 0.530, "masculino": 0.470},
        "buenos_aires": {"femenino": 0.519, "masculino": 0.481},
        "catamarca": {"femenino": 0.514, "masculino": 0.486},
        "cordoba": {"femenino": 0.519, "masculino": 0.481},
        "corrientes": {"femenino": 0.513, "masculino": 0.487},
        "chaco": {"femenino": 0.512, "masculino": 0.488},
        "chubut": {"femenino": 0.508, "masculino": 0.492},
        "entre_rios": {"femenino": 0.517, "masculino": 0.483},
        "formosa": {"femenino": 0.511, "masculino": 0.489},
        "jujuy": {"femenino": 0.513, "masculino": 0.487},
        "la_pampa": {"femenino": 0.516, "masculino": 0.484},
        "la_rioja": {"femenino": 0.514, "masculino": 0.486},
        "mendoza": {"femenino": 0.516, "masculino": 0.484},
        "misiones": {"femenino": 0.512, "masculino": 0.488},
        "neuquen": {"femenino": 0.509, "masculino": 0.491},
        "rio_negro": {"femenino": 0.511, "masculino": 0.489},
        "salta": {"femenino": 0.513, "masculino": 0.487},
        "san_juan": {"femenino": 0.515, "masculino": 0.485},
        "san_luis": {"femenino": 0.514, "masculino": 0.486},
        "santa_cruz": {"femenino": 0.507, "masculino": 0.493},
        "santa_fe": {"femenino": 0.519, "masculino": 0.481},
        "santiago_estero": {"femenino": 0.513, "masculino": 0.487},
        "tierra_del_fuego": {"femenino": 0.504, "masculino": 0.496},
        "tucuman": {"femenino": 0.514, "masculino": 0.486},
    }

    _ESTRATO_BSAS = {
        "gba": 0.632,
        "interior": 0.368,
    }

    _EDAD = {
        "nacional": {"16-29": 0.202, "30-44": 0.284, "45-59": 0.237, "60+": 0.277},
        "gba": {"16-29": 0.218, "30-44": 0.293, "45-59": 0.232, "60+": 0.257},
        "interior_buenos_aires": {"16-29": 0.196, "30-44": 0.281, "45-59": 0.237, "60+": 0.286},
        "pampeana": {"16-29": 0.198, "30-44": 0.281, "45-59": 0.238, "60+": 0.283},
        "noa": {"16-29": 0.221, "30-44": 0.284, "45-59": 0.232, "60+": 0.263},
        "nea": {"16-29": 0.224, "30-44": 0.286, "45-59": 0.231, "60+": 0.259},
        "cuyo": {"16-29": 0.207, "30-44": 0.284, "45-59": 0.236, "60+": 0.273},
        "patagonia": {"16-29": 0.210, "30-44": 0.299, "45-59": 0.244, "60+": 0.247},
        "caba": {"16-29": 0.174, "30-44": 0.279, "45-59": 0.248, "60+": 0.299},
        "buenos_aires": {"16-29": 0.210, "30-44": 0.289, "45-59": 0.234, "60+": 0.267},
        "catamarca": {"16-29": 0.214, "30-44": 0.282, "45-59": 0.234, "60+": 0.270},
        "cordoba": {"16-29": 0.196, "30-44": 0.281, "45-59": 0.239, "60+": 0.284},
        "corrientes": {"16-29": 0.226, "30-44": 0.287, "45-59": 0.229, "60+": 0.258},
        "chaco": {"16-29": 0.231, "30-44": 0.288, "45-59": 0.227, "60+": 0.254},
        "chubut": {"16-29": 0.213, "30-44": 0.301, "45-59": 0.243, "60+": 0.243},
        "entre_rios": {"16-29": 0.197, "30-44": 0.280, "45-59": 0.238, "60+": 0.285},
        "formosa": {"16-29": 0.228, "30-44": 0.287, "45-59": 0.228, "60+": 0.257},
        "jujuy": {"16-29": 0.221, "30-44": 0.287, "45-59": 0.232, "60+": 0.260},
        "la_pampa": {"16-29": 0.188, "30-44": 0.276, "45-59": 0.239, "60+": 0.297},
        "la_rioja": {"16-29": 0.218, "30-44": 0.284, "45-59": 0.232, "60+": 0.266},
        "mendoza": {"16-29": 0.206, "30-44": 0.284, "45-59": 0.236, "60+": 0.274},
        "misiones": {"16-29": 0.225, "30-44": 0.288, "45-59": 0.230, "60+": 0.257},
        "neuquen": {"16-29": 0.211, "30-44": 0.302, "45-59": 0.244, "60+": 0.243},
        "rio_negro": {"16-29": 0.205, "30-44": 0.293, "45-59": 0.244, "60+": 0.258},
        "salta": {"16-29": 0.224, "30-44": 0.285, "45-59": 0.231, "60+": 0.260},
        "san_juan": {"16-29": 0.207, "30-44": 0.283, "45-59": 0.236, "60+": 0.274},
        "san_luis": {"16-29": 0.210, "30-44": 0.287, "45-59": 0.236, "60+": 0.267},
        "santa_cruz": {"16-29": 0.211, "30-44": 0.307, "45-59": 0.247, "60+": 0.235},
        "santa_fe": {"16-29": 0.194, "30-44": 0.279, "45-59": 0.238, "60+": 0.289},
        "santiago_estero": {"16-29": 0.222, "30-44": 0.284, "45-59": 0.231, "60+": 0.263},
        "tierra_del_fuego": {"16-29": 0.205, "30-44": 0.308, "45-59": 0.248, "60+": 0.239},
        "tucuman": {"16-29": 0.220, "30-44": 0.284, "45-59": 0.232, "60+": 0.264},
    }

    _EDUCACION = {
        "nacional": {"prim": 0.285, "sec": 0.430, "terc": 0.105, "univ": 0.158, "pos": 0.022},
        "gba": {"prim": 0.291, "sec": 0.441, "terc": 0.104, "univ": 0.141, "pos": 0.023},
        "interior_buenos_aires": {"prim": 0.258, "sec": 0.432, "terc": 0.111, "univ": 0.176, "pos": 0.023},
        "pampeana": {"prim": 0.274, "sec": 0.427, "terc": 0.109, "univ": 0.168, "pos": 0.022},
        "noa": {"prim": 0.331, "sec": 0.415, "terc": 0.101, "univ": 0.133, "pos": 0.020},
        "nea": {"prim": 0.352, "sec": 0.408, "terc": 0.097, "univ": 0.124, "pos": 0.019},
        "cuyo": {"prim": 0.296, "sec": 0.429, "terc": 0.106, "univ": 0.148, "pos": 0.021},
        "patagonia": {"prim": 0.258, "sec": 0.444, "terc": 0.115, "univ": 0.162, "pos": 0.021},
        "caba": {"prim": 0.152, "sec": 0.380, "terc": 0.118, "univ": 0.297, "pos": 0.053},
        "buenos_aires": {"prim": 0.278, "sec": 0.438, "terc": 0.107, "univ": 0.155, "pos": 0.022},
        "catamarca": {"prim": 0.321, "sec": 0.419, "terc": 0.108, "univ": 0.134, "pos": 0.018},
        "cordoba": {"prim": 0.261, "sec": 0.425, "terc": 0.112, "univ": 0.179, "pos": 0.023},
        "corrientes": {"prim": 0.355, "sec": 0.409, "terc": 0.095, "univ": 0.123, "pos": 0.018},
        "chaco": {"prim": 0.372, "sec": 0.403, "terc": 0.091, "univ": 0.117, "pos": 0.017},
        "chubut": {"prim": 0.258, "sec": 0.448, "terc": 0.114, "univ": 0.160, "pos": 0.020},
        "entre_rios": {"prim": 0.279, "sec": 0.431, "terc": 0.108, "univ": 0.161, "pos": 0.021},
        "formosa": {"prim": 0.368, "sec": 0.405, "terc": 0.093, "univ": 0.117, "pos": 0.017},
        "jujuy": {"prim": 0.319, "sec": 0.419, "terc": 0.103, "univ": 0.140, "pos": 0.019},
        "la_pampa": {"prim": 0.267, "sec": 0.436, "terc": 0.110, "univ": 0.166, "pos": 0.021},
        "la_rioja": {"prim": 0.305, "sec": 0.424, "terc": 0.107, "univ": 0.146, "pos": 0.018},
        "mendoza": {"prim": 0.292, "sec": 0.431, "terc": 0.106, "univ": 0.150, "pos": 0.021},
        "misiones": {"prim": 0.363, "sec": 0.406, "terc": 0.093, "univ": 0.121, "pos": 0.017},
        "neuquen": {"prim": 0.249, "sec": 0.446, "terc": 0.117, "univ": 0.167, "pos": 0.021},
        "rio_negro": {"prim": 0.261, "sec": 0.443, "terc": 0.115, "univ": 0.161, "pos": 0.020},
        "salta": {"prim": 0.322, "sec": 0.416, "terc": 0.102, "univ": 0.141, "pos": 0.019},
        "san_juan": {"prim": 0.299, "sec": 0.429, "terc": 0.106, "univ": 0.146, "pos": 0.020},
        "san_luis": {"prim": 0.289, "sec": 0.432, "terc": 0.107, "univ": 0.151, "pos": 0.021},
        "santa_cruz": {"prim": 0.254, "sec": 0.449, "terc": 0.115, "univ": 0.161, "pos": 0.021},
        "santa_fe": {"prim": 0.271, "sec": 0.428, "terc": 0.110, "univ": 0.169, "pos": 0.022},
        "santiago_estero": {"prim": 0.358, "sec": 0.407, "terc": 0.097, "univ": 0.121, "pos": 0.017},
        "tierra_del_fuego": {"prim": 0.233, "sec": 0.451, "terc": 0.118, "univ": 0.177, "pos": 0.021},
        "tucuman": {"prim": 0.316, "sec": 0.418, "terc": 0.103, "univ": 0.144, "pos": 0.019},
    }

    _REGION_NACIONAL = {
        "Región Metropolitana": 0.42,
        "Región Pampeana": 0.23,
        "Región NOA": 0.14,
        "Región NEA": 0.08,
        "Región Cuyo": 0.07,
        "Región Patagonia": 0.06,
    }

    def obtener_poblaciones(self):
        return self._POBLACIONES

    def poblacion_existe(self, poblacion):
        return poblacion in self._POBLACIONES

    def obtener_estrato_bsas(self):
        return self._ESTRATO_BSAS

    def obtener_targets(self, poblacion):
        return {
            "sexo": self._SEXO[poblacion],
            "edad_cat": self._EDAD[poblacion],
            "nivel_educativo": self._EDUCACION[poblacion],
        }

    def obtener_region_nacional(self):
        return self._REGION_NACIONAL

app = FastAPI(
    title="API Censo 2022 — Targets para Raking",
    description=(
        "Parámetros del Censo 2022 (INDEC) para calibración por raking. "
        "Implementa el patrón de diseño Repository."
    ),
    version="1.0",
)
repo = CensoRepository()
# ==========================================
# CARGA DEL MODELO PREDICTIVO
# ========================================== 
_ruta_modelo = os.path.join(os.path.dirname(__file__), "modelo_voto.joblib")
_ruta_features = os.path.join(os.path.dirname(__file__), "features_voto.joblib")
if os.path.exists(_ruta_modelo) and os.path.exists(_ruta_features):
    modelo_voto   = joblib.load(_ruta_modelo)
    features_voto = joblib.load(_ruta_features)
    print("Modelo predictivo cargado correctamente.")
else:
    modelo_voto   = None
    features_voto = None
    print("Modelo predictivo no encontrado. Corra entrenar_modelo.py primero.")
# ==========================================
# ENDPOINTS
# ==========================================
@app.get("/")
def bienvenida():
    return {
        "estado": "API funcionando correctamente",
        "version": "1.0",
        "patron": "Repository",
        "fuente": "INDEC — Censo Nacional 2022",
        "endpoints": {
            "/poblaciones": "Lista de poblaciones disponibles",
            "/targets/{nombre}": "Targets para una población",
            "/estrato-bsas": "Proporción GBA/interior para Buenos Aires",
            "/region-nacional": "Proporciones de región para encuestas nacionales",
        }
    }

@app.get("/poblaciones")
def listar_poblaciones():
    return {"poblaciones": repo.obtener_poblaciones()}

@app.get("/estrato-bsas", dependencies=[Depends(verificar_api_key)])
def obtener_estrato_bsas():
    return {
        "descripcion": "Proporción GBA vs interior bonaerense",
        "fuente": "INDEC — Censo Nacional 2022",
        "uso": "Agregar como variable de calibración cuando la encuesta es de Buenos Aires y tiene municipios en la columna estrato",
        "estrato": repo.obtener_estrato_bsas(),
    }

@app.get("/targets/{poblacion}", dependencies=[Depends(verificar_api_key)])
def obtener_targets(poblacion: str):
    if not repo.poblacion_existe(poblacion):
        raise HTTPException(
            status_code=404,
            detail={
                "error": f"Población '{poblacion}' no encontrada.",
                "opciones": list(repo.obtener_poblaciones().keys()),
                "sugerencia": "Consulte /poblaciones para ver las opciones.",
            }
        )
    return {
        "poblacion": poblacion,
        "descripcion": repo.obtener_poblaciones()[poblacion],
        "fuente": "INDEC — Censo Nacional de Población 2022",
        "targets": repo.obtener_targets(poblacion),
    }

@app.get("/region-nacional", dependencies=[Depends(verificar_api_key)])
def obtener_region_nacional():
    return {"region": repo.obtener_region_nacional()}

@app.post("/corridas", dependencies=[Depends(verificar_api_key)])
def crear_corrida(corrida: CorrridaCreate, db: Session = Depends(get_db)):
    corrida_id = registrar_corrida(db, corrida.poblacion, corrida.n_registros, corrida.variables_calib)
    return {"corrida_id": corrida_id, "mensaje": "Corrida registrada correctamente."}
 
@app.get("/corridas", dependencies=[Depends(verificar_api_key)])
def listar_corridas(db: Session = Depends(get_db)):
    corridas = db.query(CorrridaDB).order_by(CorrridaDB.id.desc()).all()
    return {"total": len(corridas), "corridas": [
        {
            "id": c.id,
            "fecha_hora": c.fecha_hora,
            "poblacion": c.poblacion,
            "n_registros": c.n_registros,
            "variables_calib": c.variables_calib,
        }
        for c in corridas
    ]}

@app.get("/predecir", dependencies=[Depends(verificar_api_key)])
def predecir(edad: int, sexo: str, nivel_educativo: str):
    if modelo_voto is None:
        raise HTTPException(
            status_code=503,
            detail="Modelo no disponible. Corra entrenar_modelo.py primero."
        )
    datos = pd.DataFrame([{"edad": edad, "sexo": sexo, "nivel_educativo": nivel_educativo}])
    X = pd.get_dummies(datos, drop_first=True).reindex(columns=features_voto, fill_value=0)
    prediccion = modelo_voto.predict(X)[0]
    probabilidades = modelo_voto.predict_proba(X)[0].tolist()
    return {
        "prediccion": prediccion,
        "probabilidades": dict(zip(modelo_voto.classes_, probabilidades)),
        "nota": "Modelo demostrativo entrenado con datos ficticios. Puede reentrenarse con encuestas reales."
    }
