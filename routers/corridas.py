from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from schemas import CorridaCreate
from base_datos import get_db, CorridaDB, registrar_corrida
from security import verificar_api_key

router = APIRouter()

@router.post("/corridas", dependencies=[Depends(verificar_api_key)])
def crear_corrida(corrida: CorridaCreate, db: Session = Depends(get_db)):
    corrida_id = registrar_corrida(db, corrida.poblacion, corrida.n_registros, corrida.variables_calib)
    return {"corrida_id": corrida_id, "mensaje": "Corrida registrada correctamente."}

@router.get("/corridas", dependencies=[Depends(verificar_api_key)])
def listar_corridas(db: Session = Depends(get_db)):
    corridas = db.query(CorridaDB).order_by(CorridaDB.id.desc()).all()
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

@router.post("/metricas", dependencies=[Depends(verificar_api_key)])
def crear_metrica(metrica: MetricaCreate, db: Session = Depends(get_db)):
    registrar_metricas(db, metrica.corrida_id, metrica.deff, metrica.ess, metrica.essp, metrica.peso_max, metrica.peso_min)
    return {"mensaje": "Métrica registrada correctamente."}

@router.get("/metricas/{corrida_id}", dependencies=[Depends(verificar_api_key)])
def obtener_metrica(corrida_id: int, db: Session = Depends(get_db)):
    from base_datos import MetricaDB
    metrica = db.query(MetricaDB).filter(MetricaDB.corrida_id == corrida_id).first()
    if not metrica:
        return {"metrica": None}
    return {"metrica": {
        "deff": metrica.deff,
        "ess": metrica.ess,
        "essp": metrica.essp,
        "peso_max": metrica.peso_max,
        "peso_min": metrica.peso_min,
    }}
