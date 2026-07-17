from fastapi import APIRouter, HTTPException, Depends
from security import verificar_api_key
from repository import repo

router = APIRouter()

@router.get("/")
def bienvenida():
    return {
        "estado": "API funcionando correctamente",
        "version": "1.0",
        "patron": "Repository",
        "fuente": "INDEC — Censo Nacional 2022",
        "documentacion": "/docs"
    }

@router.get("/poblaciones")
def listar_poblaciones():
    return {"poblaciones": repo.obtener_poblaciones()}

@router.get("/estrato-bsas", dependencies=[Depends(verificar_api_key)])
def obtener_estrato_bsas():
    return {
        "descripcion": "Proporción GBA vs interior bonaerense",
        "fuente": "INDEC — Censo Nacional 2022",
        "uso": "Agregar como variable de calibración cuando la encuesta es de Buenos Aires y tiene municipios en la columna estrato",
        "estrato": repo.obtener_estrato_bsas(),
    }

@router.get("/targets/{poblacion}", dependencies=[Depends(verificar_api_key)])
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

@router.get("/region-nacional", dependencies=[Depends(verificar_api_key)])
def obtener_region_nacional():
    return {"region": repo.obtener_region_nacional()}
