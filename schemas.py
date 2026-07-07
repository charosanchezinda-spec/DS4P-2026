from pydantic import BaseModel, field_validator

POBLACIONES_VALIDAS = {
    "nacional", "gba", "interior_buenos_aires", "pampeana", "noa", "nea",
    "cuyo", "patagonia", "caba", "buenos_aires", "catamarca", "cordoba",
    "corrientes", "chaco", "chubut", "entre_rios", "formosa", "jujuy",
    "la_pampa", "la_rioja", "mendoza", "misiones", "neuquen", "rio_negro",
    "salta", "san_juan", "san_luis", "santa_cruz", "santa_fe",
    "santiago_estero", "tierra_del_fuego", "tucuman"
}

class CorrridaCreate(BaseModel):
    poblacion:       str
    n_registros:     int
    variables_calib: str
    @field_validator('poblacion')
    def poblacion_valida(cls, v):
        if v not in POBLACIONES_VALIDAS:
            raise ValueError("Población no válida. Consulte /poblaciones para ver las opciones.")
        return v
