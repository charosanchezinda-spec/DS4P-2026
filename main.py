from fastapi import FastAPI
from routers import censo, corridas, ml

app = FastAPI(
    title="API Censo 2022 — Targets para Raking",
    description=(
        "Parámetros del Censo 2022 (INDEC) para calibración por raking. "
        "Implementa el patrón de diseño Repository."
    ),
    version="1.0",
)

app.include_router(censo.router)
app.include_router(corridas.router)
app.include_router(ml.router)
