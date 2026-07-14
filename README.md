# Sistema de Ponderación Electoral
**DS4P 2026 — Ciencia de Datos para Politólogos — UBA Sociales**
 
---
 
## Descripción
 
Sistema de calibración de encuestas electorales con parámetros del Censo Nacional 2022 (INDEC). Permite procesar encuestas continuas para cualquier población de Argentina, aplicar ponderación por raking, monitorear la calidad de los pesos y visualizar los resultados a través de una interfaz web interactiva.
 
El sistema resuelve un problema concreto del trabajo con encuestas electorales: cada vez que cambia la población objetivo, las variables de calibración o el tamaño de la ventana temporal, el procesamiento suele depender de ajustes manuales en el código. Este sistema automatiza ese proceso.
 
---
 
## URLs de producción
 
| Servicio | URL |
|---|---|
| Frontend (Streamlit Cloud) | https://raking-ds4p.streamlit.app/ |
| Backend (Render) | https://api-censo-electoral.onrender.com |
| Documentación API | https://api-censo-electoral.onrender.com/docs |
 
---
 
---
 
## Arquitectura
 
```
.env                  → variables de configuración (solo local)
main.py               → API FastAPI con patrón Repository
schemas.py            → validación de datos con Pydantic
base_datos.py         → ORM SQLAlchemy + PostgreSQL en Neon
app.py                → frontend Streamlit
entrenar_modelo.py    → entrenamiento y serialización de modelos ML
│
├── carga.py          → carga y validación del archivo de encuesta
├── limpieza.py       → limpieza y normalización de variables
├── imputacion.py     → imputación con modelos pre-entrenados
├── ventanas.py       → creación de ventanas temporales
├── ponderacion.py    → raking, trimming y advertencias
├── tracking.py       → tracking de imagen e intención de voto
└── estadistica.py    → intervalos de confianza y test de hipótesis
```
 
---
 
## Instalación local
 
```bash
pip install -r requirements.txt
```
 
Crear un archivo `.env` en la raíz del proyecto (ver `.env.example`):
 
---
 
## Cómo correrlo localmente
 
El sistema necesita dos terminales abiertas.
 
**Terminal 1 — API:**
```bash
cd ruta/al/proyecto
python -m uvicorn main:app --reload
```
 
**Terminal 2 — Frontend:**
```bash
cd ruta/al/proyecto
streamlit run app.py
```
 
La aplicación se abre en: `http://localhost:8501`
 
---
 
## Uso
 
1. Ingresar con las credenciales de prueba
2. Ir a **📂 Carga de encuesta**
3. Cargar el archivo CSV de la encuesta
4. Seleccionar la población objetivo y la ventana temporal
5. Hacer clic en **Procesar encuesta**
6. Ir a **📊 Dashboard analítico** para ver los resultados
7. El historial queda registrado en **📋 Historial de corridas**
---
 
## Estructura del CSV de entrada
 
| Columna | Descripción |
|---|---|
| fecha | Fecha de la entrevista |
| encuesta | ID único del caso |
| estrato | Provincia, municipio, región del respondente |
| sexo | femenino / masculino |
| edad | Edad en años (mínimo 16) |
| nivel_educativo | prim / sec / terc / univ / pos |
| cantidad_de_integrantes_en_el_hogar | Número entero |
| imagen_del_candidato | Escala 0-100 (los ns/nc se imputan) |
| voto | Nombre del candidato (los ns/nc se imputan) |
| voto_anterior | Nombre del candidato o no_voto (los ns/nc se imputan) |
 
---
 
## Poblaciones disponibles
 
32 poblaciones: 24 provincias y 8 recortes regionales.
 
| Clave | Descripción |
|---|---|
| nacional | Total Argentina |
| gba | Gran Buenos Aires (39 partidos) |
| interior_buenos_aires | Provincia de Buenos Aires sin GBA |
| buenos_aires | Provincia de Buenos Aires completa |
| caba | Ciudad Autónoma de Buenos Aires |
| pampeana | Región Pampeana |
| noa | Región NOA |
| nea | Región NEA |
| cuyo | Región Cuyo |
| patagonia | Región Patagonia |
| corrientes, tucuman, cordoba... | Cada provincia por separado |
 
---
 
## API — Endpoints disponibles
 
| Endpoint | Método | Descripción |
|---|---|---|
| `/` | GET | Estado de la API |
| `/poblaciones` | GET | Lista de poblaciones disponibles |
| `/targets/{poblacion}` | GET | Targets censales para una población 🔒|
| `/estrato-bsas` | GET | Proporción GBA/interior para Buenos Aires 🔒 |
| `/region-nacional` | GET | Proporciones de región para encuestas nacionales 🔒|
| `/predecir` | GET | Predicción de intención de voto 🔒|
| `/corridas` | POST | Registrar una corrida 🔒|
| `/corridas` | GET | Historial de corridas 🔒|
| `/docs` | GET | Documentación interactiva |
 
Los endpoints marcados con 🔒 requieren `x-api-key` en el header.
 
---
 
## Machine Learning
 
El sistema usa tres modelos entrenados con scikit-learn:
 
- **`modelo_voto_anterior.joblib`** — regresión logística para imputar voto anterior
- **`modelo_voto.joblib`** — regresión logística para imputar intención de voto
- **`modelo_imagen.joblib`** — regresión lineal para imputar imagen del candidato

Los modelos son demostrativos, entrenados con datos sintéticos, generados mediante Inteligencia Artificial.
 
---
 
## Base de datos
 
Cada corrida queda registrada automáticamente en PostgreSQL (Neon.tech) con dos tablas:
 
**corridas** — fecha, población, cantidad de registros y variables de calibración.
 
**metricas** — Deff, ESS, ESSP, peso máximo y peso mínimo de cada corrida.
 
---
 
## Fuente de datos
 
**INDEC — Censo Nacional de Población, Hogares y Viviendas 2022**
https://www.indec.gob.ar/indec/web/Nivel4-Tema-2-41-165
 
Variables de calibración: sexo, grupo etario, nivel educativo y región (solo para encuestas nacionales).
 
---
 
## Decisiones metodológicas
 
Ver [decisiones_metodologicas.md](decisiones_metodologicas.md)
