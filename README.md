# Sistema de Ponderación Electoral
**DS4P 2026 — Ciencia de Datos para Politólogos — UBA Sociales**

**Alumnos: María José Perez Morinigo, Rosario Sofio, Gonzalo Murta y Charo Sanchez Inda**
 
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
── API ──────────────────────────────────────
main.py               → punto de entrada FastAPI, registra routers
security.py           → verificación de API key
repository.py         → clase CensoRepository con parámetros del Censo 2022
schemas.py            → validación de datos con Pydantic
base_datos.py         → ORM SQLAlchemy + PostgreSQL en Neon
routers/
├── censo.py          → endpoints de censo y poblaciones
├── corridas.py       → endpoints de corridas y métricas
└── ml.py             → endpoint de predicción

── PIPELINE ─────────────────────────────────
carga.py              → carga y validación del archivo de encuesta
limpieza.py           → limpieza y normalización de variables
imputacion.py         → imputación con modelos pre-entrenados
ventanas.py           → creación de ventanas temporales
ponderacion.py        → raking, trimming y reporte de calibración
tracking.py           → cálculo de tracking
estadistica.py        → intervalos de confianza y test de hipótesis
entrenar_modelo.py    → entrenamiento y serialización de modelos ML

── FRONTEND ─────────────────────────────────
app.py                → frontend Streamlit
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
| nivel_educativo | Desde sin estudios hasta posgrado |
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
| `/metricas` | POST | Registrar métricas de una corrida 🔒|
| `/metricas/{corrida_id}` | GET | Métricas de una corrida 🔒|
| `/docs` | GET | Documentación interactiva |
 
Los endpoints marcados con 🔒 requieren `x-api-key` en el header.
 
---
 
## Machine Learning

El sistema implementa aprendizaje incremental con tres modelos de scikit-learn:
 
- **`modelo_voto_anterior.joblib`** — regresión logística para imputar voto anterior
- **`modelo_voto.joblib`** — regresión logística para imputar intención de voto
- **`modelo_imagen.joblib`** — regresión lineal para imputar imagen del candidato
**Flujo de aprendizaje incremental:**
 
1. `entrenar_modelo.py` inicializa el sistema entrenando los modelos sobre la encuesta ficticia y genera los `.joblib` iniciales junto con los datos históricos
2. Al procesar una encuesta nueva, `imputacion.py` genera un hash único de la encuesta para evitar incorporarla dos veces al histórico
3. Imputa los valores faltantes usando el modelo pre-entrenado. Si no existe modelo, entrena uno temporal con los casos disponibles
4. Para `imagen_del_candidato`: si R² > 0.15 usa regresión lineal, sino usa la mediana
5. Una vez finalizada la imputación, re-entrena los modelos usando los datos acumulados anteriores más los casos observados de la encuesta nueva
6. Sobreescribe los `.joblib` con las versiones actualizadas
7. El endpoint `/predecir` usa esos modelos actualizados cuando se reinicia FastAPI
**Decisión metodológica:** el reentrenamiento se realiza exclusivamente con los valores originalmente observados de cada nueva encuesta. Los valores imputados se utilizan únicamente para completar la base de análisis y permitir el procesamiento del tracking, pero no se incorporan al entrenamiento. Esto evita el problema de self-training no supervisado, donde un modelo aprende de sus propias predicciones y puede amplificar errores a lo largo del tiempo.
 
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
