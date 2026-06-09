# Sistema de Ponderación Electoral
**DS4P 2026 — Ciencia de Datos para Politólogos — UBA Sociales**

---

## Descripción

Sistema de calibración de encuestas electorales con parámetros del Censo Nacional 2022 (INDEC). Permite procesar encuestas para cualquier población de Argentina, aplicar ponderación por raking, monitorear la calidad de los pesos y visualizar los resultados a través de una interfaz web interactiva.

El sistema resuelve un problema concreto del trabajo con encuestas electorales: cada vez que cambia la población objetivo, las variables de calibración o el tamaño de la ventana temporal, el procesamiento suele depender de ajustes manuales en el código. Este sistema automatiza ese proceso.

---

## Arquitectura

```
.env                  → variables de configuración
main.py               → punto de entrada, corre el pipeline completo
api_censo.py          → API FastAPI con patrón Repository
app.py                → frontend Streamlit
│
├── carga.py          → carga y validación del archivo de encuesta
├── limpieza.py       → limpieza y normalización de variables
├── imputacion.py     → imputación de valores faltantes
├── ventanas.py       → creación de ventanas temporales
├── ponderacion.py    → raking, trimming y reporte de calibración
├── tracking.py       → tracking de imagen e intención de voto
├── estadistica.py    → intervalos de confianza y test de hipótesis
└── base_datos.py     → historial de corridas con SQLAlchemy
```

---

## Instalación

```bash
pip install fastapi uvicorn streamlit pandas numpy scikit-learn scipy balance matplotlib plotly requests sqlalchemy python-dotenv openpyxl
```

---

## Configuración

Crear un archivo `.env` en la carpeta del proyecto:

```
API_URL=http://localhost:8000
DB_PATH=tracking_electoral.db
```

---

## Cómo correrlo

El sistema necesita dos terminales abiertas al mismo tiempo.

**Terminal 1 — API:**
```bash
cd ruta/al/proyecto
python -m uvicorn api_censo:app --reload
```

Verificar que está corriendo en: `http://localhost:8000`

**Terminal 2 — Frontend:**
```bash
cd ruta/al/proyecto
streamlit run app.py
```

La aplicación se abre automáticamente en: `http://localhost:8501`

También se puede correr sin interfaz gráfica:
```bash
python main.py
```

---

## Uso

1. Cargar el archivo CSV de la encuesta en la barra lateral
2. Seleccionar la población objetivo
3. Seleccionar la ventana temporal (diaria, semanal o mensual)
4. Hacer clic en **Procesar encuesta**

El sistema genera automáticamente:
- KPIs: registros procesados, población y ventana
- Tracking de imagen del candidato
- Tracking de intención de voto por candidato
- Reporte de calibración (ASMD, Deff, ESS)
- Distribución muestra vs población por variable
- Monitoreo de pesos
- Intervalos de confianza al 95%
- Test de hipótesis sobre cambio en la imagen
- Registro automático en la base de datos

---

## Estructura del CSV de entrada

| Columna | Descripción |
|---|---|
| fecha | Fecha de la entrevista |
| encuesta | ID único del caso |
| estrato | Provincia o municipio del respondente |
| sexo | femenino / masculino |
| edad | Edad en años (mínimo 16) |
| nivel_educativo | desde Sin Estudios hasta Posgrado |
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

| Endpoint | Descripción |
|---|---|
| `GET /` | Confirma que la API está corriendo |
| `GET /poblaciones` | Lista de poblaciones disponibles |
| `GET /targets/{poblacion}` | Targets para una población |
| `GET /estrato-bsas` | Proporción GBA/interior para Buenos Aires |
| `GET /region-nacional` | Proporciones de región para encuestas nacionales |

---

## Base de datos

Cada corrida queda registrada automáticamente en `tracking_electoral.db` con dos tablas:

**corridas** — fecha, población, cantidad de registros y variables de calibración usadas.

**metricas** — Deff, ESS, ESSP, peso máximo y peso mínimo de cada corrida.

Para visualizar la base de datos: [SQLite Viewer](https://sqliteviewer.app)

---

## Fuente de datos

**INDEC — Censo Nacional de Población, Hogares y Viviendas 2022**
https://www.indec.gob.ar/indec/web/Nivel4-Tema-2-41-165

Variables de calibración: sexo, grupo etario, nivel educativo y región (solo para encuestas nacionales y PBA).

---

## Decisiones metodológicas

Ver [decisiones_metodologicas.md](decisiones_metodologicas.md)
