# Sistema de Ponderación Electoral
**DS4P 2026 — Ciencia de Datos para Politólogos — UBA Sociales**

***Alumnos: María José Pérez Morinigo, María Rosario Sofío, Charo Sanchez Inda y Gonzalo Murta***

---

## Descripción

Sistema de calibración de encuestas electorales con parámetros del Censo Nacional 2022 (INDEC). Permite procesar encuestas continuas para cualquier población de Argentina, aplicar ponderación por raking, monitorear la calidad de los pesos y visualizar los resultados a través de una interfaz web interactiva.

El sistema resuelve un problema concreto del trabajo con encuestas electorales: cada vez que cambia la población objetivo, las variables de calibración o el tamaño de la ventana temporal, el procesamiento suele depender de ajustes manuales en el código. Este sistema automatiza ese proceso.

---

## Arquitectura

```
api_censo.py          → API FastAPI con patrón Repository
                        Sirve parámetros del Censo 2022 por población
                        Corre localmente con uvicorn

tracking_electoral.py → Pipeline de análisis
                        Limpieza, imputación, raking, evaluación
                        Puede correr como script independiente

app.py                → Frontend Streamlit
                        Interfaz web para el usuario final
```

---

## Instalación

```bash
pip install fastapi uvicorn streamlit pandas numpy scikit-learn scipy balance matplotlib plotly requests openpyxl
```

---

## Cómo correrlo

El sistema necesita dos terminales abiertas al mismo tiempo.

**Terminal 1 — API:**
```bash
cd C:\ruta\al\proyecto
python -m uvicorn api_censo:app --reload
```

Verificar que está corriendo en: `http://localhost:8000`

**Terminal 2 — Frontend:**
```bash
cd C:\ruta\al\proyecto
streamlit run app.py
```

La aplicación se abre automáticamente en: `http://localhost:8501`

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
- Monitoreo de pesos (Deff, ESS, distribución)
- Intervalos de confianza al 95%
- Test de hipótesis sobre cambio en la imagen

---

## Estructura del CSV de entrada

| Columna | Descripción |
|---|---|
| fecha | Fecha de la entrevista (YYYY-MM-DD) |
| encuesta | ID único del caso |
| estrato | Provincia o municipio del respondente |
| sexo | femenino / masculino |
| edad | Edad en años (mínimo 16) |
| nivel_educativo | prim / sec / terc / univ / pos |
| cantidad_de_integrantes_en_el_hogar | Número entero |
| imagen_del_candidato | Escala 0-100 |
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
| `GET /docs` | Documentación interactiva |

---

## Fuente de datos

**INDEC — Censo Nacional de Población, Hogares y Viviendas 2022**
https://www.indec.gob.ar/indec/web/Nivel4-Tema-2-41-165

Variables de calibración: sexo, grupo etario, nivel educativo y región (solo para encuestas nacionales).

---

## Decisiones metodológicas

Ver `decisiones_metodologicas.md`
