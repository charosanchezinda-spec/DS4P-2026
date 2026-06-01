# Decisiones Metodológicas
**Sistema de Ponderación Electoral — DS4P 2026**

---

## 1. Problema que resuelve el sistema

En la práctica del trabajo de campo, las encuestas producen sesgos muestrales: ciertos perfiles de la población quedan sistematicamente sobre o subrepresentados respecto a su peso real en el universo. Esos sesgos no son una falla del diseño sino una consecuencia inevitable de la falta de cobertura muestral perfecta.

El raking (calibración iterativa por márgenes) corrige esos sesgos asignando un peso a cada caso de manera que la muestra ponderada replique la distribución real de la población según el Censo 2022. El sistema automatiza ese proceso para cualquier población de Argentina, adaptando los parámetros de calibración según la jurisdicción elegida.

---

## 2. Variables de calibración

Se calibra por tres variables base: sexo, grupo etario y nivel educativo. Para encuestas nacionales se agrega región como cuarta variable.

**Criterios de selección:**

**Disponibilidad.** El Censo 2022 publica estas variables para todas las jurisdicciones del país, lo que permite calibrar contra datos reales y no contra estimaciones.

**Relevancia electoral.** Las tres variables tienen relación documentada con el comportamiento electoral en Argentina. El voto varía sistemáticamente por edad, nivel educativo y región.

**Estabilidad del raking.** Incorporar más variables aumenta el riesgo de no convergencia, especialmente en ventanas con pocos casos. Tres o cuatro variables es el estándar en la práctica de encuestas electorales argentinas.

**Por qué región solo para nacional.** Cuando la encuesta cubre una sola provincia o región, todos los casos tienen el mismo valor en la variable región. El raking no puede calibrar una variable sin varianza y falla o produce resultados sin sentido. La variable se agrega automáticamente solo cuando la población elegida es nacional.

---

## 3. Arquitectura del sistema — patrón Repository

La API implementa el patrón de diseño Repository. La clase `CensoRepository` es la única responsable de saber dónde están los datos y cómo devolverlos. La API no accede a los datos directamente: siempre le pregunta al Repository.

```
CensoRepository  →  datos del Censo 2022 (diccionarios)
       ↓
API  →  recibe consultas y devuelve respuestas
       ↓
tracking_electoral.py  →  consume la API y procesa la encuesta
```

Si en el futuro cambia la fuente de datos, solo se modifica el Repository. La API y el tracking no se tocan.

---

## 4. Fuente de los parámetros censales

Todos los parámetros están incorporados directamente en diccionarios dentro del Repository. Esta decisión se tomó por dos razones:

El INDEC publica los datos de edad y educación en archivos con formato variable por provincia, sin estructura uniforme para descarga automática confiable.

Incorporar los datos directamente hace el sistema independiente de conectividad externa y el código completamente auditable.

Los valores provienen de los resultados definitivos del Censo Nacional 2022 publicados por el INDEC.

---

## 5. Ponderación por municipio

El sistema no pondera por municipio por defecto. Esta decisión responde a dos limitaciones metodológicas:

**Categorías vacías.** Si la encuesta llegó a pocos municipios pero la provincia tiene muchos, el raking no puede calibrar los municipios con cero casos y falla.

**Pesos extremos.** Si un municipio concentra mucha población pero tiene pocos casos en la muestra, el peso resultante puede ser muy alto, distorsionando las estimaciones.

La distinción GBA/interior sí está implementada para encuestas de la Provincia de Buenos Aires, porque es la única jurisdicción donde esa división tiene consenso metodológico establecido, peso electoral suficiente y datos del Censo 2022 disponibles.

---

## 5b. Recodificación de municipios en GBA/interior

Para encuestas de la Provincia de Buenos Aires donde el estrato viene codificado por municipio, el sistema recodifica automáticamente cada municipio en dos categorías: GBA o interior bonaerense.

Esta distinción se implementa solo para Buenos Aires por dos razones. Primero, es la provincia más poblada del país y concentra el mayor peso electoral a nivel nacional. Segundo, el GBA y el interior bonaerense muestran patrones de comportamiento electoral sistemáticamente distintos, lo que hace metodológicamente relevante distinguirlos como unidades de calibración separadas.

La detección es automática: si el sistema encuentra municipios bonaerenses en la columna estrato, crea la variable estrato_bsas y la incorpora como variable de calibración adicional. Para cualquier otra provincia el bloque no hace nada.

---

## 6. Trimming de pesos

Se aplica trimming con factor 3 dentro de cada función de raking, inmediatamente después de calcular los pesos y antes de la normalización. Ningún peso puede superar 3 veces el promedio de su ventana ni ser menor a un tercio del promedio.

El trimming se aplica por ventana porque los pesos se calculan por ventana. Aplicar un promedio global generaría límites incorrectos dado que cada ventana tiene su propia distribución de pesos.

---

## 7. Advertencias automáticas del raking

Dentro de cada función de raking, después de calcular los pesos, el sistema evalúa dos métricas:

**DEFF (efecto de diseño).** Si supera 2.5 indica pesos muy heterogéneos. El sistema muestra una advertencia sugiriendo ampliar la ventana temporal.

**CV de pesos.** Si supera 80% indica que hay perfiles muy subrepresentados en la muestra. El sistema sugiere revisar la composición de la muestra.

---

## 8. Ventanas temporales

El sistema ofrece tres ventanas: diaria, semanal y mensual. La ventana se elige por el usuario al momento de correr el análisis.

Con 1000 casos en 6 meses, la ventana diaria produce ventanas de 5 a 7 casos en promedio. Eso es insuficiente para una calibración correcta y genera pesos extremos. Las advertencias automáticas del sistema detectan y reportan este problema. La ventana semanal o mensual es más apropiada para muestras de ese tamaño.

---

## 9. Imputación de valores faltantes

Los valores faltantes en las variables dependientes (voto, voto anterior, imagen) se imputan mediante modelos supervisados:

**Voto y voto anterior:** regresión logística multinomial con predictores socioeconómicos (edad, sexo, estrato, nivel educativo).

**Imagen del candidato:** regresión lineal. Si el R² del modelo es menor a 0.15, se imputa por mediana.

Los modelos se evalúan antes de imputar para verificar su capacidad predictiva.

---

## 10. Evaluación de la calibración

Al final de cada corrida el sistema genera un reporte de calibración usando el flujo completo de la librería `balance`:

**ASMD (Average Standardized Mean Difference).** Mide el sesgo antes y después del raking. Una reducción del 100% indica calibración perfecta.

**DEFF y ESS.** Miden la calidad de los pesos sobre el total de la encuesta.

**Distribución muestra vs población.** Gráfico de barras por variable comparando la composición de la muestra con los targets del censo.

Este diagnóstico es global (sobre el total de la encuesta) y complementa las advertencias por ventana que se generan durante el raking.

---

## 11. Qué no cambió respecto al trabajo anterior

La imputación de valores faltantes, los trackings, los intervalos de confianza y los tests de hipótesis son idénticos al trabajo anterior (MET4OP). El aporte de este trabajo es la arquitectura de obtención de parámetros, la flexibilidad geográfica del sistema, el monitoreo automático de pesos y la interfaz web.
