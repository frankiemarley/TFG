# Análisis Exploratorio de Datos (EDA)

## 1. Dimensiones del dataset

- `tracks.csv`: 586.672 filas × 20 columnas (una fila por canción)
- `artists.csv`: 1.162.095 filas × 5 columnas (una fila por artista)

## 2. Calidad de datos

**tracks.csv**
- `name`: 71 nulos — descartables.
- `release_year` (tras parseo): 138.591 nulos (23,6%) — Preocupa porque el año  de lanzamiento es necesario para sacar la decada. Aparte que las fechas no son ausente sino formatos heterogeneos.

**artists.csv**
- `followers`: 11 nulos — descartables.
- `name`: 3 nulos — descartables.

## 3. Features de audio y normalización

La mayoría de features estan acotadas entre 0 y 1 pero el loudness (-60, 0) el tempo esta medido bpm (0, 250) y key (0, 11).
La solución es usar StandardScaler que ajusta los datos para que puedan tener datos comparables con el resto.

## 4. Cobertura de artistas

- Artistas únicos en tracks: 96.277
- Artistas sin género en artists.csv: 856.500 (73,7%)
- Mediana de followers: 57 | p75: 417
- Mediana de popularity: 2 | p75: 13
- Tracks con múltiples artistas: 106.916 (18,2%)

El dataset está dominado por artistas con presencia mínima. La mayoría
es ruido para modelar influencias musicales. La mitad de artistas tiene menos de 57 seguidores, el 73,3% sin genero asignado y la mayoria de artistas contituye ruido para el objetivo de modelar influencias estilísticas

## 5. Géneros

Top 10: dance pop, pop, rock, electro house, classical performance, latin,
indie rock, hip hop, pop rap, rap.

Microgéneros de Spotify: muy granulares (pop rap ≠ rap ≠ hip hop),
complica el one-hot. Decisión inicial: usar top-30 tal cual.
En one-hot, rap = [1,0,0] y hip hop = [0,1,0] están a la misma distancia que rap y classical, lo cual es musicalmente absurdo. Se considerará agrupar los microgéneros en macrogéneros.

## 6. Decisiones de filtrado para el modelado

A partir del análisis anterior se define un filtrado de artistas basado en la intersección de cuatro criterios, justificados individualmente:

-Género asignado (≥1): sin género no puede construirse la feature correspondiente; además, los artistas sin género en Spotify tienden a ser contenido irrelevante (versiones, demos, audiolibros). Reduce el universo a ~305.000 artistas.
-Número mínimo de canciones (umbral a determinar, p. ej. ≥5): las features de cada artista se calculan como la media de las de sus canciones; con una sola canción la estimación del "estilo" es inestable. Más canciones implican una media más fiable.
-Popularidad o seguidores mínimos (p. ej. popularity ≥ 10 o followers ≥ 100): descarta cuentas inactivas o sin promoción que aportan ruido.
-Año de debut disponible: necesario para la feature de década; sin él no puede ubicarse temporalmente al artista.

## 7. Selección del dataset

Se eligió `yamaerenay/spotify-dataset-19212020-600k-tracks` por: (1) tabla
separada de artistas con géneros, (2) span temporal 1921-2020 para análisis de
influencias, (3) las 11 features de audio completas.

Limitación: cubre hasta 2020. Spotify cerró el endpoint de audio features en
noviembre de 2024, lo que dificulta extender el corpus. Se propone como trabajo
futuro.