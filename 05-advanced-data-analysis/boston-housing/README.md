# Boston Housing Prices

## Descripción del Dataset

El dataset de Boston Housing contiene información sobre suburbios y pueblos del área metropolitana de Boston (SMSA) en 1970. Este es un problema clásico de **regresión** donde el objetivo es predecir el valor mediano de las viviendas (MEDV) en función de diversas características del área.

### Características del Dataset

- **Número de registros**: 506
- **Número de características**: 11 variables predictoras
- **Variable objetivo**: MEDV (Median value of owner-occupied homes in $1000s)

### Variables del Dataset

0. **CRIM**: Tasa de criminalidad per cápita por ciudad
1. **ZN**: Proporción de terreno residencial zonificado para lotes de más de 25,000 sq.ft.
2. **INDUS**: Proporción de acres de negocios no minoristas por ciudad
3. **CHAS**: Variable dummy del río Charles (= 1 si el tracto limita con el río; 0 en caso contrario)
4. **NOX**: Concentración de óxidos nítricos (partes por 10 millones)
5. **RM**: Número promedio de habitaciones por vivienda
6. **AGE**: Proporción de unidades ocupadas por propietarios construidas antes de 1940
7. **DIS**: Distancias ponderadas a cinco centros de empleo de Boston
8. **RAD**: Índice de accesibilidad a autopistas radiales
9. **TAX**: Tasa de impuesto a la propiedad de valor completo por $10,000
10. **PTRATIO**: Ratio alumno-profesor por ciudad
11. **MEDV**: Valor mediano de viviendas ocupadas por propietarios en $1000s (variable objetivo)

**Nota**: Las variables de entrada tienen una mezcla de unidades.

## Estructura de Carpetas

```
boston-housing/
├── README.md              # Este archivo
├── boston_analysis.ipynb  # Notebook con análisis y modelo
└── data/
    └── boston.csv         # Dataset en formato CSV
```

## Objetivos del Análisis

1. **Exploración de Datos (EDA)**:
   - Cargar y examinar el dataset
   - Analizar la distribución de las variables
   - Identificar relaciones entre variables

2. **Modelado**:
   - Preparar los datos para el modelado (train/test split)
   - Implementar un modelo de regresión (por ejemplo, regresión lineal)
   - Evaluar el rendimiento del modelo usando métricas apropiadas (MSE, RMSE, R²)

3. **Visualización**:
   - Crear visualizaciones de la distribución de la variable objetivo
   - Comparar valores reales vs. predichos

## Referencias

- Agredecimientos al Dr. Jason.
