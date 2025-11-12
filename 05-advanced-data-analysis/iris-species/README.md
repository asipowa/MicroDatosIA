# Iris Species

## Descripción del Dataset

El dataset de Iris es uno de los datasets más famosos en machine learning y estadística. Contiene mediciones de tres especies diferentes de flores Iris: **Iris setosa**, **Iris versicolor** e **Iris virginica**. Este es un problema clásico de **clasificación** donde el objetivo es predecir la especie de la flor basándose en las medidas de sus sépalos y pétalos.

### Características del Dataset

- **Número de registros**: 150 (50 por cada especie)
- **Número de características**: 4 variables predictoras
- **Variable objetivo**: species (3 clases: setosa, versicolor, virginica)

### Variables del Dataset

1. **sepal_length**: Longitud del sépalo (cm)
2. **sepal_width**: Ancho del sépalo (cm)
3. **petal_length**: Longitud del pétalo (cm)
4. **petal_width**: Ancho del pétalo (cm)
5. **species**: Especie de la flor (setosa, versicolor, virginica) - variable objetivo

### Especies

- **Iris setosa**: Generalmente tiene pétalos más pequeños y es fácilmente distinguible
- **Iris versicolor**: Especie intermedia
- **Iris virginica**: Generalmente tiene pétalos más grandes

## Estructura de Carpetas

```
iris-species/
├── README.md              # Este archivo
├── iris_analysis.ipynb    # Notebook con análisis y visualizaciones
└── data/
    └── iris.csv           # Dataset en formato CSV
```

## Objetivos del Análisis

1. **Exploración de Datos (EDA)**:
   - Cargar y examinar el dataset
   - Analizar las estadísticas descriptivas de cada variable
   - Explorar las diferencias entre las tres especies

2. **Visualización**:
   - Crear gráficos de dispersión para visualizar las relaciones entre variables
   - Usar colores diferentes para cada especie
   - Identificar qué características mejor distinguen entre las especies

3. **Modelado (opcional)**:
   - Implementar un modelo de clasificación (por ejemplo, K-Nearest Neighbors, SVM, o Random Forest)
   - Evaluar el rendimiento del modelo usando métricas de clasificación (accuracy, confusion matrix)

## Notas

- Este dataset es perfecto para principiantes debido a su simplicidad y tamaño manejable
- Las características están en las mismas unidades (centímetros), lo que facilita el análisis
- El dataset está balanceado (50 muestras por clase), lo que es ideal para clasificación

## Referencias

- [UCI Machine Learning Repository - Iris Dataset](https://archive.ics.uci.edu/ml/datasets/iris)
- Dataset original creado por R.A. Fisher (1936)
