# USArrests: Violent Crime Rates by US State

## Descripción del Dataset

El dataset USArrests contiene estadísticas sobre arrestos por delitos violentos (asesinato, asalto y violación) en los 50 estados de los Estados Unidos durante el año 1973. Además, incluye el porcentaje de la población que vive en áreas urbanas.


### Características del Dataset

- **Número de registros**: 50 (uno por cada estado de EE.UU.)
- **Número de características**: 4 variables numéricas
- **Tipo de problema**: Exploratorio / Clustering / Análisis multivariante

### Variables del Dataset

1. **Murder**: Arrestos por asesinato por cada 100,000 habitantes
2. **Assault**: Arrestos por asalto por cada 100,000 habitantes.
3. **UrbanPop**: Porcentaje de población urbana
4. **Rape**: Arrestos por violación por cada 100,000 habitantes


**Nota**: Todas las variables son numéricas y no hay variable objetivo explícita.

# Digits: Imágenes de Dígitos Escritos a Mano

## Descripción del Dataset

El dataset Digits contiene imágenes en escala de grises de dígitos escritos a mano (0–9). Cada imagen es una matriz de 8×8 píxeles y representa una única muestra hecha sobre tabletas digitales.

Los datos son una versión del conjunto de test del famoso repositorio UCI Optical Recognition of Handwritten Digits


### Características del Dataset

- **Número de imágenes**: 1797
- **Número de clases**: 10 (dígitos del 0 al 9)
- **Imágenes por clase**: ~180
- **Dimensionalidad**: 64 características por imagen

### Estructura de los Datos 
- **data**:
    Cada pixel es un valor entero entre 0 y 16.
    Imagen original 8×8 (images):
    Matriz de intensidad que permite visualizar cada dígito.

    Matriz de tamaño 1797 × 64, donde:
      - Cada fila corresponde a una imagen de un dígito escrito a mano (0–9).
      - Cada imagen original es una cuadrícula de 8×8 píxeles, en forma de un vector de 64 características.Cada característica (Pixel 0, Pixel 1, …, Pixel 63) contiene un valor entero entre 0 y 16, que representa la intensidad en escala de grises del pixel correspondiente.

- **target**: Un entero entre 0 y 9 indicando el dígito representado.

- **feature_names**: Lista de nombres de las 64 características (píxeles).

- **DESCR**: Descripción completa del dataset.

**Nota**: Este dataset sí contiene una variable objetivo (target = dígito de 0 a 9).


## Estructura de Carpetas

```
PCA/
├── README.md              # Este archivo
├── PCA_USArrets_practical.ipynb  # Notebook ejemplo con análisis y modelo
├── PCA_HandWritting_practical.ipynb  # Notebook para practicar
```

## Objetivos del Análisis

1. **Exploración de Datos (EDA)**:
   - Cargar y examinar el dataset
   - Analizar las magnitudes de las variables
   - Identificar relaciones entre variables

2. **Principal Component Analysis**:
   - Preparar los datos para el análisis con esta técnica
   - Obtener los "projecting vectors" y las nuevas coordenadas
   - Evaluar la varianza explicada

3. **Visualización**:
   - Biplot

## Referencias

   The textbook An Introduction to Statistical Learning
   https://www.statlearning.com/
   
