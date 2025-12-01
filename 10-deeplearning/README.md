# 10 - Deep Learning: Reducción de Dimensionalidad y Procesamiento de Lenguaje Natural

Este módulo contiene dos ejemplos prácticos de deep learning:

1. **Comparación de PCA vs Autoencoder** para reducción de dimensionalidad
2. **Introducción a Hugging Face** para procesamiento de lenguaje natural

## 📚 Contenidos

### 1. PCA_vs_Autoencoder.ipynb
**Comparación de PCA vs Autoencoder para Reducción de Dimensionalidad**

Este notebook compara dos técnicas de reducción de dimensionalidad aplicadas a datos de presión atmosférica (MSL - Mean Sea Level Pressure):

#### Objetivos:
- Entender las diferencias entre PCA (método lineal) y Autoencoders (método no lineal)
- Aplicar ambas técnicas a datos climáticos reales
- Comparar la calidad de reconstrucción de ambos métodos
- Analizar la correlación entre las representaciones latentes

#### Contenido:
1. **Carga de datos**: Datos de presión atmosférica ERA5 del Océano Pacífico
2. **Aplicación de PCA**: 
   - Reducción de dimensionalidad manteniendo el 90% de la varianza
   - Visualización de componentes principales (EOFs)
   - Reconstrucción de datos originales
3. **Aplicación de Autoencoder CNN**:
   - Preparación de datos para redes neuronales
   - Entrenamiento de autoencoder con espacio latente de dimensión 15
   - Obtención de representación latente
   - Reconstrucción de datos
4. **Comparación de métodos**:
   - Visualización de errores de reconstrucción espacial
   - Análisis de correlación entre componentes PCA y dimensiones latentes del autoencoder

#### Requisitos:
- `bluemath_tk` (para PCA y Autoencoders)
- `xarray` (para datos climáticos)
- `cartopy` (para visualización geográfica)
- `numpy`, `pandas`, `matplotlib`, `seaborn`

#### Conceptos clave:
- **PCA (Principal Component Analysis)**: Método lineal de reducción de dimensionalidad basado en álgebra lineal
- **Autoencoder**: Red neuronal que aprende representaciones comprimidas de los datos
- **Espacio latente**: Representación de menor dimensión que captura las características más importantes
- **Reconstrucción**: Recuperar los datos originales desde la representación comprimida

---

### 2. what_is_hugging_face.ipynb
**Introducción a Hugging Face y Procesamiento de Lenguaje Natural**

Este notebook proporciona una introducción práctica a Hugging Face, la plataforma líder para modelos de IA pre-entrenados, especialmente en procesamiento de lenguaje natural (NLP).

#### Objetivos:
- Comprender qué es Hugging Face y su ecosistema
- Usar modelos pre-entrenados para diferentes tareas de NLP
- Aplicar transformers a problemas reales de análisis de texto
- Integrar modelos de IA con análisis de datos

#### Contenido:

**1. Introducción a Hugging Face**
- ¿Qué es Hugging Face?
- ¿Por qué es importante para análisis de datos?
- Características principales de la plataforma

**2. Casos de uso prácticos con código ejecutable:**

   - **3.1. Análisis de Sentimientos**
     - Detectar si un texto expresa opinión positiva o negativa
     - Ejemplos con múltiples textos
   
   - **3.2. Clasificación Zero-Shot**
     - Clasificar texto en categorías personalizadas sin entrenar
     - Útil cuando no tienes datos etiquetados
   
   - **3.3. Generación de Resúmenes**
     - Condensar textos largos manteniendo información clave
     - Usando modelos BART
   
   - **3.4. Análisis de Sentimientos en Español**
     - Modelos específicos para español
     - Mejores resultados con textos en español
   
   - **3.5. Question Answering**
     - Responder preguntas basándose en un contexto
     - Útil para sistemas de búsqueda y chatbots

**3. Instalación y configuración**
- Verificación de librerías instaladas
- Tu primer pipeline de Hugging Face

**4. Conexión con proyectos existentes**
- Aplicaciones con datos de COVID-19
- Integración con análisis de datos HAR
- Procesamiento de logs de sensores

#### Requisitos:
```bash
pip install transformers torch datasets accelerate
```