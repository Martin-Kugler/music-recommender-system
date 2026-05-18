# Music Recommender System
Comparativa de sistemas de recomendación musical empleando el Million Song Dataset. Incluye la implementación, optimización de hiperparámetros y evaluación de 4+ modelos frente a un modelo base (media) usando métricas de rendimiento. Proyecto para Computación Social y Personalización.

## Datos
Datos obtenidos del siguiente dataset de Kaggle:
https://www.kaggle.com/datasets/undefinenull/million-song-dataset-spotify-lastfm

## Estructura

- data

    - processed: datos procesados.
    - raw: datos crudos. 

- notebooks: enfoque experimentativo y explicativo (recomendable para entender el proceso subyacente). 

    - EDA: exploración y análisis de datos, incluyendo limpieza de los mismos. 
    - user_items_matrix: generación de la matriz dispersa usuarios-ítems. 
    - baseline_model: modelo base usado como benchmark. 

    - knn: modelo K-Nearest Neighbours. 
    - matrix_factorization: modelo de Matrix Factorization. 
    - bernoulli_matrix_factorization: modelo de Matrix Factorization de Bernoulli. 
    - neural_collaborative_filtering: modelo de NCF. 

    - analysis: análisis del funcionamiento en métricas de los respectivos modelos. 

- scripts: funciones y modelos modularizados a partir de los notebooks.

    - user_items_matrix.
    - baseline_model.

    - knn_functions: funciones de KNN. 
    - knn_model: modelo K-Nearest Neighbours en formato de clase.
    - knn_functions_optimized: versión optimizada de funciones de KNN modificado con IA. 
    - knn_model_optimized: versión optimizada del modelo K-Nearest Neighbours en formato de clase modificado con IA. 

    - matrix_factorization.
    - bernoulli_matrix_factorization.
    - neural_collaborative_filtering. 

    - metrics: script donde se guardan las principales funciones de métricas para analizar el funcionamiento de nuestros modelos. 

## Ejecución del proyecto: 

Se recomienda ejecutar el proyecto del siguiente modo y en el siguiente orden, priorizando primero la explicación clara en notebooks frente a los scripts, los cuales deberán de ser usados para ejecutar los distintos modelos y funciones: 

1) EDA. 

2) Creación de matrices usuarios-ítems (primero explicación en notebooks y luego ejecutar en src). 

3) Modelo KNN (primero explicación en notebooks y luego ejecutar en src con knn_model_optimized.py (versión optimizada)). 

4) Modelo MF (primero explicación en notebooks y luego ejecutar en src). 

5) Modelo BMF (primero explicación en notebooks y luego ejecutar en src). 

6) Modelo NCF (primero explicación en notebooks y luego ejecutar en src). 

7) Análisis de modelos a partir de las métricas. 