# Music Recommender System
Comparativa de sistemas de recomendación musical empleando el Million Song Dataset. Incluye la implementación, optimización de hiperparámetros y evaluación de 4+ modelos frente a un modelo base (media) usando métricas de rendimiento. Proyecto para Computación Social y Personalización.

## Datos
Datos obtenidos del siguiente dataset de Kaggle:
https://www.kaggle.com/datasets/undefinenull/million-song-dataset-spotify-lastfm

## Estructura

- data
    - processed: datos procesados.
    - raw: datos crudos. 

- notebooks: enfoque experimentativo
    - base_model: modelo base usado como benchmark. 
    - exploratory_data_analysis: obtención de datos crudos y creación de matriz users-items. 
    - knn_algorithm: algoritmo KNN sobre matriz users-items. 
    - knn_algorithm_improved: algoritmo KNN con eficiencia mejorada. 
