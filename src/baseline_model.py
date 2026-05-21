# Construcción de un modelo base (baseline model) basado en la media de reproducciones:

import pandas as pd
import numpy as np

class BaselineModel:
    def __init__(self):
        self.item_means = {}
        self.global_mean = 0.0
        self.top_items = []
        
    def fit(self, df_train, item_col='track_n', score_col='rating'):
        """
        Entrena el modelo calculando la media de reproducciones (o ratings) por canción.
        """
        # 11) Calculamos y guardamos la media global (para ítems desconocidos en test):
        self.global_mean = df_train[score_col].mean()
        
        # 2) Calculamos las medias por canción y las pasamos a un diccionario para acceso rápido:
        means_series = df_train.groupby(item_col)[score_col].mean()
        self.item_means = means_series.to_dict()
        
        # 3) Guardamos el Top-N general para recomendaciones puras: 
        self.top_items = means_series.sort_values(ascending=False).index.tolist()
        
    def predict(self, item_id):
        """
        Devuelve la predicción (media) para un ítem específico.
        Si el ítem no existe en el entrenamiento, devuelve la media global.
        """
        return self.item_means.get(item_id, self.global_mean)
    
    def evaluate(self, test_set, item_col='track_n', k=10, threshold=4.0):
        """
        Evalúa el modelo base generando predicciones vectorizadas y llamando 
        a la función externa evaluate_model. Soporta tanto DataFrames como matrices CSR.
        """
        from metrics import evaluate_model
        
        # 1) Control de formato: convertimos matriz CSR a DataFrame si es necesario
        if not isinstance(test_set, pd.DataFrame):
            import scipy.sparse as sp
            if sp.issparse(test_set):
                coo = test_set.tocoo()
                test_df = pd.DataFrame({
                    'user_n': coo.row,
                    'track_n': coo.col,
                    'rating': coo.data
                })
            else:
                raise ValueError("test_set debe ser un DataFrame o una matriz esparsa de SciPy.")
        else:
            test_df = test_set.copy()

        # 2) Generación vectorizada de predicciones
        # .map() busca cada track_n en el diccionario de medias. 
        # Si un track no existe (cold start), devuelve NaN. 
        # .fillna() reemplaza esos NaNs por la media global automáticamente.
        y_pred = test_df[item_col].map(self.item_means).fillna(self.global_mean).values

        # 3) Llamada a la función de métricas externa:
        return evaluate_model(test_df, y_pred, k=k, threshold=threshold)