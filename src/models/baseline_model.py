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
    
    def evaluate(self, test_df, max_users, item_col='track_n', user_col='user_n', k=10, threshold=4.0):
        """
        Evalúa el modelo base generando predicciones vectorizadas y llamando 
        a la función externa evaluate_model, solo sobre un subconjunto específico de usuarios.
        """
        from metrics import evaluate_model
        
        # 1) Filtrar el test_df para trabajar solo con los usuarios del subconjunto: 
        selected_users = test_df[user_col].unique()[:max_users]
        subset_df = test_df[test_df[user_col].isin(selected_users)].copy()
        
        if subset_df.empty:
            print("El subconjunto de usuarios no tiene interacciones en el set de test.")
            return None

        # 2) Generación vectorizada de predicciones sobre el subconjunto filtrado: 
        y_pred = subset_df[item_col].map(self.item_means).fillna(self.global_mean).values

        # 3) Llamada a la función de métricas externa con el subconjunto: 
        return evaluate_model(subset_df, y_pred, k=k, threshold=threshold)