# Construcción de un modelo base (baseline model) basado en la media de reproducciones:

import pandas as pd

class BaselineModel:
    def __init__(self):
        self.item_means = None
        
    def fit(self, df_train, item_col, score_col):
        """
        Entrena el modelo calculando la media de reproducciones por canción.
        """
        # Calculamos la media de reproducciones de cada canción:
        means = df_train.groupby(item_col)[score_col].mean().reset_index()
        
        # Ordenamos de mayor a menor media:
        self.item_means = means.sort_values(by=score_col, ascending=False)
        
    def recommend(self, user_id, n=10):
        """
        Devuelve el Top-N basado en las medias más altas.
        Al no estar personalizado, devuelve lo mismo para todos los usuarios.
        """
        return self.item_means.head(n)['track_id'].tolist()