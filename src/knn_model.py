import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from knn_functions import * # En este caso ejecutamos esta práctica tan poco 
                            # deseable al saber que son pocas las funciones que importamos. 

class KNN_model(): 

    def __init__(self, train_set, test_set, k=20): 
        self.train_set = train_set
        self.test_set = test_set
        self.k = k
    
    def recommend(self, user_id, n_recommendations=5): 
        ''' 
        Genera predicciones y recomendaciones para un usuario específico.
        '''
        predictions = user_prediction(user_id, self.train_set, k=self.k)
        recommended_items, scores = get_recommendations(predictions, N=n_recommendations)
        
        return recommended_items, scores
    
    def evaluate(self):
        ''' 
        Evalúa el modelo usando el conjunto de test y la función predefinida en metrics.py.
        '''
        from metrics import evaluate_model
        
        if not isinstance(self.test_set, pd.DataFrame):
            coo = self.test_set.tocoo()
            test_df = pd.DataFrame({
                'user_n': coo.row,
                'track_n': coo.col,
                'rating': coo.data
            })
        else:
            test_df = self.test_set.copy()
            
        test_df['pred'] = 0.0
        for user_id in test_df['user_n'].unique():
            preds_all = user_prediction(user_id, self.train_set, k=self.k)
            mask = test_df['user_n'] == user_id
            track_ids = test_df.loc[mask, 'track_n'].values
            user_preds = preds_all[track_ids]
            user_preds[np.isinf(user_preds)] = 0.0
            test_df.loc[mask, 'pred'] = user_preds

        return evaluate_model(test_df, test_df['pred'].values)
    

if __name__ == '__main__': 

    path = '../data/processed/'

    train_set = load_npz(path+'train_set.npz')
    test_set = load_npz(path+'test_set.npz')

    knn_model = KNN_model(train_set, test_set)

    option = ''
    
    while True: 
        option = str(input('Introduzca si desea evaluar el modelo (E), si desea recomendar canciones (R) o si desea cerrar sesión (C): '))

        if option == 'E': 
            print('Evaluando modelo...')
            print(knn_model.evaluate())
        
        elif option == 'R': 
            user_id = int(input('Introduzca el índice de un usuario para recomendarle canciones: '))
            recommended_items, scores = knn_model.recommend(user_id)
            print(f'Al usuario {user_id} se le recomiendan las canciones {recommended_items} con reproducciones esperadas de {scores}, respectivamente.')
        
        elif option == 'C': 
            print('Cerrando sesión...')
            break