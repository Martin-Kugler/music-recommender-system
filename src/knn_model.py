import numpy as np
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
        Evalúa el modelo usando el conjunto de test.
        '''
        print('Aún por hacer.')
        return 
    
if __name__ == '__main__': 

    path = '../data/processed/'

    train_set = load_npz(path+'train_set.npz')
    test_set = load_npz(path+'test_set.npz')

    knn_model = KNN_model(train_set, test_set)

    user_id = int(input('Introduzca el índice de un usuario para recomendarle canciones: '))
    recommended_items, scores = knn_model.recommend(user_id)
    print(f'Al usuario {user_id} se le recomiendan las canciones {recommended_items} con reproducciones esperadas de {scores}, respectivamente.')