import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from typing import Any


def _to_int(x: Any) -> int:
    """Conversión robusta a int (incluye np/pandas scalars)."""
    try:
        return int(x)
    except Exception:
        return int(np.asarray(x).item())

from knn_functions_optimized import (
    compute_user_means,
    compute_user_norms,
    get_k_neighbours,
    get_k_neighbours_filtered,
    predict_user_items,
    recommend_user,
)

class KNN_model(): 

    def __init__(self, train_set, test_set, k=20, eval_max_users: int | None = 500): 
        # Aseguramos CSR para que sum/indptr sean eficientes.
        self.train_set = train_set.tocsr() if hasattr(train_set, "tocsr") else train_set
        self.test_set = test_set
        self.k = k
        self.eval_max_users = eval_max_users

        # Cachés para acelerar: medias por usuario + índice KNN.
        self._user_means = compute_user_means(self.train_set)
        self._all_norms = compute_user_norms(self.train_set)

        # Estructuras auxiliares para filtrar candidatos por ítem (acelera la evaluación).
        self._train_csc = self.train_set.tocsc()
        self._item_user_counts = np.diff(self._train_csc.indptr)
        # Heurísticas: ignora ítems muy populares y exige >=2 ítems en común.
        self._max_users_per_item = 647
        self._min_common_items = 2
    
    def recommend(self, user_id, n_recommendations=5): 
        ''' 
        Genera predicciones y recomendaciones para un usuario específico.
        '''
        return recommend_user(
            int(user_id),
            self.train_set,
            k=self.k,
            n_recommendations=int(n_recommendations),
            precomputed_means=self._user_means,
            all_norms=self._all_norms,
        )
    
    def evaluate(self, max_users=None):
        ''' 
        Evalúa el modelo usando el conjunto de test y la función predefinida en metrics.py.
        '''
        if max_users is None: 
            return self.evaluate_subset(max_users=self.eval_max_users, random_state=42)
        else: 
            return self.evaluate_subset(max_users, random_state=42)

    def evaluate_subset(self, max_users: int | None = 500, random_state: int = 42):
        '''
        Evalúa el modelo sobre un subconjunto de usuarios para obtener una estimación rápida.

        Args:
            max_users: Nº máximo de usuarios únicos a evaluar. Si es None, evalúa todos.
            random_state: Semilla para que el muestreo sea reproducible.
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
            
        test_df = test_df.copy()

        # Submuestreo por usuario (acelera muchísimo cuando el test es enorme)
        if max_users is not None:
            unique_users = test_df['user_n'].unique()
            if unique_users.size > max_users:
                rng = np.random.default_rng(random_state)
                sampled_users = rng.choice(unique_users, size=int(max_users), replace=False)
                test_df = test_df[test_df['user_n'].isin(sampled_users)].copy()

        print(f"Evaluando subset: {test_df['user_n'].nunique()} usuarios, {len(test_df)} interacciones")

        test_df['pred'] = 0.0

        grouped = test_df.groupby('user_n', sort=False)
        for user_id, group in grouped:
            user_id = _to_int(user_id)

            neigh, _ = get_k_neighbours_filtered(
                user_id,
                self.train_set,
                self._train_csc,
                k=self.k,
                all_norms=self._all_norms,
                item_user_counts=self._item_user_counts,
                max_users_per_item=self._max_users_per_item,
                min_common_items=self._min_common_items,
            )
            if neigh.size == 0:
                neigh, _ = get_k_neighbours(
                    user_id,
                    self.train_set,
                    k=self.k,
                    all_norms=self._all_norms,
                )

            track_ids = group['track_n'].to_numpy(dtype=int)
            user_preds = predict_user_items(
                user_id,
                self.train_set,
                track_ids,
                neighbours=neigh,
                precomputed_means=self._user_means,
                exclude_seen=True,
            )

            # Penalizamos ítems sin predicción.
            user_preds[np.isinf(user_preds)] = 0.0
            test_df.loc[group.index, 'pred'] = user_preds

        return evaluate_model(test_df, test_df['pred'].to_numpy(dtype=float))
    

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
            max_users = int(input('Introduzca el número del subconjunto usuarios que se desea evaluar (unos 200 para rapidez pero poca precisión, unos 2000 para precisión con rapidez media, mayor que 2000 para precisión con lentitud (aproximadamente) y None para el valor predeterminado): '))
            print('Evaluación aproximada (subset de usuarios)')
            print(knn_model.evaluate(max_users))
        
        elif option == 'R': 
            user_id = int(input('Introduzca el índice de un usuario para recomendarle canciones: '))
            recommended_items, scores = knn_model.recommend(user_id)
            print(f'Al usuario {user_id} se le recomiendan las canciones {recommended_items} con reproducciones esperadas de {scores}, respectivamente.')
        
        elif option == 'C': 
            print('Cerrando sesión...')
            break