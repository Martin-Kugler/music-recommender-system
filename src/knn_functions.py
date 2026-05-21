import numpy as np

def cosine_similarity_user(user_idx, matrix):
    """
    Calcula similitud coseno entre un usuario y todos los demás.
    """
    user_vec = matrix[user_idx] 
    user_norm = np.sqrt(np.array(user_vec.power(2).sum()))
    if user_norm == 0:
        user_norm = 1.0
    
    all_norms = np.sqrt(np.array(matrix.power(2).sum(axis=1)).flatten())
    all_norms[all_norms == 0] = 1.0
    dot_products = (matrix @ user_vec.T).toarray().flatten()
    similarities = dot_products / (user_norm * all_norms)
    
    return similarities

def get_k_neighbours(user_idx, matrix, k=10):
    """
    Encuentra los k vecinos más similares a un usuario.
    """
    similarities = cosine_similarity_user(user_idx, matrix) 
    similarities[user_idx] = -1.0
     
    top_k_indices = np.argpartition(similarities, -k)[-k:]
    top_k_similarities = similarities[top_k_indices]
    sort_order = np.argsort(top_k_similarities)[::-1]
    
    return top_k_indices[sort_order], top_k_similarities[sort_order]

def mean_users(users, matrix): 
    """
    Calcula la media de valoraciones/playcounts de uno o varios usuarios.
    """
    users = np.atleast_1d(users)
    rows = matrix[users]
    row_sums = np.asarray(rows.sum(axis=1)).ravel()
    row_counts = np.diff(rows.indptr)
    means = np.divide(row_sums, row_counts, out=np.zeros_like(row_sums, dtype=float), where=row_counts!=0)
    return means

def deviation_from_mean_prediction(u, i, neighbours, matrix): 
    """
    Función vectorizada que calcula la predicción de la valoración de un ítem i 
    por parte de un usuario u según sus k vecinos.
    """
    sub_matrix = matrix[neighbours, i]
    relevant_idx = sub_matrix.nonzero()[0]
    n = relevant_idx.size
    if n == 0:
        return None

    actual_relevant_neighbours = neighbours[relevant_idx]
    relevant_neighbours_means = mean_users(actual_relevant_neighbours, matrix)
    u_mean = mean_users(u, matrix)[0]
    ratings_i = matrix[actual_relevant_neighbours, i].toarray().ravel()
    
    deviation = (ratings_i - relevant_neighbours_means).sum() / n
    prediction = u_mean + deviation

    return prediction

def user_prediction(u, matrix, k=20):
    """
    Calcula las predicciones de ítems para un usuario u basándose en sus k vecinos.
    """
    num_items = matrix.shape[1]
    predictions = np.full(num_items, -np.inf) 
    neighbors, _ = get_k_neighbours(u, matrix, k=k)
    if len(neighbors) == 0:
        return predictions
    
    neighbours_items = np.unique(matrix[neighbors].indices)
    user_seen_items = matrix[u].indices
    items_to_predict = np.setdiff1d(neighbours_items, user_seen_items)
     
    for i in items_to_predict:
        pred = deviation_from_mean_prediction(u, i, neighbors, matrix)
        if pred is not None:
            predictions[i] = pred
            
    return predictions

def get_recommendations(predictions, N=5): 
    ''' 
    Función que devuelve las k mejores recomendaciones según las estimaciones predichas. 
    '''
    preds_array = np.asarray(predictions, dtype=float)
    preds_array[np.isnan(preds_array)] = -np.inf
    valid_count = np.sum(preds_array != -np.inf)
    actual_n = min(N, valid_count)
    if actual_n == 0:
        return np.array([]), np.array([])
        
    top_n_indices = np.argpartition(preds_array, -actual_n)[-actual_n:]
    top_n_values = preds_array[top_n_indices]
    sort_order = np.argsort(top_n_values)[::-1]
    
    final_items = top_n_indices[sort_order]
    final_scores = top_n_values[sort_order]
    
    return final_items, final_scores