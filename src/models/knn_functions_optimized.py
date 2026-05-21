import numpy as np


def compute_user_means(matrix) -> np.ndarray:
    """Precalcula la media de cada usuario (fila) en una matriz CSR."""
    row_sums = np.asarray(matrix.sum(axis=1)).ravel()
    row_counts = np.diff(matrix.indptr)
    means = np.divide(
        row_sums,
        row_counts,
        out=np.zeros_like(row_sums, dtype=float),
        where=row_counts != 0,
    )
    return means


def compute_user_norms(matrix) -> np.ndarray:
    """Precalcula la norma L2 de cada usuario (fila) en una matriz CSR."""
    # matrix.multiply(matrix) mantiene sparsity; sum(axis=1) es eficiente.
    norms = np.sqrt(np.asarray(matrix.multiply(matrix).sum(axis=1)).ravel())
    norms[norms == 0] = 1.0
    return norms


def build_knn_index(matrix, metric: str = "cosine"):
    """Construye un índice KNN sobre usuarios para consultas rápidas.

    Usa scikit-learn (ya está en dependencias del proyecto). Para matrices sparse,
    el algoritmo 'brute' suele ser la opción más estable.
    """
    from sklearn.neighbors import NearestNeighbors

    knn = NearestNeighbors(metric=metric, algorithm="brute")
    knn.fit(matrix)
    return knn

def cosine_similarity_user(user_idx, matrix, *, all_norms: np.ndarray | None = None):
    """Calcula similitud coseno entre un usuario y todos los demás.

    Nota: esta vía es un fallback. Para evaluar/recomendar a escala, es preferible
    usar `build_knn_index` + `get_k_neighbours(..., knn_index=...)`.
    """
    user_vec = matrix[user_idx]
    user_norm = float(np.sqrt(user_vec.multiply(user_vec).sum()))
    if user_norm == 0:
        user_norm = 1.0

    if all_norms is None:
        all_norms = compute_user_norms(matrix)

    dot_products = (matrix @ user_vec.T).toarray().ravel()
    similarities = dot_products / (user_norm * all_norms)
    return similarities

def get_k_neighbours(
    user_idx,
    matrix,
    k=10,
    *,
    knn_index=None,
    all_norms: np.ndarray | None = None,
):
    """Encuentra los k vecinos más similares a un usuario.

    Si se pasa `knn_index` (NearestNeighbors), se usa para acelerar.
    """
    n_users = matrix.shape[0]
    k_eff = int(min(k, max(n_users - 1, 0)))
    if k_eff <= 0:
        return np.array([], dtype=int), np.array([], dtype=float)

    if knn_index is not None:
        # Pedimos k+1 para incluir al propio usuario y luego lo quitamos.
        n_query = min(k_eff + 1, n_users)
        distances, indices = knn_index.kneighbors(matrix[user_idx], n_neighbors=n_query, return_distance=True)
        indices = indices.ravel()
        distances = distances.ravel()

        mask_not_self = indices != user_idx
        indices = indices[mask_not_self][:k_eff]
        distances = distances[mask_not_self][:k_eff]
        similarities = 1.0 - distances  # distancia coseno -> similitud
        return indices.astype(int, copy=False), similarities.astype(float, copy=False)

    # Camino rápido 100% sparse: producto matriz-vector y top-k solo en no-ceros.
    user_vec = matrix[user_idx]
    user_norm = float(np.sqrt(user_vec.multiply(user_vec).sum()))
    if user_norm == 0:
        user_norm = 1.0

    if all_norms is None:
        all_norms = compute_user_norms(matrix)

    dots = (matrix @ user_vec.T).tocoo(copy=False)
    if dots.nnz == 0:
        return np.array([], dtype=int), np.array([], dtype=float)

    rows = dots.row
    data = dots.data

    mask = rows != user_idx
    rows = rows[mask]
    data = data[mask]

    if rows.size == 0:
        return np.array([], dtype=int), np.array([], dtype=float)

    sims = data / (user_norm * all_norms[rows])

    if sims.size <= k_eff:
        order = np.argsort(sims)[::-1]
        return rows[order].astype(int, copy=False), sims[order].astype(float, copy=False)

    top_idx = np.argpartition(sims, -k_eff)[-k_eff:]
    top_rows = rows[top_idx]
    top_sims = sims[top_idx]
    order = np.argsort(top_sims)[::-1]
    return top_rows[order].astype(int, copy=False), top_sims[order].astype(float, copy=False)


def get_k_neighbours_filtered(
    user_idx: int,
    matrix_csr,
    matrix_csc,
    *,
    k: int = 10,
    all_norms: np.ndarray | None = None,
    item_user_counts: np.ndarray | None = None,
    max_users_per_item: int | None = 647,
    min_common_items: int = 2,
):
    """K vecinos con pre-filtrado de candidatos para acelerar.

    Idea: limitar el conjunto de usuarios candidatos a aquellos que comparten varios
    ítems con el usuario, ignorando ítems demasiado populares.

    Esto reduce drásticamente el coste cuando la similitud u-v es no nula para casi
    todos los usuarios (catálogos pequeños + muchos usuarios).
    """
    n_users = matrix_csr.shape[0]
    k_eff = int(min(k, max(n_users - 1, 0)))
    if k_eff <= 0:
        return np.array([], dtype=int), np.array([], dtype=float)

    user_vec = matrix_csr[user_idx]
    items = user_vec.indices
    if items.size == 0:
        return np.array([], dtype=int), np.array([], dtype=float)

    if all_norms is None:
        all_norms = compute_user_norms(matrix_csr)

    if item_user_counts is None:
        item_user_counts = np.diff(matrix_csc.indptr)

    if max_users_per_item is not None:
        items = items[item_user_counts[items] <= max_users_per_item]
        if items.size == 0:
            return np.array([], dtype=int), np.array([], dtype=float)

    # Pool de candidatos = unión de usuarios que interactuaron con esos ítems.
    indptr = matrix_csc.indptr
    indices = matrix_csc.indices
    pools = []
    for it in items:
        start, end = indptr[it], indptr[it + 1]
        if end > start:
            pools.append(indices[start:end])

    if not pools:
        return np.array([], dtype=int), np.array([], dtype=float)

    candidates_raw = np.concatenate(pools)
    candidates, counts = np.unique(candidates_raw, return_counts=True)

    if min_common_items > 1:
        candidates = candidates[counts >= min_common_items]

    candidates = candidates[candidates != user_idx]
    if candidates.size == 0:
        return np.array([], dtype=int), np.array([], dtype=float)

    user_norm = float(np.sqrt(user_vec.multiply(user_vec).sum()))
    if user_norm == 0:
        user_norm = 1.0

    # Similitud coseno solo contra candidatos.
    dots = (matrix_csr[candidates] @ user_vec.T).toarray().ravel()
    sims = dots / (user_norm * all_norms[candidates])

    if sims.size <= k_eff:
        order = np.argsort(sims)[::-1]
        return candidates[order].astype(int, copy=False), sims[order].astype(float, copy=False)

    top_idx = np.argpartition(sims, -k_eff)[-k_eff:]
    top_users = candidates[top_idx]
    top_sims = sims[top_idx]
    order = np.argsort(top_sims)[::-1]
    return top_users[order].astype(int, copy=False), top_sims[order].astype(float, copy=False)

def mean_users(users, matrix, *, precomputed_means: np.ndarray | None = None):
    """Calcula la media de valoraciones/playcounts de uno o varios usuarios."""
    users = np.atleast_1d(users)
    if precomputed_means is not None:
        return precomputed_means[users]

    rows = matrix[users]
    row_sums = np.asarray(rows.sum(axis=1)).ravel()
    row_counts = np.diff(rows.indptr)
    means = np.divide(
        row_sums,
        row_counts,
        out=np.zeros_like(row_sums, dtype=float),
        where=row_counts != 0,
    )
    return means


def _predict_from_neighbours_matrix(
    neighbours_items_matrix,
    *,
    u_mean: float,
    neighbours_means_local: np.ndarray,
) -> np.ndarray:
    """Predice para un conjunto de ítems dado un sub-matriz (k vecinos x m ítems).

    Implementa: pred(i) = u_mean + mean_j(r_{j,i} - mean_j) sobre vecinos j que
    han valorado el ítem i. Todo se hace sobre sparse (CSC) para evitar densificar.
    """
    sub = neighbours_items_matrix.tocsc(copy=False)
    indptr = sub.indptr
    nnz_per_col = np.diff(indptr)
    m = sub.shape[1]

    preds = np.full(m, -np.inf, dtype=float)
    if sub.nnz == 0:
        return preds

    # sum_ratings por columna (reduceat sobre data)
    data = sub.data
    col_has_nnz = nnz_per_col > 0
    sum_r = np.zeros(m, dtype=float)
    sum_r[col_has_nnz] = np.add.reduceat(data, indptr[:-1][col_has_nnz])

    # sum_means por columna: suma de la media del vecino para cada no-cero
    means_for_nnz = neighbours_means_local[sub.indices]
    sum_mu = np.zeros(m, dtype=float)
    sum_mu[col_has_nnz] = np.add.reduceat(means_for_nnz, indptr[:-1][col_has_nnz])

    # desviación media solo en columnas con al menos 1 vecino que valoró
    preds[col_has_nnz] = u_mean + (sum_r[col_has_nnz] - sum_mu[col_has_nnz]) / nnz_per_col[col_has_nnz]
    return preds

def deviation_from_mean_prediction(
    u,
    i,
    neighbours,
    matrix,
    *,
    precomputed_means: np.ndarray | None = None,
):
    """Versión puntual (compatibilidad). Preferir `user_prediction` vectorizada."""
    neighbours = np.asarray(neighbours)
    if neighbours.size == 0:
        return None

    sub = matrix[neighbours, i].tocoo()
    if sub.nnz == 0:
        return None

    neighbours_means = mean_users(neighbours, matrix, precomputed_means=precomputed_means)
    u_mean = float(mean_users(u, matrix, precomputed_means=precomputed_means)[0])

    # sub.row son índices 0..k-1 (relativos a neighbours)
    dev = (sub.data - neighbours_means[sub.row]).mean()
    return u_mean + float(dev)


def predict_user_items(
    user_idx: int,
    matrix,
    item_ids: np.ndarray,
    *,
    neighbours: np.ndarray,
    precomputed_means: np.ndarray | None = None,
    exclude_seen: bool = True,
) -> np.ndarray:
    """Predice ratings para un usuario en un conjunto de ítems específico."""
    item_ids = np.asarray(item_ids, dtype=int)
    if item_ids.size == 0:
        return np.array([], dtype=float)

    u_mean = float(mean_users(user_idx, matrix, precomputed_means=precomputed_means)[0])
    neighbours = np.asarray(neighbours, dtype=int)
    if neighbours.size == 0:
        return np.full(item_ids.shape[0], -np.inf, dtype=float)

    neighbours_means_local = mean_users(neighbours, matrix, precomputed_means=precomputed_means)
    sub = matrix[neighbours][:, item_ids]
    preds = _predict_from_neighbours_matrix(sub, u_mean=u_mean, neighbours_means_local=neighbours_means_local)

    if exclude_seen:
        seen = matrix[user_idx].indices
        if seen.size:
            preds[np.isin(item_ids, seen, assume_unique=False)] = -np.inf

    return preds

def user_prediction(
    u,
    matrix,
    k=20,
    *,
    knn_index=None,
    precomputed_means: np.ndarray | None = None,
    all_norms: np.ndarray | None = None,
):
    """Predice ratings para un usuario, sin densificar (apto para catálogos grandes)."""
    num_items = matrix.shape[1]
    predictions = np.full(num_items, -np.inf, dtype=float)

    neighbours, _ = get_k_neighbours(u, matrix, k=k, knn_index=knn_index, all_norms=all_norms)
    if neighbours.size == 0:
        return predictions

    # Candidatos: ítems escuchados/valorados por los vecinos y no vistos por el usuario
    neighbours_items = np.unique(matrix[neighbours].indices)
    seen = matrix[u].indices
    if seen.size:
        candidates = np.setdiff1d(neighbours_items, seen, assume_unique=False)
    else:
        candidates = neighbours_items

    if candidates.size == 0:
        return predictions

    preds_candidates = predict_user_items(
        u,
        matrix,
        candidates,
        neighbours=neighbours,
        precomputed_means=precomputed_means,
        exclude_seen=False,  # ya filtramos candidates
    )
    predictions[candidates] = preds_candidates
    return predictions


def recommend_user(
    user_idx: int,
    matrix,
    *,
    k: int = 20,
    n_recommendations: int = 5,
    knn_index=None,
    precomputed_means: np.ndarray | None = None,
    all_norms: np.ndarray | None = None,
):
    """Recomienda top-N sin construir un vector de predicción de tamaño num_items."""
    neighbours, _ = get_k_neighbours(user_idx, matrix, k=k, knn_index=knn_index, all_norms=all_norms)
    if neighbours.size == 0:
        return np.array([], dtype=int), np.array([], dtype=float)

    neighbours_items = np.unique(matrix[neighbours].indices)
    seen = matrix[user_idx].indices
    if seen.size:
        candidates = np.setdiff1d(neighbours_items, seen, assume_unique=False)
    else:
        candidates = neighbours_items

    if candidates.size == 0:
        return np.array([], dtype=int), np.array([], dtype=float)

    preds = predict_user_items(
        user_idx,
        matrix,
        candidates,
        neighbours=neighbours,
        precomputed_means=precomputed_means,
        exclude_seen=False,
    )
    preds[np.isnan(preds)] = -np.inf
    valid_mask = preds != -np.inf
    if not np.any(valid_mask):
        return np.array([], dtype=int), np.array([], dtype=float)

    valid_candidates = candidates[valid_mask]
    valid_preds = preds[valid_mask]
    n = int(min(n_recommendations, valid_preds.size))
    top_idx = np.argpartition(valid_preds, -n)[-n:]
    order = np.argsort(valid_preds[top_idx])[::-1]
    return valid_candidates[top_idx][order], valid_preds[top_idx][order]

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