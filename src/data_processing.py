
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix, save_npz

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def process_user_listening_history(input_path=None, seed=77): 
    '''
    Carga el historial, realiza el split de interacciones y genera matrices CSR.
    '''
    # 1) Carga optimizada del historial de escuchas (Listening History): 
    dtype_dict = {
        'track_id': 'category',
        'user_id': 'category',
        'playcount': 'uint16' 
    }
    
    if input_path is None:
        input_path = PROJECT_ROOT / 'data' / 'raw' / 'User Listening History.csv'
    else:
        input_path = Path(input_path)

    df_history = pd.read_csv(input_path, dtype=dtype_dict)

    # 2) Creamos la matriz usuarios-items mediante la Sparse Matrix de Scipy, CSR:

    # Obtener dimensiones e índices:
    num_users = df_history.user_id.nunique()
    num_items = df_history.track_id.nunique()

    row_indices = df_history['user_id'].cat.codes.values
    col_indices = df_history['track_id'].cat.codes.values
    playcounts = df_history['playcount'].values

    # Determinar el número total de interacciones reales: 
    n_interactions = len(playcounts)

    # Seleccionar aleatoriamente el 20% de las interacciones para el test: 
    np.random.seed(seed)
    test_size = int(n_interactions * 0.2)

    # Elegimos índices aleatorios del array unidimensional de interacciones
    test_indices = np.random.choice(n_interactions, test_size, replace=False)

    # Crear máscaras booleanas para test y train: 
    test_mask = np.zeros(n_interactions, dtype=bool)
    test_mask[test_indices] = True
    train_mask = ~test_mask

    # 2.1) Generar el set de entrenamiento (80% de las reproducciones): 
    train_set = csr_matrix(
        (playcounts[train_mask], (row_indices[train_mask], col_indices[train_mask])), 
        shape=(num_users, num_items)
    )

    # 2.2) Generar el set de validación (20% de las reproducciones): 
    test_set = csr_matrix(
        (playcounts[test_mask], (row_indices[test_mask], col_indices[test_mask])), 
        shape=(num_users, num_items)
    )
    
    return train_set, test_set

def save_matrices(train_matrix, test_matrix, folder=None):
    '''
    Guarda las matrices en formato .npz
    '''
    if folder is None:
        folder = PROJECT_ROOT / 'data' / 'processed'
    else:
        folder = Path(folder)

    if not folder.exists():
        folder.mkdir(parents=True, exist_ok=True)
    
    save_npz(folder / 'train_set.npz', train_matrix)
    save_npz(folder / 'test_set.npz', test_matrix)

    print(f"Matrices guardadas en {folder}.")

if __name__ == "__main__":
    train, test = process_user_listening_history()
    save_matrices(train, test)