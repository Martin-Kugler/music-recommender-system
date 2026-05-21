import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix, save_npz
import os

def load_data(train_path, test_path):
    """Carga los datasets de entrenamiento y test."""
    dtype_dict = {
        'track_n': 'uint16',
        'user_n': 'uint16',
        'rating': 'float32' 
    }
    df_train = pd.read_csv(train_path, usecols=['user_n', 'track_n', 'rating'], dtype=dtype_dict)
    df_test = pd.read_csv(test_path, usecols=['user_n', 'track_n', 'rating'], dtype=dtype_dict)
    return df_train, df_test

def create_sparse_matrices(df_train, df_test):
    """Crea las matrices dispersas a partir de los DataFrames."""
    n_users = int(max(df_train['user_n'].max(), df_test['user_n'].max())) + 1
    n_tracks = int(max(df_train['track_n'].max(), df_test['track_n'].max())) + 1

    train_matrix = csr_matrix(
        (df_train['rating'], (df_train['user_n'].astype(int), df_train['track_n'].astype(int))),
        shape=(n_users, n_tracks)
    )

    test_matrix = csr_matrix(
        (df_test['rating'], (df_test['user_n'], df_test['track_n'])),
        shape=(n_users, n_tracks)
    )
    return train_matrix, test_matrix

def save_matrices(train_matrix, test_matrix, output_path):
    """Guarda las matrices en formato npz."""
    os.makedirs(output_path, exist_ok=True)
    save_npz(os.path.join(output_path, 'train_set.npz'), train_matrix)
    save_npz(os.path.join(output_path, 'test_set.npz'), test_matrix)

def main():
    # Definimos rutas relativas asumiendo ejecución desde la carpeta src/ o raíz
    # Ajustamos al path original del notebook
    base_dir = os.path.dirname(os.path.abspath(__file__))
    train_csv_path = os.path.join(base_dir, '../data/processed/train.csv')
    test_csv_path  = os.path.join(base_dir, '../data/processed/test.csv')
    output_dir     = os.path.join(base_dir, '../data/processed/')
    
    print("Cargando datos...")
    df_train, df_test = load_data(train_csv_path, test_csv_path)
    
    print("Creando matrices dispersas (CSR)...")
    train_matrix, test_matrix = create_sparse_matrices(df_train, df_test)
    
    print(f"Dimensiones de train: {train_matrix.shape}")
    print(f"Dimensiones de test: {test_matrix.shape}")
    
    print("Guardando matrices...")
    save_matrices(train_matrix, test_matrix, output_dir)
    print("Proceso completado exitosamente.")

if __name__ == "__main__":
    main()
