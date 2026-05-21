"""
Módulo de preparación de datos para evaluación de métricas de Ranking.
Implementa la estrategia "Leave-One-Out" con Negative Sampling (1 positivo + N negativos).

Mejoras respecto a la versión original:
- Vectorizado con numpy: elimina el iterrows() fila a fila (inviable a 1.9M filas).
- Muestreo de usuarios: evalúa sobre una muestra representativa en lugar del test completo.
- Solo evalúa positivos con rating >= threshold, garantizando que las métricas de ranking
  tienen sentido (si el positivo no es relevante, la evaluación es ruido).
"""

import numpy as np
import pandas as pd


def preparar_datos_evaluacion(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    num_items_totales: int,
    num_negativos: int = 99,
    n_usuarios_muestra: int = 1000,
    threshold: float = 4.0,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Genera el dataset de evaluación con Negative Sampling vectorizado.

    Para evaluar la capacidad de *ranking* de un recomendador, escondemos cada
    ítem relevante del usuario entre 'num_negativos' ítems que nunca ha visto,
    y comprobamos si el modelo es capaz de colocarlo en el Top-K.

    Solo se incluyen interacciones positivas con rating >= threshold, para
    garantizar que el ítem "oculto" es genuinamente relevante para el usuario.

    Args:
        df_train           : DataFrame de entrenamiento (para excluir ítems vistos).
        df_test            : DataFrame de test.
        num_items_totales  : Tamaño total del catálogo.
        num_negativos      : Negativos por positivo. Por defecto 99.
        n_usuarios_muestra : Usuarios a muestrear del test. Por defecto 1000.
        threshold          : Rating mínimo para considerar un positivo relevante.
        seed               : Semilla de reproducibilidad.

    Returns:
        DataFrame con columnas 'user_n', 'track_n', 'rating' listo para inferencia.
        Los positivos conservan su rating real; los negativos tienen rating 0.0.
    """
    rng = np.random.default_rng(seed)
    todos_los_items = np.arange(num_items_totales, dtype=np.int32)

    # Items vistos en train por usuario
    vistos_train = df_train.groupby('user_n')['track_n'].apply(np.array).to_dict()

    # Filtrar test: solo positivos relevantes
    df_positivos = df_test[df_test['rating'] >= threshold].copy()

    if df_positivos.empty:
        raise ValueError(
            f"No hay interacciones con rating >= {threshold} en test. "
            "Baja el threshold o revisa la escala del rating."
        )

    # Muestrear usuarios
    usuarios_disponibles = df_positivos['user_n'].unique()
    n_muestra = min(n_usuarios_muestra, len(usuarios_disponibles))
    usuarios_muestra = rng.choice(usuarios_disponibles, n_muestra, replace=False)
    df_positivos = df_positivos[df_positivos['user_n'].isin(usuarios_muestra)]

    print(f"Evaluando sobre {n_muestra:,} usuarios | "
          f"{len(df_positivos):,} positivos (rating >= {threshold}) | "
          f"{num_negativos} negativos por positivo")

    partes = [df_positivos[['user_n', 'track_n', 'rating']].reset_index(drop=True)]

    for u, grupo in df_positivos.groupby('user_n'):
        # Items prohibidos = vistos en train + los propios positivos del usuario en test
        prohibidos = np.union1d(
            vistos_train.get(u, np.array([], dtype=np.int32)),
            grupo['track_n'].values
        )
        candidatos = np.setdiff1d(todos_los_items, prohibidos, assume_unique=True)

        n = min(num_negativos * len(grupo), len(candidatos))
        if n == 0:
            continue

        negativos = rng.choice(candidatos, n, replace=False)
        partes.append(pd.DataFrame({
            'user_n':  np.full(n, u, dtype=np.int32),
            'track_n': negativos.astype(np.int32),
            'rating':  np.zeros(n, dtype=np.float32),
        }))

    df_eval = pd.concat(partes, ignore_index=True)
    print(f"Dataset de evaluación listo: {len(df_eval):,} filas "
          f"({len(df_positivos):,} positivos + {len(df_eval) - len(df_positivos):,} negativos)")
    return df_eval
