"""
Métricas de evaluación para modelos de filtrado colaborativo.

Decisiones del equipo:
- Umbral relevante: rating >= 4.0
- K para top-K: 10 (configurable)
- Usuarios sin relevantes en test: se saltan
- Precision@K: hits/K (no se ajusta K)
- Evaluación estricta (con Negative Sampling implementado en el pipeline exterior)
"""

import numpy as np
import pandas as pd

def metrica_mae(prediccion: np.ndarray, real: np.ndarray) -> float:
    """
    Calcula el Error Absoluto Medio (MAE) entre las predicciones y los valores reales.
    El MAE mide el promedio de los errores absolutos. Es facil de interpretar 
    porque esta en la misma escala que las calificaciones originales (estrellas).
    A diferencia del RMSE, no penaliza de forma desproporcionada los errores grandes.

    Args:
        prediccion (np.ndarray): Array con las calificaciones predichas por el modelo.
        real (np.ndarray): Array con las calificaciones reales dadas por los usuarios.

    Returns:
        float: El valor del MAE. Un valor de 0.0 indica una prediccion perfecta.
    """
    if len(prediccion) != len(real):
        raise ValueError("El array de predicción y el real deben ser del mismo tamaño")
    
    return np.mean(np.abs(real - prediccion))


def metrica_rmse(prediccion: np.ndarray, real: np.ndarray) -> float:
    """
    Calcula la Raiz del Error Cuadratico Medio (RMSE). 
    El RMSE eleva al cuadrado los errores antes de promediarlos, lo que significa 
    que penaliza fuertemente las desviaciones grandes. Es la metrica preferida 
    cuando los errores severos son especialmente indeseables.

    Args:
        prediccion (np.ndarray): Array con las calificaciones predichas por el modelo.
        real (np.ndarray): Array con las calificaciones reales dadas por los usuarios.

    Returns:
        float: El valor del RMSE.
    """
    if len(prediccion) != len(real):
        raise ValueError("El array de predicción y el real deben ser del mismo tamaño")
    
    return np.sqrt(np.mean((real - prediccion) ** 2))


def precision_recall_at_k(test_df: pd.DataFrame, k: int = 10, threshold: float = 4.0) -> tuple:
    """
    Calcula la Precision@K y el Recall@K promediado para todos los usuarios.
    - Precision@K: De los K items recomendados, que proporcion le gusto al usuario.
    - Recall@K: De todos los items que le gustaron al usuario en el dataset, 
        que proporcion logramos incluir en el Top-K.
    - Los usuarios que no tienen ningun item relevante (rating >= threshold) en 
        sus datos de test son omitidos para evitar divisiones por cero en el recall.

    Args:
        test_df (pd.DataFrame): DataFrame que debe contener las columnas 'user_n', 
            'rating' (calificacion real) y 'pred' (calificacion predicha).
        k (int, optional): Numero maximo de recomendaciones a evaluar por usuario. 
            Por defecto es 10.
        threshold (float, optional): Calificacion minima para considerar un item 
            como "relevante" para el usuario. Por defecto es 4.0.

    Returns:
        tuple: Una tupla que contiene (mean_precision, mean_recall).
    """
    precisiones, recalls = [], []
    for user, grupo in test_df.groupby('user_n'):
        # Verificamos si el usuario tiene items relevantes para no penalizar artificialmente
        total_rel = (grupo['rating'] >= threshold).sum()
        
        if total_rel == 0:
            continue
            
        top_k = grupo.nlargest(k, 'pred')
        hits = (top_k['rating'] >= threshold).sum()
        
        precisiones.append(hits / k)
        recalls.append(hits / total_rel)
        
    if not precisiones:
        return 0.0, 0.0
        
    return float(np.mean(precisiones)), float(np.mean(recalls))


def f1_at_k(precision: float, recall: float) -> float:
    """
    Calcula el F1-Score para un K dado, usando la media armonica de Precision y Recall.
    El F1-Score es util cuando se busca un equilibrio entre devolver recomendaciones 
    precisas y no dejarse atras items relevantes.

    Args:
        precision (float): El valor de Precision@K calculado previamente.
        recall (float): El valor de Recall@K calculado previamente.

    Returns:
        float: El valor del F1-Score. Retorna 0.0 si la suma de precision y recall es 0.
    """
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def ndcg_at_k(test_df: pd.DataFrame, k: int = 10, threshold: float = 4.0) -> float:
    """
    Calcula la Ganancia Acumulada Descontada Normalizada (NDCG@K) promediada por usuario.
    A diferencia de la Precision, el NDCG asume que el orden de las recomendaciones 
    importa.
    Una recomendacion perfecta obtiene un NDCG de 1.0.

    Args:
        test_df (pd.DataFrame): DataFrame con las columnas 'user_n', 'rating', y 'pred'.
        k (int, optional): Longitud de la lista de recomendaciones. Por defecto es 10.
        threshold (float, optional): Calificacion minima para considerar un item 
            como "relevante". Por defecto es 4.0.

    Returns:
        float: El valor medio de NDCG@K.
    """
    ndcgs = []
    for user, group in test_df.groupby('user_n'):
        total_rel = (group['rating'] >= threshold).sum()
        if total_rel == 0:
            continue
            
        # Relevancia binaria para evitar que items basura sumen puntos al NDCG
        group = group.copy()
        group['relevance'] = (group['rating'] >= threshold).astype(int)

        top_k = group.nlargest(k, 'pred')
        gains = (2 ** top_k['relevance'].values - 1)
        discounts = np.log2(np.arange(2, len(gains) + 2))
        dcg = np.sum(gains / discounts)

        ideal = group.nlargest(k, 'relevance')
        ideal_gains = (2 ** ideal['relevance'].values - 1)
        ideal_discounts = np.log2(np.arange(2, len(ideal_gains) + 2))
        idcg = np.sum(ideal_gains / ideal_discounts)
        
        if idcg > 0:
            ndcgs.append(dcg / idcg)
        else:
            ndcgs.append(0.0)
            
    if not ndcgs:
        return 0.0
        
    return float(np.mean(ndcgs))


def evaluate_model(test_df: pd.DataFrame, y_pred: np.ndarray, k: int = 10, threshold: float = 4.0) -> dict:
    """
    Ejecuta un pipeline de evaluacion completo sobre las predicciones de un modelo.

    Args:
        test_df (pd.DataFrame): DataFrame de validacion/test.
        y_pred (np.ndarray): Predicciones generadas por el modelo para test_df.
        k (int, optional): Tamaño del top-K a evaluar. Por defecto es 10.
        threshold (float, optional): Umbral de relevancia. Por defecto es 4.0.

    Returns:
        dict: Diccionario que contiene todas las metricas calculadas (MAE, RMSE, Precision@K, Recall@K, F1@K, NDCG@K).
    """
    df = test_df.copy()
    df['pred'] = y_pred
    
    mae = metrica_mae(y_pred, df['rating'].values)
    rmse = metrica_rmse(y_pred, df['rating'].values)
    precision, recall = precision_recall_at_k(df, k, threshold)
    f1 = f1_at_k(precision, recall)
    ndcg = ndcg_at_k(df, k, threshold)
    
    return {
        'mae': mae,
        'rmse': rmse,
        f'precision@{k}': precision,
        f'recall@{k}': recall,
        f'f1@{k}': f1,
        f'ndcg@{k}': ndcg
    }