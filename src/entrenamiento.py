"""
Motor de entrenamiento universal para modelos Neural Collaborative Filtering (NCF).
Agnóstico a la arquitectura (GMF, MLP, NeuMF).
"""

import torch
import torch.nn as nn
import time
import copy


def entrenar_modelo_universal(modelo, train_loader, lr=0.01, num_epocas=15, device='cpu', verbose=True):
    """
    Entrena el modelo durante num_epocas y devuelve el modelo + historial de loss en train.

    Args:
        modelo (nn.Module): Arquitectura de red neuronal a entrenar.
        train_loader (DataLoader): Datos de entrenamiento.
        lr (float): Tasa de aprendizaje para Adam.
        num_epocas (int): Número de épocas de entrenamiento.
        device (str): 'cpu' o 'cuda'.
        verbose (bool): Imprimir trazas de entrenamiento.

    Returns:
        modelo: El modelo entrenado.
        historial_train: Lista con la evolución de la loss por época.
    """
    modelo = modelo.to(device)
    optimizer = torch.optim.Adam(modelo.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    historial_train = []

    for epoch in range(num_epocas):
        start_time = time.time()
        loss_total, n_muestras = 0, 0

        modelo.train()
        for users, items, ratings in train_loader:
            users   = users.to(device)
            items   = items.to(device)
            ratings = ratings.to(device)

            preds = modelo(users, items)
            loss  = loss_fn(preds, ratings)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            loss_total += loss.item() * len(ratings)
            n_muestras += len(ratings)

        loss_train = loss_total / n_muestras
        historial_train.append(loss_train)

        if verbose:
            print(f"Epoca {epoch+1:02d}/{num_epocas:02d} | train_loss: {loss_train:.4f} ({time.time()-start_time:.1f}s)")

    return modelo, historial_train
