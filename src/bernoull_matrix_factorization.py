import numpy as np
import math
from numba import njit

@njit
def _train_epoch(U, V, B_u, B_i, B_s, train_u, train_i, train_r, scores, lr, reg):
    n_samples = len(train_u)
    for idx in range(n_samples):
        u = train_u[idx]
        i = train_i[idx]
        r = train_r[idx]
        
        for s_idx in range(len(scores)):
            s_val = scores[s_idx]
            
            dot = B_s[s_idx] + B_u[s_idx, u] + B_i[s_idx, i] + np.dot(U[s_idx, u], V[s_idx, i])
            
            if dot > 20: prob = 1.0
            elif dot < -20: prob = 0.0
            else: prob = 1.0 / (1.0 + math.exp(-dot))
            
            # Ordinal Label Smoothing
            diff = abs(r - s_val)
            if diff == 0: target = 1.0
            elif diff == 1: target = 0.4
            elif diff == 2: target = 0.1
            else: target = 0.0
                
            err = target - prob
            
            B_s[s_idx] += lr * (err - reg * B_s[s_idx])
            B_u[s_idx, u] += lr * (err - reg * B_u[s_idx, u])
            B_i[s_idx, i] += lr * (err - reg * B_i[s_idx, i])
            
            U_upd = err * V[s_idx, i] - reg * U[s_idx, u]
            V_upd = err * U[s_idx, u] - reg * V[s_idx, i]
            
            U[s_idx, u] += lr * U_upd
            V[s_idx, i] += lr * V_upd


@njit
def _predict(U, V, B_u, B_i, B_s, test_u, test_i, scores, g_mean):
    n_samples = len(test_u)
    preds = np.zeros(n_samples)
    
    for idx in range(n_samples):
        u = test_u[idx]
        i = test_i[idx]
        
        if u == -1 or i == -1 or u >= U.shape[1] or i >= V.shape[1]:
            preds[idx] = g_mean
            continue
            
        probs = np.zeros(len(scores))
        sum_probs = 0.0
        
        for s_idx in range(len(scores)):
            dot = B_s[s_idx] + B_u[s_idx, u] + B_i[s_idx, i] + np.dot(U[s_idx, u], V[s_idx, i])
            
            if dot > 20: prob = 1.0
            elif dot < -20: prob = 0.0
            else: prob = 1.0 / (1.0 + math.exp(-dot))
            
            probs[s_idx] = prob
            sum_probs += prob
        
        # Valor esperado (no argmax)
        expected_value = 0.0
        if sum_probs > 0:
            for s_idx in range(len(scores)):
                expected_value += scores[s_idx] * (probs[s_idx] / sum_probs)
        else:
            expected_value = g_mean
            
        preds[idx] = expected_value
        
    return preds
class BeMFModel:
    def __init__(self, num_factors, scores, B_s_init, num_users, num_items, g_mean):
        self.nf = num_factors
        self.scores = scores
        self.g_mean = g_mean
        
        # Inicialización de estados
        self.U = np.random.normal(scale=0.01, size=(len(scores), num_users, num_factors))
        self.V = np.random.normal(scale=0.01, size=(len(scores), num_items, num_factors))
        self.B_u = np.zeros((len(scores), num_users))
        self.B_i = np.zeros((len(scores), num_items))
        self.B_s = B_s_init.copy()
        
    def fit(self, train_data, test_data, num_epochs, lr_inicial, reg, decay=0.1, paciencia=3):
        train_u, train_i, train_r = train_data
        test_u, test_i, test_r = test_data
        
        history = []
        mejor_rmse = float('inf')
        
        for it in range(num_epochs):
            current_lr = lr_inicial / (1.0 + decay * it)
            
            # Entrenamiento
            _train_epoch(self.U, self.V, self.B_u, self.B_i, self.B_s, 
                         train_u, train_i, train_r, self.scores, current_lr, reg)
            
            # Predicción
            preds = _predict(self.U, self.V, self.B_u, self.B_i, self.B_s, 
                             test_u, test_i, self.scores, self.g_mean)
            
            rmse = np.sqrt(np.mean((test_r - np.clip(preds, 1.0, 5.0)) ** 2))
            history.append(rmse)
            
            # Early stopping lógico
            if rmse < mejor_rmse - 1e-4:
                mejor_rmse = rmse
                # Aquí podrías guardar el estado en self.best_state
            
        return history