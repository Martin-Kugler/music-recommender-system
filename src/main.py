import os
import sys
import torch
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.sparse import load_npz
from torch.utils.data import DataLoader, TensorDataset

# Configuración del path para importaciones dentro de la carpeta 'models'
sys.path.append(os.path.join(os.path.dirname(__file__)))
from GMF import GMF
from MLP import MLP
from PMF import PMF
from knn_model_optimized import KNN_model
from evaluador import preparar_datos_evaluacion
from entrenamiento import entrenar_modelo_universal
from metrics import evaluate_model
from entrenar_bmf import BeMFModel, _predict

CONFIG = {
    "GMF": {"lr": 0.01, "latent_dim": 8, "epochs": 15},
    "MLP": {"lr": 0.005, "latent_dim": 16, "hidden_layers": [64, 32], "epochs": 15},
    "PMF": {"lr": 0.01, "latent_dim": 8, "epochs": 15},
    "BeMF": {"lr": 0.01, "latent_dim": 10, "epochs": 10, "reg": 0.01}
}

def plot_learning_curves(historiales):
    plt.figure(figsize=(10, 6))
    for nombre, hist in historiales.items():
        plt.plot(hist['train_loss'], label=f"{nombre} (Train)")
    plt.title("Evolución de la pérdida (Loss)")
    plt.xlabel("Época")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('curvas_entrenamiento.png', dpi=150, bbox_inches='tight')
    plt.show()

def main():
    ruta_datos = '../data/processed/'
    ruta_checkpoints = '../models/checkpoints/'
    os.makedirs(ruta_checkpoints, exist_ok=True)

    historiales_sesion = {}

    input("Asegúrate de haber ejecutado el pipeline de EDA. Pulsa ENTER para continuar.")

    while True:
        print("\n--- MENÚ PRINCIPAL ---")
        print("1. Generalized Matrix Factorization (GMF)")
        print("2. Multi-Layer Perceptron (MLP)")
        print("3. Probabilistic Matrix Factorization (PMF)")
        print("4. Bernoulli Matrix Factorization (BeMF - Compañero)")
        print("5. K-Nearest Neighbours (KNN)")
        print("6. Salir")

        opcion = input("Selecciona una opción: ").strip()

        if opcion == '6':
            if historiales_sesion:
                if input("¿Visualizar curvas de aprendizaje? (s/n): ").lower() == 's':
                    plot_learning_curves(historiales_sesion)
            sys.exit(0)

        # Carga y dimensiones comunes de datos tabulares
        if opcion in ['1', '2', '3']:
            df_train = pd.read_csv(f"{ruta_datos}train.csv")
            df_test = pd.read_csv(f"{ruta_datos}test.csv")

            num_users = int(max(df_train['user_n'].max(), df_test['user_n'].max()) + 1)
            num_items = int(max(df_train['track_n'].max(), df_test['track_n'].max()) + 1)
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # --- OPCIONES 1, 2 y 3: MODELOS PYTORCH (GMF / MLP / PMF) ---
        if opcion in ['1', '2', '3']:
            if opcion == '1':
                nombre_modelo = "GMF"
            elif opcion == '2':
                nombre_modelo = "MLP"
            else:
                nombre_modelo = "PMF"
                
            params = CONFIG[nombre_modelo]
            path_pesos = f"{ruta_checkpoints}{nombre_modelo.lower()}_best.pt"
            path_historial = f"{ruta_checkpoints}{nombre_modelo.lower()}_hist.json"

            if nombre_modelo == "GMF":
                modelo = GMF(num_users, num_items, latent_dim=params["latent_dim"])
            elif nombre_modelo == "MLP":
                modelo = MLP(num_users, num_items, latent_dim=params["latent_dim"],
                             hidden_layers=params["hidden_layers"])
            else:
                modelo = PMF(num_users, num_items, num_factors=params["latent_dim"])

            # --- Checkpoint PyTorch ---
            if os.path.exists(path_pesos) and os.path.exists(path_historial):
                print(f"Checkpoint encontrado para {nombre_modelo}. Cargando...")
                modelo.load_state_dict(torch.load(path_pesos, map_location=device))
                with open(path_historial) as f:
                    historial = json.load(f)
            else:
                print(f"No hay checkpoint para {nombre_modelo}. Iniciando entrenamiento...")
                dl_train = DataLoader(
                    TensorDataset(
                        torch.LongTensor(df_train['user_n'].values),
                        torch.LongTensor(df_train['track_n'].values),
                        torch.FloatTensor(df_train['rating'].values)
                    ), batch_size=1024, shuffle=True)

                modelo, h_train = entrenar_modelo_universal(
                    modelo, dl_train,
                    lr=params["lr"], num_epocas=params["epochs"], device=device)

                torch.save(modelo.state_dict(), path_pesos)
                historial = {'train_loss': h_train}
                with open(path_historial, 'w') as f:
                    json.dump(historial, f)

            historiales_sesion[nombre_modelo] = historial

            # --- Evaluación PyTorch ---
            df_eval = preparar_datos_evaluacion(df_train, df_test, num_items_totales=num_items)
            modelo.eval()
            with torch.no_grad():
                preds = modelo(
                    torch.tensor(df_eval['user_n'].values, dtype=torch.long).to(device),
                    torch.tensor(df_eval['track_n'].values, dtype=torch.long).to(device)
                ).cpu().numpy()
            print(evaluate_model(df_eval, preds))

        # --- OPCIÓN 4: MODELO BMF (NUMBA - COMPAÑERO) ---
        elif opcion == '4':
            df_train = pd.read_csv(f"{ruta_datos}train.csv")
            df_test = pd.read_csv(f"{ruta_datos}test.csv")
            num_users = int(max(df_train['user_n'].max(), df_test['user_n'].max()) + 1)
            num_items = int(max(df_train['track_n'].max(), df_test['track_n'].max()) + 1)

            nombre_modelo = "BeMF"
            params = CONFIG[nombre_modelo]
            path_json = f"{ruta_checkpoints}bmf_hist.json"
            path_npz_arrays = f"{ruta_checkpoints}bmf_arrays.npz"

            g_mean = float(df_train['rating'].mean())
            bmf_model = BeMFModel(num_factors=params["latent_dim"], scores=np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
                                  B_s_init=np.zeros(5), num_users=num_users, num_items=num_items, g_mean=g_mean)

            # --- Checkpoint Numba ---
            if os.path.exists(path_json) and os.path.exists(path_npz_arrays):
                print(f"Checkpoint encontrado para {nombre_modelo}. Cargando...")
                with open(path_json, 'r') as f:
                    historial = json.load(f)
                
                arrays_guardados = np.load(path_npz_arrays)
                bmf_model.U = arrays_guardados['U']
                bmf_model.V = arrays_guardados['V']
                bmf_model.B_u = arrays_guardados['B_u']
                bmf_model.B_i = arrays_guardados['B_i']
                bmf_model.B_s = arrays_guardados['B_s']
            else:
                print(f"No hay checkpoint para {nombre_modelo}. Iniciando entrenamiento...")
                train_data = (df_train['user_n'].values, df_train['track_n'].values, df_train['rating'].values)
                test_data = (df_test['user_n'].values, df_test['track_n'].values, df_test['rating'].values)

                h_train = bmf_model.fit(train_data, test_data, num_epochs=params["epochs"],
                                        lr_inicial=params["lr"], reg=params["reg"])

                historial = {'train_loss': h_train}
                with open(path_json, 'w') as f:
                    json.dump(historial, f)
                np.savez(path_npz_arrays, U=bmf_model.U, V=bmf_model.V, B_u=bmf_model.B_u, B_i=bmf_model.B_i, B_s=bmf_model.B_s)

            historiales_sesion[nombre_modelo] = historial

            # --- Evaluación BMF con Negative Sampling ---
            df_eval = preparar_datos_evaluacion(df_train, df_test, num_items_totales=num_items)
            print("Calculando predicciones de ranking con la función _predict JIT...")
            preds = _predict(bmf_model.U, bmf_model.V, bmf_model.B_u, bmf_model.B_i, bmf_model.B_s,
                             df_eval['user_n'].values, df_eval['track_n'].values, bmf_model.scores, bmf_model.g_mean)
            print(evaluate_model(df_eval, preds))

        # --- OPCIÓN 5: BASELINE KNN ---
        elif opcion == '5':
            train_set = load_npz(f"{ruta_datos}train_set.npz")
            test_set = load_npz(f"{ruta_datos}test_set.npz")
            knn_model = KNN_model(train_set, test_set, k=20)
            print(knn_model.evaluate(max_users=500))

if __name__ == '__main__':
    main()