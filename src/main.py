import os
import sys
import torch
import json
import matplotlib.pyplot as plt
import pandas as pd
from scipy.sparse import load_npz
from torch.utils.data import DataLoader, TensorDataset

sys.path.append(os.path.join(os.path.dirname(__file__), 'models'))
from GMF import GMF
from MLP import MLP
from knn_model_optimized import KNN_model
from evaluador import preparar_datos_evaluacion
from entrenamiento import entrenar_modelo_universal
from metrics import evaluate_model

CONFIG = {
    "GMF": {"lr": 0.01, "latent_dim": 8,  "epochs": 15},
    "MLP": {"lr": 0.005, "latent_dim": 16, "hidden_layers": [64, 32], "epochs": 15}
}

def plot_learning_curves(historiales):
    plt.figure(figsize=(10, 6))
    for nombre, hist in historiales.items():
        plt.plot(hist['train_loss'], label=f"{nombre} (Train)")
    plt.title("Evolución de la pérdida (Loss)")
    plt.xlabel("Época")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('curvas_entrenamiento.png', dpi=150, bbox_inches='tight')
    plt.show()

def main():
    ruta_datos       = '../data/processed/'
    ruta_checkpoints = '../models/checkpoints/'
    os.makedirs(ruta_checkpoints, exist_ok=True)

    historiales_sesion = {}

    input("Asegúrate de haber ejecutado el pipeline de EDA. Pulsa ENTER para continuar.")

    while True:
        print("\n--- MENÚ PRINCIPAL ---")
        print("1. Generalized Matrix Factorization (GMF)")
        print("2. Multi-Layer Perceptron (MLP)")
        print("3. K-Nearest Neighbours (KNN)")
        print("4. Salir")

        opcion = input("Selecciona una opción: ").strip()

        if opcion == '4':
            if historiales_sesion:
                if input("¿Visualizar curvas de aprendizaje? (s/n): ").lower() == 's':
                    plot_learning_curves(historiales_sesion)
            sys.exit(0)

        if opcion in ['1', '2']:
            df_train = pd.read_csv(f"{ruta_datos}train.csv")
            df_test  = pd.read_csv(f"{ruta_datos}test.csv")

            num_users = int(df_train['user_n'].max() + 1)
            num_items = int(df_train['track_n'].max() + 1)
            device    = 'cuda' if torch.cuda.is_available() else 'cpu'

            nombre_modelo = "GMF" if opcion == '1' else "MLP"
            params        = CONFIG[nombre_modelo]
            path_pesos    = f"{ruta_checkpoints}{nombre_modelo.lower()}_best.pt"
            path_historial= f"{ruta_checkpoints}{nombre_modelo.lower()}_hist.json"

            if nombre_modelo == "GMF":
                modelo = GMF(num_users, num_items, latent_dim=params["latent_dim"])
            else:
                modelo = MLP(num_users, num_items, latent_dim=params["latent_dim"],
                             hidden_layers=params["hidden_layers"])

            # --- Checkpoint ---
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

            # --- Evaluación ---
            df_eval = preparar_datos_evaluacion(df_train, df_test, num_items_totales=num_items)
            modelo.eval()
            with torch.no_grad():
                preds = modelo(
                    torch.tensor(df_eval['user_n'].values, dtype=torch.long).to(device),
                    torch.tensor(df_eval['track_n'].values, dtype=torch.long).to(device)
                ).cpu().numpy()
            print(evaluate_model(df_eval, preds))

        elif opcion == '3':
            train_set = load_npz(f"{ruta_datos}train_set.npz")
            test_set  = load_npz(f"{ruta_datos}test_set.npz")
            knn_model = KNN_model(train_set, test_set, k=20)
            print(knn_model.evaluate(max_users=500))

if __name__ == '__main__':
    main()
