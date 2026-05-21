import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, num_users, num_items, latent_dim, hidden_layers):
        super().__init__()
        self.user_emb = nn.Embedding(num_users, latent_dim)
        self.item_emb = nn.Embedding(num_items, latent_dim)

        # Construcción dinámica de las capas densas
        capas = []
        input_size = latent_dim * 2          # ← x2 por la CONCATENACIÓN
        for h in hidden_layers:
            capas.append(nn.Linear(input_size, h))
            capas.append(nn.ReLU())          # ← no-linealidad (clave del MLP)
            input_size = h
        capas.append(nn.Linear(input_size, 1))   # capa de salida: a un escalar
        self.mlp = nn.Sequential(*capas)

    def forward(self, user_ids, item_ids):
        u = self.user_emb(user_ids)              # (batch, latent_dim)
        i = self.item_emb(item_ids)              # (batch, latent_dim)
        x = torch.cat([u, i], dim=1)             # ← CONCATENAR, no multiplicar. (batch, latent_dim*2)
        pred = self.mlp(x)                        # (batch, 1)
        return pred.squeeze()                     # (batch,)