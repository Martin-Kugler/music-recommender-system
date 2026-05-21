import torch
import torch.nn as nn

class PMF(nn.Module):
    """Probabilistic Matrix Factorization en PyTorch (GPU)."""
    def __init__(self, num_users, num_items, num_factors):
        super().__init__()
        self.user_emb = nn.Embedding(num_users, num_factors)
        self.item_emb = nn.Embedding(num_items, num_factors)

        # Inicialización de pesos con distribución normal pequeña
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)

    def forward(self, user_ids, item_ids):
        u = self.user_emb(user_ids)
        i = self.item_emb(item_ids)

        # Producto escalar: multiplicar elemento a elemento y sumar
        preds = (u * i).sum(dim=1)
        return torch.clamp(preds, 1.0, 5.0)