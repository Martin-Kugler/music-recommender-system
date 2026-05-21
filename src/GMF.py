import torch
import torch.nn as nn

class GMF(nn.Module):
    """
    Generalized Matrix Factorization (GMF) model.
    Proyecta los embeddings a una salida continua acotada en el rango [1, 5].
    """
    def __init__(self, num_users, num_items, latent_dim):
        super().__init__()
        self.user_emb = nn.Embedding(num_users, latent_dim)
        self.item_emb = nn.Embedding(num_items, latent_dim)
        self.output = nn.Linear(latent_dim, 1)

    def forward(self, user_ids, item_ids):
        u = self.user_emb(user_ids)
        i = self.item_emb(item_ids)
        
        # Producto Hadamard (elemento a elemento)
        prod = u * i
        out_lineal = self.output(prod)
        
        # Activación Sigmoide escalada al rango explícito de ratings [1.0, 5.0]
        pred = torch.sigmoid(out_lineal) * 4.0 + 1.0 
        return pred.squeeze()