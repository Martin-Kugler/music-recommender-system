import torch
import torch.nn as nn

class PMF(nn.Module):
    def __init__(self, num_users, num_items, latent_dim):
        super().__init__()
        # Mismos embeddings que GMF
        self.user_emb = nn.Embedding(num_users, latent_dim)
        self.item_emb = nn.Embedding(num_items, latent_dim)
        
        # Inicialización estándar para estabilizar el producto escalar temprano
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)

    def forward(self, user_ids, item_ids):
        u = self.user_emb(user_ids)
        i = self.item_emb(item_ids)
        
        # Producto escalar puro (sin la capa nn.Linear de GMF)
        # Multiplicamos elemento a elemento y sumamos en la dimensión latente
        dot_product = (u * i).sum(dim=1)
        
        # Acotamos entre 1 y 5 para cuadrar con el rating esperado (igual que GMF)
        pred = torch.sigmoid(dot_product) * 4.0 + 1.0
        return pred