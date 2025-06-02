# training_library.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# GPT Model minimal
class GPTModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.vocab_size = config["vocab_size"]
        self.context_length = config["context_length"]
        emb_dim = config["emb_dim"]

        # Embedding pour les tokens et les positions
        self.token_embedding = nn.Embedding(self.vocab_size, emb_dim)
        self.position_embedding = nn.Embedding(self.context_length, emb_dim)

        # Bloc Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=config["n_heads"],
            dropout=config["drop_rate"],
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config["n_layers"])

        # Normalisation + couche de sortie
        self.norm = nn.LayerNorm(emb_dim)
        self.head = nn.Linear(emb_dim, self.vocab_size)

    def forward(self, x):
        B, T = x.shape
        positions = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        x = self.token_embedding(x) + self.position_embedding(positions)
        x = self.transformer(x)
        x = self.norm(x)
        return self.head(x)

# Outils utiles pour l'entraînement (placeholders pour l'instant)
def create_dataloader_v1(token_ids, batch_size, max_length, shuffle=True, drop_last=True, num_workers=0):
    class TokenDataset(torch.utils.data.Dataset):
        def __init__(self, ids):
            self.ids = ids
            self.max_length = max_length

        def __len__(self):
            return len(self.ids) - self.max_length

        def __getitem__(self, idx):
            chunk = self.ids[idx:idx + self.max_length + 1]
            x = torch.tensor(chunk[:-1], dtype=torch.long)
            y = torch.tensor(chunk[1:], dtype=torch.long)
            return x, y

    dataset = TokenDataset(token_ids)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)

# Génération de texte simple (facultatif pour le moment)
def generate_text_simple(model, idx, max_new_tokens, context_size):
    model.eval()
    device = next(model.parameters()).device
    idx = idx[:context_size]
    input_tensor = torch.tensor(idx, dtype=torch.long, device=device).unsqueeze(0)

    for _ in range(max_new_tokens):
        input_cond = input_tensor[:, -context_size:]
        logits = model(input_cond)
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        input_tensor = torch.cat((input_tensor, next_token), dim=1)

    return input_tensor.squeeze().tolist()

# À utiliser avec SentencePiece uniquement
def text_to_token_ids(text, sp):
    return sp.encode(text, out_type=int)

def token_ids_to_text(token_ids, sp):
    return sp.decode(token_ids)
