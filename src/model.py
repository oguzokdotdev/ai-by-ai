import torch
import torch.nn as nn
from torch.nn import functional as F
from src.config import *

class TransformerModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.vocab_size = vocab_size
        # Здесь будут слои: Embedding, Blocks (Attention + FFNN), Linear
        
    def forward(self, idx, targets=None):
        # Логика прохода данных через слои
        pass

    def generate(self, idx, max_new_tokens):
        # Логика генерации нового текста символ за символом
        pass

