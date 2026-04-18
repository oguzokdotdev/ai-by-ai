import torch
from src.config import BLOCK_SIZE, BATCH_SIZE, DEVICE
from src.tokenizer import CharTokenizer

# Читаем данные
with open('data/input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Инициализируем токенизатор и данные
tokenizer = CharTokenizer(text)
vocab_size = tokenizer.vocab_size
data = torch.tensor(tokenizer.encode(text), dtype=torch.long)

# Разделение на train/val (90% / 10%)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    data_source = train_data if split == 'train' else val_data
    ix = torch.randint(len(data_source) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([data_source[i:i+BLOCK_SIZE] for i in ix])
    y = torch.stack([data_source[i+1:i+BLOCK_SIZE+1] for i in ix])
    x, y = x.to(DEVICE), y.to(DEVICE)
    return x, y
