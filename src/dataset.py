import torch
from src.config import BLOCK_SIZE, BATCH_SIZE, DEVICE

def get_batch(data):
    # Генерация случайных индексов для захвата блоков текста
    ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([data[i:i+BLOCK_SIZE] for i in ix])
    y = torch.stack([data[i+1:i+BLOCK_SIZE+1] for i in ix])
    # Перенос данных на видеокарту/процессор
    x, y = x.to(DEVICE), y.to(DEVICE)
    return x, y

