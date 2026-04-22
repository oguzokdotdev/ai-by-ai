import torch

# Автоматический выбор устройства (GPU, если есть, иначе CPU)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Гиперпараметры модели
BLOCK_SIZE = 128      # Длина контекста
BATCH_SIZE = 32     # Размер батча
N_EMBED = 384        # Размерность эмбеддингов
N_HEAD = 6           # Количество голов Attention
N_LAYER = 6          # Количество слоев Трансформера
DROPOUT = 0.2        # Дропаут для защиты от переобучения

# Гиперпараметры обучения
LEARNING_RATE = 3e-4
MAX_ITERS = 10000
EVAL_INTERVAL = 500

