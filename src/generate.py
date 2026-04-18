import torch
from src.config import *
from src.model import TransformerModel
from src.tokenizer import CharTokenizer

def main():
    # 1. Подготовка
    with open('data/input.txt', 'r', encoding='utf-8') as f:
        text = f.read()
    tokenizer = CharTokenizer(text)
    
    # 2. Модель и веса
    model = TransformerModel(tokenizer.vocab_size)
    model.load_state_dict(torch.load('weights/model.pt', map_location=DEVICE, weights_only=True))
    model.eval()
    model.to(DEVICE)
    
    # 3. Генерация (начинаем с чистого листа)
    context = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)
    print("\n--- ГЕНЕРАЦИЯ ТЕКСТА ---")
    print(tokenizer.decode(model.generate(context, max_new_tokens=500)[0].tolist()))
    print("------------------------\n")

if __name__ == "__main__":
    main()
