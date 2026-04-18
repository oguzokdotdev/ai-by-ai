import torch
from src.config import *
from src.model import TransformerModel
from src.dataset import get_batch, vocab_size

@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(200)
        for k in range(200):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def main():
    print(f"Запуск на устройстве: {DEVICE}")
    model = TransformerModel(vocab_size).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    for iter in range(MAX_ITERS):
        if iter % EVAL_INTERVAL == 0:
            losses = estimate_loss(model)
            print(f"Step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            # Сохраняем веса
            torch.save(model.state_dict(), 'weights/model.pt')

        xb, yb = get_batch('train')
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    print("Обучение завершено!")

if __name__ == "__main__":
    main()
