# train_sanity.py - Training GPT on "I love machine learning" (Sanity Check Task)

import os
import torch
import torch.nn.functional as F
import numpy as np
import random
from torch.utils.data import DataLoader, TensorDataset
from model import GPT, GPTConfig

# ========== Config ==========
BLOCK_SIZE = 4
NUM_HEADS = 2
N_EMBD = 64
FFN_DIM = 256
DROPOUT = 0.0
WEIGHT_DECAY = 1.0
LEARNING_RATE = 1e-3
BETAS = (0.9, 0.98)
MAX_TRAIN_STEPS = 500
BATCH_SIZE = 1
LOG_INTERVAL = 10
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ========== Utilities ==========
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def load_data(file_path):
    with open(file_path, 'r') as f:
        return [line.strip() for line in f]

def tokenize(text):
    return text.strip().split()

def build_vocab(data):
    tokens = set()
    for line in data:
        tokens.update(tokenize(line))
    tokens = sorted(tokens)
    stoi = {tok: i for i, tok in enumerate(tokens)}
    itos = {i: tok for tok, i in stoi.items()}
    return stoi, itos

def encode_example(line, stoi):
    tokens = tokenize(line)
    input_tokens = tokens[:3]  # "I love machine"
    target_token = tokens[3]  # "learning"
    x = torch.tensor([stoi[tok] for tok in input_tokens], dtype=torch.long)
    y = torch.tensor([stoi[target_token]], dtype=torch.long)
    return x, y

def pad_batch(pairs, block_size):
    x_batch, y_batch = zip(*pairs)
    x_pad = torch.stack([F.pad(x, (0, block_size - len(x)), value=0) for x in x_batch])
    y_tensor = torch.cat(y_batch)
    return x_pad, y_tensor

def compute_accuracy(logits, targets):
    pred_ids = logits.argmax(dim=-1)[:, -1]
    return (pred_ids == targets.view(-1)).sum().item() / targets.size(0)

# ========== Training ==========
def train(seed=1):
    set_seed(seed)
    data = load_data("data/sanity.txt")  # should contain "I love machine learning"
    stoi, itos = build_vocab(data)

    train_encoded = [encode_example(line, stoi) for line in data]
    x_train, y_train = pad_batch(train_encoded, BLOCK_SIZE)
    dataset = TensorDataset(x_train, y_train)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    config = GPTConfig(
        vocab_size=len(stoi), block_size=BLOCK_SIZE,
        n_layer=1, n_head=NUM_HEADS, n_embd=N_EMBD,
        dropout=DROPOUT, bias=True
    )
    model = GPT(config).to(DEVICE)
    optimizer = model.configure_optimizers(WEIGHT_DECAY, LEARNING_RATE, BETAS, DEVICE)

    losses = []
    step = 0
    while step < MAX_TRAIN_STEPS:
        for xb, yb in loader:
            if step >= MAX_TRAIN_STEPS: break
            model.train()
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            logits = model(xb)
            loss = F.cross_entropy(logits[:, -1, :], yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step += 1
            if step ==1 or step % LOG_INTERVAL == 0:
                acc = compute_accuracy(logits, yb)
                print(f"Step {step}: Loss = {loss.item():.4f}, Accuracy = {acc:.2%}")
                losses.append((step, loss.item(), acc))

    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/gpt_sanity_check.pt")
    np.savetxt("checkpoints/log_sanity.csv", np.array(losses), delimiter=",", header="step,loss,accuracy")
    print("Sanity model and log saved.")

# ========== Entry ==========
if __name__ == "__main__":
    train()
