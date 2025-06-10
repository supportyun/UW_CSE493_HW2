# train_problem2_epoch.py - Epoch-based Mini-batch Training for Modular Arithmetic GPT

import os
import torch
import torch.nn.functional as F
import numpy as np
import random
import re
from torch.utils.data import DataLoader, TensorDataset
from model import GPT, GPTConfig

# ========== Config ==========
BLOCK_SIZE = 4
NUM_HEADS = 4
N_EMBD = 128
FFN_DIM = 512
DROPOUT = 0.0
WEIGHT_DECAY = 1.0
LEARNING_RATE = 1e-3
BETAS = (0.9, 0.98)
MAX_TRAIN_STEPS = 10**5
BATCH_SIZE = 64
LOG_INTERVAL = 100 # log every 1000 steps
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ========== Utilities ==========
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def load_data(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    return [line.strip() for line in lines]

def tokenize_expr(expr):
    return re.findall(r'\d+|[+\-/=]', expr)

def build_vocab(data):
    tokens = set()
    for expr in data:
        input_str, target_str = expr.split('=')
        expr_tokens = tokenize_expr(input_str.strip() + '=' + target_str.strip())
        tokens.update(expr_tokens)
    tokens = sorted(tokens)
    stoi = {tok: i + 1 for i, tok in enumerate(tokens)}
    stoi['<pad>'] = 0
    stoi['<eos>'] = len(stoi)
    itos = {i: tok for tok, i in stoi.items()}
    return stoi, itos

def encode_single_target(examples, stoi):
    encoded_pairs = []
    for ex in examples:
        input_str, target_str = ex.split('=')
        input_tokens = tokenize_expr(input_str.strip() + '=')
        target_token = target_str.strip()

        x = torch.tensor([stoi[tok] for tok in input_tokens], dtype=torch.long)
        y = torch.tensor([stoi[target_token]], dtype=torch.long)
        encoded_pairs.append((x, y))
    return encoded_pairs

def pad_batch_single(pairs, block_size):
    x_batch, y_batch = zip(*pairs)
    x_pad = torch.stack([F.pad(x, (0, block_size - len(x)), value=0) for x in x_batch])
    y_tensor = torch.cat(y_batch)
    return x_pad, y_tensor

def compute_accuracy_single_token(logits, targets):
    pred_ids = logits.argmax(dim=-1)[:, -1]
    correct = (pred_ids == targets.view(-1)).sum().item()
    return correct / targets.size(0)

# ========== Training ==========
def train(seed, n_layer, data_dir, op, p):
    print(f"\n[Seed {seed}] Training {n_layer}-layer GPT for {op}_mod_{p}")
    set_seed(seed)

    train_data = load_data(os.path.join(data_dir, 'train.txt'))
    test_data = load_data(os.path.join(data_dir, 'test.txt'))

    stoi, itos = build_vocab(train_data)
    train_encoded = encode_single_target(train_data, stoi)
    test_encoded = encode_single_target(test_data, stoi)

    x_train, y_train = pad_batch_single(train_encoded, BLOCK_SIZE)
    x_test, y_test = pad_batch_single(test_encoded, BLOCK_SIZE)
    x_test, y_test = x_test.to(DEVICE), y_test.to(DEVICE)

    train_dataset = TensorDataset(x_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    config = GPTConfig(
        vocab_size=len(stoi),
        block_size=BLOCK_SIZE,
        n_layer=n_layer,
        n_head=NUM_HEADS,
        n_embd=N_EMBD,
        dropout=DROPOUT,
        bias=True
    )
    model = GPT(config).to(DEVICE)
    optimizer = model.configure_optimizers(WEIGHT_DECAY, LEARNING_RATE, BETAS, DEVICE)

    losses = []
    step = 0
    while step < MAX_TRAIN_STEPS:
        for x_batch, y_batch in train_loader:
            model.train()
            if step >= MAX_TRAIN_STEPS:
                break
            x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
            logits = model(x_batch)
            loss = F.cross_entropy(logits[:, -1, :], y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step += 1
            if step % LOG_INTERVAL == 0:
                model.eval()
                with torch.no_grad():
                    test_logits = model(x_test)
                    test_loss = F.cross_entropy(test_logits[:, -1, :], y_test).item()
                    test_acc = compute_accuracy_single_token(test_logits, y_test)
                    train_acc = compute_accuracy_single_token(logits, y_batch)

                print(f"Step {step}: Train Loss = {loss.item():.4f}, Train Acc = {train_acc:.2%}, Test Loss = {test_loss:.4f}, Test Acc = {test_acc:.2%}")
                losses.append((step, loss.item(), test_loss, train_acc, test_acc))

    tag = f"{op}_mod_{p}_seed{seed}_layer{n_layer}"
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), f"checkpoints/gpt_{tag}.pt")
    np.savetxt(f"checkpoints/log_{tag}.csv", np.array(losses), delimiter=",", header="step,train_loss,test_loss,train_acc,test_acc")
    print(f"Saved model and logs for {tag}")

# ========== Entry Point ==========
if __name__ == "__main__":
    for op in ["div"]:
        for p in [97]:
            DATASET_PATH = f"data/{op}_mod_{p}"
            for seed in [1,2,3]:
                for n_layer in [2]:
                    train(seed, n_layer, DATASET_PATH, op, p)
