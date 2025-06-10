# inference.py - Single-token inference for Modular Arithmetic Tasks

import torch
from model import GPT, GPTConfig
from train import tokenize_expr, build_vocab
import os

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_model(checkpoint_path, vocab_size, block_size, n_layer, n_head, n_embd, dropout=0.0):
    config = GPTConfig(
        vocab_size=vocab_size,
        block_size=block_size,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        dropout=dropout,
        bias=True
    )
    model = GPT(config).to(DEVICE)
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    model.eval()
    return model

def encode_prompt(prompt, stoi, block_size):
    tokens = tokenize_expr(prompt)
    idxs = [stoi[tok] for tok in tokens if tok in stoi]
    x = torch.tensor([idxs], dtype=torch.long).to(DEVICE)
    if x.size(1) > block_size:
        raise ValueError(f"Prompt too long ({x.size(1)} > {block_size})")
    return x

def decode_token(token_id, itos):
    return itos[token_id] if token_id in itos else "<unk>"

def predict_single_token(model, x):
    with torch.no_grad():
        logits = model(x)  # [1, T, vocab]
        pred_token_id = logits[:, -1, :].argmax(dim=-1).item()
    return pred_token_id

if __name__ == "__main__":
    # Settings (should match your training setup)
    op = "div"
    p = 97
    seed = 1
    n_layer = 2
    block_size = 4  # must match your training
    n_head = 4
    n_embd = 128

    # Load vocab
    with open(f"data/{op}_mod_{p}/train.txt", "r") as f:
        lines = [line.strip() for line in f]
    stoi, itos = build_vocab(lines)

    # Load model
    tag = f"{op}_mod_{p}_seed{seed}_layer{n_layer}"
    model = load_model(f"checkpoints/gpt_{tag}.pt", len(stoi), block_size, n_layer, n_head, n_embd)

    while True:
        expr = input("Enter expression like '12/25=' (or type 'exit'): ")
        if expr.strip().lower() == 'exit':
            break
        try:
            x = encode_prompt(expr, stoi, block_size)
            pred_id = predict_single_token(model, x)
            answer = decode_token(pred_id, itos)
            print(f"Prediction: {expr}{answer}")
        except Exception as e:
            print(f"Error: {e}")
