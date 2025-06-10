# inference_sanity.py - Inference for "I love machine learning" sanity check

import torch
from model import GPT, GPTConfig
import os

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def tokenize(text):
    return text.lower().strip().split()  

def build_vocab_from_file(file_path):
    with open(file_path, 'r') as f:
        lines = [line.strip() for line in f]
    tokens = sorted(set(tok for line in lines for tok in tokenize(line)))
    stoi = {tok: i for i, tok in enumerate(tokens)}
    itos = {i: tok for tok, i in stoi.items()}
    return stoi, itos

def encode_input(prompt, stoi, block_size):
    tokens = tokenize(prompt)
    try:
        idxs = [stoi[tok] for tok in tokens]
    except KeyError as e:
        raise ValueError(f"Unknown token: {e.args[0]}. Make sure your input matches the training data.")
    x = torch.tensor([idxs], dtype=torch.long).to(DEVICE)
    if x.size(1) > block_size:
        raise ValueError(f"Prompt too long ({x.size(1)} > {block_size})")
    return x

def decode_token(token_id, itos):
    return itos.get(token_id, "<unk>")

def predict_next_token(model, x):
    with torch.no_grad():
        logits = model(x)  # shape: [1, T, vocab_size]
        pred_token_id = logits[:, -1, :].argmax(dim=-1).item()
    return pred_token_id

if __name__ == "__main__":
    # === Settings: must match training ===
    checkpoint_path = "checkpoints/gpt_sanity_check.pt"
    data_path = "data/sanity.txt"
    n_layer = 1
    n_head = 2
    n_embd = 64
    block_size = 4

    # === Load vocab ===
    stoi, itos = build_vocab_from_file(data_path)

    # === Load model ===
    model = GPT(GPTConfig(
        vocab_size=len(stoi),
        block_size=block_size,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        dropout=0.0,
        bias=True
    )).to(DEVICE)
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    model.eval()

    print("number of parameters:", sum(p.numel() for p in model.parameters()) / 1e6, "M")

    # === Inference loop ===
    while True:
        prompt = input("Enter prompt (e.g., 'I love machine') or type 'exit': ").strip()
        if prompt.lower() == "exit":
            break
        try:
            x = encode_input(prompt, stoi, block_size)
            pred_id = predict_next_token(model, x)
            print(f"Prediction: {prompt} {decode_token(pred_id, itos)}")
        except Exception as e:
            print(f"Error: {e}")
