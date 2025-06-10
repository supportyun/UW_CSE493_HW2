import os
import random

# Compute modular inverse
def modinv(b, p):
    return pow(b, -1, p)

# Compute result of operation under modulo p
def compute(a, b, op, p):
    if op == "+":
        return (a + b) % p
    elif op == "-":
        return (a - b) % p
    elif op == "/":
        if b == 0:
            raise ValueError("Division by zero")
        return (a * modinv(b, p)) % p
    else:
        raise ValueError("Unknown operation")

# Generate all (a, b, c) triplets such that a op b ≡ c mod p
def generate_examples(p, op):
    examples = []
    for a in range(p):
        for b in range(p):
            try:
                c = compute(a, b, op, p)
                examples.append((a, b, c))
            except Exception:
                continue  # skip division by 0
    return examples

# Save dataset to files and write description
def save_split(examples, out_path, p, op, train_frac=0.25):
    os.makedirs(out_path, exist_ok=True)
    random.shuffle(examples)

    split = int(len(examples) * train_frac)
    train_data = examples[:split]
    test_data = examples[split:]

    def write_file(filename, data):
        with open(filename, "w") as f:
            for a, b, c in data:
                f.write(f"{a} {op} {b} = {c}\n")

    write_file(os.path.join(out_path, "train.txt"), train_data)
    write_file(os.path.join(out_path, "test.txt"), test_data)

    with open(os.path.join(out_path, "description.txt"), "w") as f:
        f.write(f"Operation: {op}\n")
        f.write(f"Modulus p = {p}\n")
        f.write(f"Total datapoints: {len(examples)}\n")
        f.write(f"Train split: {len(train_data)} examples\n")
        f.write(f"Test split: {len(test_data)} examples\n")
        f.write(f"Format: 'a {op} b = c' where c = a {op} b mod p\n")
        if op == "/":
            f.write("(Note: modular inverse is used for division)\n")

# Main entry point: generate all datasets
def generate_all_datasets(base_dir, ps=[97, 113], ops=["+", "-", "/"]):
    for p in ps:
        for op in ops:
            print(f"Generating dataset for {op} mod {p}...")
            dataset = generate_examples(p, op)
            folder_name = f"{op}_mod_{p}".replace("/", "div").replace("+", "add").replace("-", "sub")
            out_path = os.path.join(base_dir, folder_name)
            save_split(dataset, out_path, p, op)
            print(f"Saved to {out_path}")

# Run if executed directly
if __name__ == "__main__":
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(SCRIPT_DIR, "data")
    generate_all_datasets(base_dir=DATA_DIR)
