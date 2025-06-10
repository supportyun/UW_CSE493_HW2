import os
import glob
import numpy as np
import matplotlib.pyplot as plt

def load_logs(pattern):
    paths = sorted(glob.glob(pattern))
    logs = [np.loadtxt(p, delimiter=",", skiprows=1) for p in paths]
    return logs, paths

def plot_accuracy(logs, paths, op, p, n_layer, save_dir="plots_log"):
    steps = logs[0][:, 0]
    train_accs = np.array([log[:, 3] for log in logs])  # train_acc column
    test_accs = np.array([log[:, 4] for log in logs])   # test_acc column

    train_mean = train_accs.mean(axis=0)
    train_std = train_accs.std(axis=0)
    test_mean = test_accs.mean(axis=0)
    test_std = test_accs.std(axis=0)

    plt.figure(figsize=(10, 6))


    # Mean lines
    plt.plot(steps, train_mean, label="Train Accuracy", color="blue", linewidth=2)
    plt.plot(steps, test_mean, label="Test Accuracy", color="orange", linewidth=2)

    # Shaded area for ±1 std
    plt.fill_between(steps, train_mean - train_std, train_mean + train_std, color="blue", alpha=0.1)
    plt.fill_between(steps, test_mean - test_std, test_mean + test_std, color="orange", alpha=0.1)

    plt.xlabel("Steps")
    plt.ylabel("Accuracy")
    plt.xscale('log')
    plt.title(f"{op.upper()} Mod {p} (Layer {n_layer})\nFinal Train Acc: {train_mean[-1]:.2%}, Test Acc: {test_mean[-1]:.2%}")
    plt.legend()
    plt.grid(True)

    os.makedirs(save_dir, exist_ok=True)
    filename = f"{save_dir}/acc_{op}_mod_{p}_layer{n_layer}_muon_lr0.01.png"
    plt.savefig(filename)
    plt.close()
    print(f"Saved plot to {filename}")

if __name__ == "__main__":
    for op in ["div"]:
        for p in [97]:
            for n_layer in [2]:
                pattern = f"checkpoints/log_{op}_mod_{p}_seed*_layer{n_layer}_muon_lr0.01.csv"
                logs, paths = load_logs(pattern)
                if logs:
                    plot_accuracy(logs, paths, op, p, n_layer)
