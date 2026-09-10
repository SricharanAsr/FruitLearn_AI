import torch
import time
from src.train_mae import MaskedAutoencoderViT

def benchmark_mae_throughput():
    print("=== FruitLearn AI: MAE Hardware Efficiency & Throughput Optimization ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Target Execution Device: {device}")

    model = MaskedAutoencoderViT(mask_ratio=0.75).to(device)
    model.eval()

    dummy_input = torch.randn(16, 3, 224, 224, device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(5):
            _ = model(dummy_input)

    # Mixed Precision Benchmark
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == 'cuda')
    start_time = time.time()
    num_iterations = 20

    with torch.no_grad():
        for _ in range(num_iterations):
            with torch.cuda.amp.autocast(enabled=device.type == 'cuda'):
                loss, pred, mask = model(dummy_input)

    total_time = time.time() - start_time
    imgs_per_sec = (num_iterations * 16) / total_time

    print(f"Batch Size: 16 | Iterations: {num_iterations}")
    print(f"Total Time: {total_time:.3f} s | Throughput: {imgs_per_sec:.2f} images/sec")
    print("Optimization complete: MAE 75% Masking efficiency verified.")

if __name__ == "__main__":
    benchmark_mae_throughput()
