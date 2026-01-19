import torch
import time
import torch.multiprocessing as mp

MATRIX_SIZE = 8192*7

def worker(gpu_id):
    torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}")

    while True:
        A = torch.randn(MATRIX_SIZE, MATRIX_SIZE, device=device)
        B = torch.randn(MATRIX_SIZE, MATRIX_SIZE, device=device)

        start = time.time()
        C = torch.matmul(A, B)
        torch.cuda.synchronize()
        end = time.time()

def main():
    num_gpus = torch.cuda.device_count()
    print(f"Using {num_gpus} GPUs")

    mp.spawn(
        worker,
        args=(),
        nprocs=num_gpus,
        join=True
    )

if __name__ == "__main__":
    main()
