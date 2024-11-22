import numpy as np
import torch
import time
import matplotlib.pyplot as plt


# Fast CPU-based matrix multiplication
def fast_matmul(a, b):
    """
    Perform fast matrix multiplication using NumPy.
    
    Args:
        a (np.ndarray): Matrix A
        b (np.ndarray): Matrix B
        
    Returns:
        np.ndarray: Resultant matrix
    """
    return np.dot(a, b)


# GPU-based matrix multiplication using CUDA
def gpu_matmul(a, b):
    """
    Perform matrix multiplication on the GPU using PyTorch.
    
    Args:
        a (np.ndarray): Matrix A
        b (np.ndarray): Matrix B
        
    Returns:
        np.ndarray: Resultant matrix (transferred back to CPU)
    """
    a_gpu = torch.tensor(a, device="cuda")
    b_gpu = torch.tensor(b, device="cuda")
    c_gpu = torch.matmul(a_gpu, b_gpu)
    return c_gpu.cpu().numpy()


# Benchmarking function
def benchmark(sizes):
    """
    Benchmark matrix multiplication methods for different matrix sizes.
    
    Args:
        sizes (list): List of matrix sizes to test.
        
    Returns:
        dict: Timing results for each method and size.
    """
    results = {}

    for size in sizes:
        print(f"Running size {size}")
        a = np.random.rand(size, size).astype(np.float32)
        b = np.random.rand(size, size).astype(np.float32)

        timings = {}

        # Timing fast_matmul
        start = time.time()
        fast_matmul(a, b)
        timings["fast"] = time.time() - start

        # Timing gpu_matmul
        start = time.time()
        gpu_matmul(a, b)
        timings["gpu"] = time.time() - start

        print(timings)
        results[size] = timings

    return results


# Plotting function
def plot_results(results):
    """
    Plot the timing results for matrix multiplication methods.
    
    Args:
        results (dict): Timing results from benchmark().
    """
    sizes = list(results.keys())
    fast_times = [results[size]["fast"] for size in sizes]
    gpu_times = [results[size]["gpu"] for size in sizes]

    plt.figure(figsize=(10, 6))
    plt.plot(sizes, fast_times, label="Fast (CPU)", marker="o")
    plt.plot(sizes, gpu_times, label="GPU", marker="o")
    plt.xlabel("Matrix Size")
    plt.ylabel("Time (seconds)")
    plt.title("Matrix Multiplication Timing Comparison")
    plt.legend()
    plt.grid(True)
    plt.show()


# Main execution
if __name__ == "__main__":
    sizes = [64, 128, 256, 512, 1024]
    results = benchmark(sizes)

    print("\nTiming summary")
    for size in sizes:
        print(f"Size: {size}")
        print(f"    fast: {results[size]['fast']:.5f}")
        print(f"    gpu: {results[size]['gpu']:.5f}")

    plot_results(results)
