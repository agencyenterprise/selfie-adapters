#!/usr/bin/env python3
"""
Start vLLM as a persistent OpenAI-compatible server.

This server will stay running and handle incoming requests with automatic batching.
You only need to start it once, and it will serve all your inference requests.

Usage:
    python start_vllm_server.py
    python start_vllm_server.py --tensor-parallel-size 1  # Single GPU
    python start_vllm_server.py --port 8080  # Custom port
"""
import argparse
import subprocess
import sys


def start_server(
    model: str = "meta-llama/Llama-3.1-8B-Instruct",
    host: str = "0.0.0.0",
    port: int = 8000,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.90,
):
    """
    Start the vLLM OpenAI-compatible server.
    
    Args:
        model: HuggingFace model identifier
        host: Host to bind to (0.0.0.0 for all interfaces)
        port: Port to bind to
        tensor_parallel_size: Number of GPUs to use (1 for single GPU, 2+ for multi-GPU)
        gpu_memory_utilization: Fraction of GPU memory to use
    """
    print("=" * 80)
    print("Starting vLLM Server")
    print("=" * 80)
    print(f"Model: {model}")
    print(f"Host: {host}:{port}")
    print(f"Tensor Parallel Size: {tensor_parallel_size}")
    print(f"GPU Memory Utilization: {gpu_memory_utilization}")
    print("=" * 80)
    print("\nThis will take a few minutes to load the model...")
    print("Once started, the server will stay running until you stop it (Ctrl+C)")
    print("=" * 80 + "\n")
    
    cmd = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", model,
        "--host", host,
        "--port", str(port),
        "--tensor-parallel-size", str(tensor_parallel_size),
        "--gpu-memory-utilization", str(gpu_memory_utilization),
        "--trust-remote-code",
    ]
    
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n\nServer stopped by user.")
        sys.exit(0)
    except FileNotFoundError:
        print("\n\nError: vLLM not found. Install it with: pip install vllm")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nError starting server: {e}")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Start vLLM server for TwoHopFact filtering"
    )
    parser.add_argument(
        "--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct",
        help="HuggingFace model identifier"
    )
    parser.add_argument(
        "--host", type=str, default="0.0.0.0",
        help="Host to bind to"
    )
    parser.add_argument(
        "--port", type=int, default=8000,
        help="Port to bind to"
    )
    parser.add_argument(
        "--tensor-parallel-size", type=int, default=1,
        help="Number of GPUs for tensor parallelism (default: 1)"
    )
    parser.add_argument(
        "--gpu-memory-utilization", type=float, default=0.90,
        help="Fraction of GPU memory to use (default: 0.90)"
    )
    
    args = parser.parse_args()
    
    start_server(
        model=args.model,
        host=args.host,
        port=args.port,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
