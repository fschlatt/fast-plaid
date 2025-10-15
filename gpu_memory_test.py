import gc
import os
import shutil
from argparse import ArgumentParser

import torch
from fast_plaid import search


def format_memory(bytes_mem):
    """Format memory in MB or GB"""
    mb = bytes_mem / (1024 * 1024)
    if mb >= 1024:
        return f"{mb / 1024:.2f} GB"
    return f"{mb:.2f} MB"


def get_gpu_memory_info():
    """Get current GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated()
        reserved = torch.cuda.memory_reserved()
        max_allocated = torch.cuda.max_memory_allocated()
        return {
            "allocated": allocated,
            "reserved": reserved,
            "max_allocated": max_allocated,
            "free": reserved - allocated,
        }
    return None


def print_memory_stats(stage_name):
    """Print GPU memory statistics for a given stage"""
    memory_info = get_gpu_memory_info()
    if memory_info:
        print(f"\n{'=' * 50}")
        print(f"GPU Memory Usage - {stage_name}")
        print(f"{'=' * 50}")
        print(f"Allocated: {format_memory(memory_info['allocated'])}")
        print(f"Reserved:  {format_memory(memory_info['reserved'])}")
        print(f"Peak:      {format_memory(memory_info['max_allocated'])}")
        print(f"Free:      {format_memory(memory_info['free'])}")
        print(f"{'=' * 50}")
    else:
        print(f"{stage_name}: CUDA not available")


def clear_gpu_cache():
    """Clear GPU cache and collect garbage"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def main():
    """Main function to run the GPU memory test"""

    parser = ArgumentParser()

    parser.add_argument("--num_documents", type=int, default=500, help="Number of documents to create")

    args = parser.parse_args()

    # Clean up any existing test index
    if os.path.exists("test_index"):
        shutil.rmtree("test_index")

    # Reset memory stats
    torch.cuda.reset_peak_memory_stats()

    num_documents = args.num_documents
    embedding_dim = 128
    tokens_per_doc = 300

    documents_embeddings = [
        torch.randn(tokens_per_doc, embedding_dim, device="cpu", dtype=torch.float16) for _ in range(num_documents)
    ]
    clear_gpu_cache()

    index = search.FastPlaidIndex.create("test_index", documents_embeddings, embedding_dim=embedding_dim, device="cuda")

    print_memory_stats("After Index Creation")

    # Clean up test index
    if os.path.exists("test_index"):
        shutil.rmtree("test_index")


if __name__ == "__main__":
    main()
