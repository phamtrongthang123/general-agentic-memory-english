#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GAM Evaluation Suite - Unified evaluation entry

Usage examples:
    # HotpotQA
    python -m eval.run --dataset hotpotqa --data-path data/hotpotqa.json

    # NarrativeQA
    python -m eval.run --dataset narrativeqa --data-path narrativeqa --max-samples 100

    # LoCoMo
    python -m eval.run --dataset locomo --data-path data/locomo.json

    # RULER
    python -m eval.run --dataset ruler --data-path data/ruler.jsonl --dataset-name niah_single_1
"""

import argparse
import sys
import os

# Add project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from eval.datasets import (
    HotpotQABenchmark,
    NarrativeQABenchmark,
    LoCoMoBenchmark,
    RULERBenchmark,
)
from eval.datasets.base import BenchmarkConfig


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="GAM Framework evaluation tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate HotpotQA (using OpenAI GPT-4)
  python -m eval.run --dataset hotpotqa --data-path data/hotpotqa.json \\
      --generator openai --model gpt-4 --api-key YOUR_API_KEY

  # Evaluate NarrativeQA (using local VLLM model)
  python -m eval.run --dataset narrativeqa --data-path narrativeqa \\
      --generator vllm --model meta-llama/Llama-3-8B --max-samples 50

  # Evaluate RULER (specify dataset name)
  python -m eval.run --dataset ruler --data-path data/ruler_niah.jsonl \\
      --dataset-name niah_single_1 --retriever bm25
        """
    )

    # Dataset parameters
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["hotpotqa", "narrativeqa", "locomo", "ruler"],
        help="Dataset name"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Dataset path (file or directory)"
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="",
        help="Dataset subset name (only RULER needs this)"
    )
    
    # Generator parameters
    parser.add_argument(
        "--generator",
        type=str,
        default="openai",
        choices=["openai", "vllm"],
        help="Generator type"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4",
        help="Model name or path"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="OpenAI API Key (or read from environment variable OPENAI_API_KEY)"
    )
    parser.add_argument(
        "--api-base",
        type=str,
        default=None,
        help="OpenAI API Base URL"
    )
    
    # Retriever parameters
    parser.add_argument(
        "--retriever",
        type=str,
        default="dense",
        choices=["index", "bm25", "dense"],
        help="Retriever type"
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="BAAI/bge-base-en-v1.5",
        help="Embedding model path (Dense Retriever)"
    )
    
    # Evaluation parameters
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples (for quick testing)"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of parallel worker processes"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=2000,
        help="Text chunk size (number of tokens)"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Retrieve top-k relevant fragments"
    )
    
    # Output parameters
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Output directory"
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not save prediction results"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Quiet mode (reduce output)"
    )
    
    return parser.parse_args()


def main():
    """Main function"""
    args = parse_args()

    # If no API Key is provided, try to read from environment variable
    if args.generator == "openai" and not args.api_key:
        args.api_key = os.environ.get("OPENAI_API_KEY")
        if not args.api_key:
            print("Error: Using OpenAI Generator requires --api-key or setting environment variable OPENAI_API_KEY")
            sys.exit(1)

    # Create configuration
    config = BenchmarkConfig(
        data_path=args.data_path,
        generator_type=args.generator,
        model_name=args.model,
        api_key=args.api_key,
        api_base=args.api_base,
        retriever_type=args.retriever,
        embedding_model=args.embedding_model,
        max_samples=args.max_samples,
        num_workers=args.num_workers,
        chunk_size=args.chunk_size,
        top_k=args.top_k,
        output_dir=args.output_dir,
        save_predictions=not args.no_save,
        verbose=not args.quiet,
    )

    # Create corresponding Benchmark
    print(f"\n{'='*60}")
    print(f"GAM Framework - {args.dataset.upper()} Evaluation")
    print(f"{'='*60}\n")

    if args.dataset == "hotpotqa":
        benchmark = HotpotQABenchmark(config)
    elif args.dataset == "narrativeqa":
        benchmark = NarrativeQABenchmark(config)
    elif args.dataset == "locomo":
        benchmark = LoCoMoBenchmark(config)
    elif args.dataset == "ruler":
        benchmark = RULERBenchmark(config, dataset_name=args.dataset_name)
    else:
        print(f"Error: Unsupported dataset {args.dataset}")
        sys.exit(1)

    # Run evaluation
    try:
        results = benchmark.run()

        # Print results
        print(f"\n{'='*60}")
        print("Evaluation Results:")
        print(f"{'='*60}")
        for metric, value in results.items():
            if isinstance(value, float):
                print(f"  {metric:20s}: {value:.4f}")
            else:
                print(f"  {metric:20s}: {value}")
        print(f"{'='*60}\n")
        
    except Exception as e:
        print(f"\nError: Exception occurred during evaluation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

