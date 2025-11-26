#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GAM Model Usage Example

Demonstrates how to use different LLM models (OpenAI API or local VLLM) with the GAM framework.
"""

import sys
import os

# Add project root directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

from gam import (
    MemoryAgent,
    OpenAIGenerator,
    OpenAIGeneratorConfig,
    InMemoryMemoryStore,
    InMemoryPageStore,
)


def openai_api_example():
    """OpenAI API model usage example"""
    print("=== OpenAI API Model Example ===\n")

    # 1. Configure OpenAI Generator
    gen_config = OpenAIGeneratorConfig(
        model="gpt-4o-mini",  # or "gpt-4", "gpt-3.5-turbo"
        api_key=os.getenv("OPENAI_API_KEY"),
        temperature=0.3,
        max_tokens=1000
    )

    # 2. Create Generator
    generator = OpenAIGenerator(gen_config)

    # 3. Create storage
    memory_store = InMemoryMemoryStore()
    page_store = InMemoryPageStore()

    # 4. Create MemoryAgent
    memory_agent = MemoryAgent(
        generator=generator,
        memory_store=memory_store,
        page_store=page_store
    )

    # 5. Test simple documents
    documents = [
        "Machine learning is a subset of artificial intelligence.",
        "Deep learning uses multi-layer neural networks.",
        "Natural language processing focuses on human language understanding."
    ]

    print(f"Processing {len(documents)} documents...")
    for doc in documents:
        memory_agent.memorize(doc)

    memory_state = memory_agent.get_memory_state()
    print(f"Constructed {len(memory_state.events)} memory events")
    print(f"Generated {len(memory_state.abstracts)} memory abstracts\n")
    
    return True


def custom_api_endpoint_example():
    """Custom API endpoint example (OpenAI-compatible third-party services)"""
    print("=== Custom API Endpoint Example ===\n")

    # 1. Configure OpenAI Generator with custom endpoint
    gen_config = OpenAIGeneratorConfig(
        model="gpt-4o-mini",
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url="https://your-custom-endpoint.com/v1",  # Custom endpoint
        temperature=0.3
    )

    # 2. Create Generator
    generator = OpenAIGenerator(gen_config)

    # 3. Create storage
    memory_store = InMemoryMemoryStore()
    page_store = InMemoryPageStore()

    # 4. Create MemoryAgent
    memory_agent = MemoryAgent(
        generator=generator,
        memory_store=memory_store,
        page_store=page_store
    )

    print("Configured custom API endpoint")
    print(f"   Endpoint: {gen_config.base_url}")
    print(f"   Model: {gen_config.model}\n")
    
    return True


def vllm_local_model_example():
    """VLLM local model usage example"""
    print("=== VLLM Local Model Example ===\n")

    try:
        from gam import VLLMGenerator, VLLMGeneratorConfig

        # 1. Configure VLLM Generator
        gen_config = VLLMGeneratorConfig(
            model_path="meta-llama/Llama-3-8B",  # Local model path
            temperature=0.7,
            max_tokens=512,
            gpu_memory_utilization=0.9
        )

        # 2. Create Generator
        generator = VLLMGenerator(gen_config)

        # 3. Create storage
        memory_store = InMemoryMemoryStore()
        page_store = InMemoryPageStore()

        # 4. Create MemoryAgent
        memory_agent = MemoryAgent(
            generator=generator,
            memory_store=memory_store,
            page_store=page_store
        )

        # 5. Test simple documents
        documents = [
            "AI is artificial intelligence.",
            "ML is machine learning.",
            "DL is deep learning."
        ]

        print(f"Processing {len(documents)} documents...")
        for doc in documents:
            memory_agent.memorize(doc)

        memory_state = memory_agent.get_memory_state()
        print(f"Constructed {len(memory_state.events)} memory events")
        print(f"Generated {len(memory_state.abstracts)} memory abstracts\n")

        return True

    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("   Please install: pip install vllm>=0.6.0")
        return False
    except Exception as e:
        print(f"Local model error: {e}")
        print("   Tip: If memory is limited, try using a smaller model")
        return False


def model_comparison():
    """Model comparison guide"""
    print("\n=== Model Selection Guide ===\n")

    print("OpenAI API Models:")
    print("   Advantages:")
    print("     - Quick start, no local resources required")
    print("     - Powerful performance and accuracy")
    print("     - Automatic updates and maintenance")
    print("   Disadvantages:")
    print("     - Requires network connection")
    print("     - Pay-per-use pricing")
    print("     - Data sent to external servers")
    print()

    print("VLLM Local Models:")
    print("   Advantages:")
    print("     - Fully offline operation")
    print("     - Data privacy protection")
    print("     - No usage limits")
    print("   Disadvantages:")
    print("     - Requires GPU resources")
    print("     - Need to download and manage models")
    print("     - May require more configuration")
    print()

    print("Recommendations:")
    print("   - Rapid prototyping and development: Use OpenAI API")
    print("   - Production environment and privacy requirements: Consider local VLLM")
    print("   - Large-scale usage: Choose based on cost and performance trade-offs")


def main():
    """Main function"""
    print("=" * 60)
    print("GAM Model Usage Examples")
    print("=" * 60)
    print()

    # Check API Key
    has_api_key = bool(os.getenv("OPENAI_API_KEY"))

    if not has_api_key:
        print("OPENAI_API_KEY environment variable not detected")
        print("   Some examples will not run")
        print("   To set: export OPENAI_API_KEY='your-api-key'\n")

    # Test OpenAI API model
    if has_api_key:
        try:
            openai_success = openai_api_example()
        except Exception as e:
            print(f"OpenAI API example failed: {e}\n")
            openai_success = False
    else:
        print("Skipping OpenAI API example (API Key not set)\n")
        openai_success = False

    # Custom endpoint example (configuration only, not actually run)
    custom_endpoint_success = custom_api_endpoint_example()

    # Test VLLM local model (optional)
    print("Test VLLM local model? (Requires GPU and model files)")
    print("Note: This will download and load large models, which takes considerable time")
    test_vllm = input("Enter 'yes' to continue, or press Enter to skip: ").strip().lower()

    if test_vllm == 'yes':
        vllm_success = vllm_local_model_example()
    else:
        print("Skipping VLLM local model example\n")
        vllm_success = False

    # Display model comparison
    model_comparison()

    # Summary
    print("\n" + "=" * 60)
    print("Example Summary")
    print("=" * 60)
    if openai_success:
        print("OpenAI API Model: Suitable for rapid prototyping and deployment")
    if custom_endpoint_success:
        print("Custom Endpoint: Suitable for using OpenAI-compatible third-party services")
    if vllm_success:
        print("VLLM Local Model: Suitable for privacy and offline usage")

    print("\nChoose the appropriate model type based on your needs!")
    print("\nMore information:")
    print("  - Check the eval/ directory for evaluation examples")
    print("  - Check gam/generator/ for generator implementations")


if __name__ == "__main__":
    main()
