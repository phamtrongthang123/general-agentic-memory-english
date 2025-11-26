#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GAM Basic Usage Example

This example demonstrates how to use the GAM framework for basic memory construction and question answering.
It showcases the complete workflow of memory construction, retrieval, and research.
"""

import sys
import os

# Add project root directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

from gam import (
    MemoryAgent,
    ResearchAgent,
    OpenAIGenerator,
    OpenAIGeneratorConfig,
    InMemoryMemoryStore,
    InMemoryPageStore,
    DenseRetriever,
    DenseRetrieverConfig,
)


def basic_memory_example():
    """Basic memory construction example"""
    print("=== Basic Memory Construction Example ===\n")

    # 1. Configure and create Generator
    gen_config = OpenAIGeneratorConfig(
        model="gpt-4o-mini",
        api_key=os.getenv("OPENAI_API_KEY"),  # Read from environment variable
        temperature=0.3
    )
    generator = OpenAIGenerator(gen_config)

    # 2. Create storage
    memory_store = InMemoryMemoryStore()
    page_store = InMemoryPageStore()

    # 3. Create MemoryAgent
    memory_agent = MemoryAgent(
        generator=generator,
        memory_store=memory_store,
        page_store=page_store
    )

    # 4. Prepare text to memorize (simulating long documents)
    documents = [
        """Artificial Intelligence (AI) is a branch of computer science dedicated to creating systems that can perform tasks that typically require human intelligence.
        Machine learning is a subset of AI that enables computers to learn without being explicitly programmed.""",

        """Deep learning is a subset of machine learning that uses multi-layer neural networks to simulate how the human brain works.
        Natural Language Processing (NLP) is another important branch of AI that focuses on enabling computers to understand, interpret, and generate human language.""",

        """Computer vision is another key area of AI that aims to enable computers to "see" and understand visual information.
        Reinforcement learning is a machine learning method that learns optimal behavioral strategies through interaction with the environment.""",

        """Neural networks are the foundation of deep learning, consisting of interconnected nodes (neurons).
        Convolutional Neural Networks (CNNs) are particularly suitable for image processing tasks, while Recurrent Neural Networks (RNNs) excel at handling sequential data.""",

        """The introduction of the Transformer architecture has revolutionized the field of natural language processing, laying the foundation for large language models such as GPT and BERT."""
    ]

    # 5. Memorize documents one by one
    print(f"Memorizing {len(documents)} documents...")
    for i, doc in enumerate(documents, 1):
        print(f"  Memorizing document {i}/{len(documents)}...")
        memory_agent.memorize(doc)

    # 6. View memory state
    memory_state = memory_agent.get_memory_state()
    print(f"\nSuccessfully constructed memory:")
    print(f"  - Number of memory events: {len(memory_state.events)}")
    print(f"  - Number of memory abstracts: {len(memory_state.abstracts)}")

    if memory_state.abstracts:
        print(f"\nMemory summary overview:")
        for i, abstract in enumerate(memory_state.abstracts[:3], 1):
            print(f"  {i}. {abstract.content[:100]}...")
    
    return memory_agent, memory_store, page_store


def research_example(memory_agent, memory_store, page_store):
    """Memory-based research example"""
    print("\n=== Memory-Based Research Example ===\n")

    # 1. Configure and create Generator
    gen_config = OpenAIGeneratorConfig(
        model="gpt-4o-mini",
        api_key=os.getenv("OPENAI_API_KEY"),
        temperature=0.3
    )
    generator = OpenAIGenerator(gen_config)

    # 2. Create retriever
    retriever_config = DenseRetrieverConfig(
        model_path="BAAI/bge-base-en-v1.5",
        top_k=5
    )
    retriever = DenseRetriever(
        config=retriever_config,
        memory_store=memory_store,
        page_store=page_store
    )

    # 3. Create ResearchAgent
    research_agent = ResearchAgent(
        generator=generator,
        retriever=retriever
    )

    # 4. Conduct research
    question = "What are the key differences between machine learning and deep learning?"
    print(f"Research question: {question}\n")

    result = research_agent.research(question=question, top_k=3)

    # 5. Display results
    print(f"Research completed:")
    print(f"  - Number of iterations: {len(result.iterations)}")
    print(f"  - Sufficient: {result.enough}")
    print(f"\nFinal answer:")
    print(f"  {result.final_answer}")
    
    return result


def main():
    """Main function"""
    print("=" * 60)
    print("GAM Framework Quick Start Example")
    print("=" * 60)
    print()

    # Check API Key
    if not os.getenv("OPENAI_API_KEY"):
        print("Please set the OPENAI_API_KEY environment variable")
        print("   export OPENAI_API_KEY='your-api-key'")
        return

    try:
        # 1. Run basic memory construction example
        memory_agent, memory_store, page_store = basic_memory_example()

        # 2. Run memory-based research example
        research_result = research_example(memory_agent, memory_store, page_store)

        print("\n" + "=" * 60)
        print("Example completed successfully!")
        print("=" * 60)
        print("\nYou can develop your own applications based on these examples!")
        print("\nTips:")
        print("  - Modify document content to test different scenarios")
        print("  - Try different questions to test research capabilities")
        print("  - Check the eval/ directory for more evaluation examples")

    except Exception as e:
        print(f"\nExecution error: {e}")
        print("\nPlease check:")
        print("  1. Network connection is working")
        print("  2. API Key is correct")
        print("  3. Required dependencies are installed: pip install -r requirements.txt")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
