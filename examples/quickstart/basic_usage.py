#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GAM Basic Usage Example

This example demonstrates how to use the GAM framework for basic memory building and Q&A.
Shows both direct list processing and utils-based text chunking approaches.
"""

import sys
import os

# Add project root directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

from gam import (
    MemoryAgent, 
    DeepResearchAgent,
    OpenRouterModel,
    build_session_chunks_from_text,
    build_pages_from_sessions_and_abstracts
)


def long_text_processing_example():
    """Long text processing example using utils chunking"""
    print("=== Long Text Processing Example ===")
    
    # 1. Create LLM model
    llm = OpenRouterModel(
        model="openai/gpt-4o-mini",
        base_url="https://api2.aigcbest.top/v1"
    )
    
    # 2. Create memory agent
    memory_agent = MemoryAgent(llm, temperature=0.3)
    
    # 3. Prepare long text (simulating a long document)
    long_text = """
    Artificial Intelligence (AI) is a branch of computer science dedicated to creating systems capable of performing tasks that typically require human intelligence.
    Machine Learning is a subset of AI that enables computers to learn without being explicitly programmed.
    Deep Learning is a subset of machine learning that uses multi-layer neural networks to simulate the way the human brain works.
    
    Natural Language Processing (NLP) is another important branch of AI, focusing on enabling computers to understand, interpret, and generate human language.
    Computer Vision is another key area of AI, dedicated to enabling computers to 'see' and understand visual information.
    
    Reinforcement Learning is a machine learning approach that learns optimal behavioral strategies through interaction with the environment.
    Neural Networks are the foundation of deep learning, consisting of interconnected nodes (neurons).
    
    Convolutional Neural Networks (CNNs) are particularly effective for image processing tasks, while Recurrent Neural Networks (RNNs) excel at processing sequential data.
    The introduction of the Transformer architecture has revolutionized the field of natural language processing, laying the foundation for large language models like GPT and BERT.
    """
    
    # 4. Use utils function to chunk the long text
    sessions = build_session_chunks_from_text(long_text, max_tokens=100, model="gpt-4o-mini")
    print(f"Text chunked into {len(sessions)} sessions")
    
    # 5. Run memory agent on chunked sessions
    print("Building memory from chunked text...")
    memory_history = memory_agent.run_memory_agent(sessions)
    
    # 6. Get final memory state with session abstracts
    memory_with_abstracts = memory_agent.get_memory_with_abstracts()
    print(f"Built {len(memory_with_abstracts.get('events', []))} memory events")
    print(f"Generated {len(memory_with_abstracts.get('session_abstracts', []))} session abstracts")
    
    # 7. Display memory abstract
    if memory_with_abstracts.get('abstract'):
        print(f"Memory abstract: {memory_with_abstracts['abstract']}")
    
    # 8. Show session abstracts
    session_abstracts = memory_with_abstracts.get('session_abstracts', [])
    for i, abstract in enumerate(session_abstracts, 1):
        if abstract.strip():
            print(f"Session {i} abstract: {abstract[:100]}...")
    
    return memory_with_abstracts


def list_processing_with_research_example():
    """List processing example with integrated deep research"""
    print("\n=== List Processing with Deep Research Example ===")
    
    # 1. Create LLM model
    llm = OpenRouterModel(
        model="openai/gpt-4o-mini",
        base_url="https://api2.aigcbest.top/v1"
    )
    
    # 2. Create memory agent
    memory_agent = MemoryAgent(llm, temperature=0.3)
    
    # 3. Prepare list of documents (direct list approach)
    documents = [
        "Artificial Intelligence (AI) is a branch of computer science that aims to create systems capable of performing tasks that typically require human intelligence.",
        "Machine Learning is a subset of AI that enables computers to learn without being explicitly programmed.",
        "Deep Learning is a subset of machine learning that uses multi-layer neural networks to simulate the way the human brain works.",
        "Natural Language Processing (NLP) is another important branch of AI, focusing on enabling computers to understand, interpret, and generate human language.",
        "Computer Vision is another key area of AI, dedicated to enabling computers to 'see' and understand visual information."
    ]
    
    print(f"Processing {len(documents)} documents directly...")
    
    # 4. Run memory agent directly on the list
    print("Building memory from document list...")
    memory_history = memory_agent.run_memory_agent(documents)
    
    # 5. Get final memory state with session abstracts
    memory_with_abstracts = memory_agent.get_memory_with_abstracts()
    print(f"Built {len(memory_with_abstracts.get('events', []))} memory events")
    print(f"Generated {len(memory_with_abstracts.get('session_abstracts', []))} session abstracts")
    
    # 6. Display memory abstract
    if memory_with_abstracts.get('abstract'):
        print(f"Memory abstract: {memory_with_abstracts['abstract']}")
    
    # 7. Now perform deep research using the built memory
    print("\n--- Starting Deep Research ---")
    
    # Create deep research agent
    research_agent = DeepResearchAgent(llm, temperature=0.3)
    
    # Build pages using session abstracts and documents
    session_abstracts = memory_with_abstracts.get('session_abstracts', [])
    pages = build_pages_from_sessions_and_abstracts(documents, session_abstracts)
    print(f"Built {len(pages)} pages for research")
    
    # Conduct deep research
    question = "What are the key differences between machine learning and deep learning?"
    print(f"Research question: {question}")
    
    result = research_agent.deep_research(
        question=question,
        memory=memory_with_abstracts,
        pages=pages,
        max_sessions=3
    )
    
    print(f"Research completed, used {result['iterations']} iterations")
    print(f"Used {len(result['session_ids_used'])} relevant documents")
    print(f"Research result: {result['summary']}")
    
    return memory_with_abstracts, result


def main():
    """Main function"""
    print("GAM Framework Quick Start Example")
    print("=" * 50)
    
    try:
        # Run long text processing example
        long_text_memory = long_text_processing_example()
        
        # Run list processing with integrated deep research
        list_memory, research_result = list_processing_with_research_example()
        
        print("\n=== Example Completed ===")
        print("✅ Long text processing successful")
        print("✅ List processing with deep research successful")
        print("\nYou can develop your own applications based on these examples!")
        
    except Exception as e:
        print(f"❌ Runtime error: {e}")
        print("Please check network connection and API configuration")


if __name__ == "__main__":
    main()
