#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GAM Model Usage Example

Simple examples showing how to use different LLM models with GAM framework.
"""

import sys
import os

# Add project root directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

from gam import MemoryAgent, OpenRouterModel, HFModel


def api_model_example():
    """API model usage example"""
    print("=== API Model Example ===")
    
    # Create OpenRouter API model
    llm = OpenRouterModel(
        model="openai/gpt-4o-mini",
        base_url="https://api2.aigcbest.top/v1"
    )
    
    # Create memory agent
    memory_agent = MemoryAgent(llm, temperature=0.3)
    
    # Test with simple documents
    documents = [
        "Machine Learning is a subset of artificial intelligence.",
        "Deep Learning uses neural networks with multiple layers.",
        "Natural Language Processing focuses on human language understanding."
    ]
    
    print(f"Processing {len(documents)} documents...")
    memory_history = memory_agent.run_memory_agent(documents)
    memory_with_abstracts = memory_agent.get_memory_with_abstracts()
    
    print(f"✅ Built {len(memory_with_abstracts.get('events', []))} memory events")
    print(f"✅ Generated {len(memory_with_abstracts.get('session_abstracts', []))} session abstracts")
    
    return True


def local_model_example():
    """Local model usage example"""
    print("\n=== Local Model Example ===")
    
    try:
        # Create HuggingFace local model
        llm = HFModel(
            model_name="microsoft/DialoGPT-small",
            temperature=0.7,
            max_new_tokens=50
        )
        
        # Create memory agent
        memory_agent = MemoryAgent(llm, temperature=0.3)
        
        # Test with simple documents
        documents = [
            "AI is artificial intelligence.",
            "ML is machine learning.",
            "DL is deep learning."
        ]
        
        print(f"Processing {len(documents)} documents...")
        memory_history = memory_agent.run_memory_agent(documents)
        memory_with_abstracts = memory_agent.get_memory_with_abstracts()
        
        print(f"✅ Built {len(memory_with_abstracts.get('events', []))} memory events")
        print(f"✅ Generated {len(memory_with_abstracts.get('session_abstracts', []))} session abstracts")
        
        return True
        
    except ImportError as e:
        print(f"❌ Missing dependencies: {e}")
        print("Please install: pip install transformers torch")
        return False
    except Exception as e:
        print(f"❌ Local model error: {e}")
        print("Tip: Try a smaller model if you have limited memory")
        return False


def main():
    """Main function"""
    print("GAM Model Usage Example")
    print("=" * 50)
    
    # Test API model
    api_success = api_model_example()
    
    # Test local model
    local_success = local_model_example()
    
    print("\n=== Summary ===")
    if api_success:
        print("✅ API model: Good for quick prototyping and deployment")
    if local_success:
        print("✅ Local model: Good for privacy and offline usage")
    
    print("\n💡 Choose the model type that fits your needs!")


if __name__ == "__main__":
    main()
