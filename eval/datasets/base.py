# -*- coding: utf-8 -*-
"""
Benchmark base class

All dataset evaluations should inherit from this base class
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from dataclasses import dataclass


@dataclass
class BenchmarkConfig:
    """Evaluation configuration"""
    # Data path
    data_path: str

    # Model configuration
    generator_type: str = "openai"  # "openai" or "vllm"
    model_name: str = "gpt-4"
    api_key: Optional[str] = None
    api_base: Optional[str] = None

    # Retriever configuration
    retriever_type: str = "dense"  # "index", "bm25", "dense"
    embedding_model: Optional[str] = None

    # Evaluation configuration
    max_samples: Optional[int] = None
    num_workers: int = 4
    chunk_size: int = 2000
    top_k: int = 5

    # Output configuration
    output_dir: str = "outputs"
    save_predictions: bool = True
    verbose: bool = True


class BaseBenchmark(ABC):
    """Benchmark base class"""

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.data = []
        self.predictions = []
        self.results = {}

    @abstractmethod
    def load_data(self) -> List[Dict[str, Any]]:
        """Load dataset"""
        pass

    @abstractmethod
    def prepare_chunks(self, sample: Dict[str, Any]) -> List[str]:
        """Prepare text chunks for memorization"""
        pass

    @abstractmethod
    def extract_question(self, sample: Dict[str, Any]) -> str:
        """Extract question"""
        pass

    @abstractmethod
    def extract_ground_truth(self, sample: Dict[str, Any]) -> List[str]:
        """Extract ground truth answers"""
        pass

    @abstractmethod
    def compute_metrics(self, predictions: List[str], ground_truths: List[List[str]]) -> Dict[str, float]:
        """Compute evaluation metrics"""
        pass
    
    def run(self) -> Dict[str, float]:
        """
        Run complete evaluation pipeline

        Returns:
            Evaluation results dictionary
        """
        # 1. Load data
        print(f"Loading dataset: {self.config.data_path}")
        self.data = self.load_data()

        if self.config.max_samples:
            self.data = self.data[:self.config.max_samples]

        print(f"Loaded {len(self.data)} samples")

        # 2. Initialize Agent
        print("Initializing GAM Agent...")
        memory_agent, research_agent = self._setup_agents()

        # 3. Run evaluation
        print("Starting evaluation...")
        self.predictions = []
        ground_truths = []

        for idx, sample in enumerate(self.data):
            if self.config.verbose:
                print(f"\nProcessing sample {idx + 1}/{len(self.data)}")

            try:
                # Prepare chunks and memorize
                chunks = self.prepare_chunks(sample)
                for chunk in chunks:
                    memory_agent.memorize(chunk)

                # Extract question and research
                question = self.extract_question(sample)
                research_output = research_agent.research(
                    question=question,
                    top_k=self.config.top_k
                )

                prediction = research_output.final_answer
                self.predictions.append(prediction)

                # Extract ground truth
                gt = self.extract_ground_truth(sample)
                ground_truths.append(gt)

                if self.config.verbose:
                    print(f"Prediction: {prediction[:100]}...")
                    print(f"Ground truth: {gt[0][:100] if gt else 'N/A'}...")

            except Exception as e:
                print(f"Error processing sample {idx}: {e}")
                self.predictions.append("")
                ground_truths.append([""])

        # 4. Compute metrics
        print("\nComputing evaluation metrics...")
        self.results = self.compute_metrics(self.predictions, ground_truths)

        # 5. Save results
        if self.config.save_predictions:
            self._save_results()

        return self.results
    
    def _setup_agents(self):
        """Setup Memory and Research Agent"""
        from gam import (
            MemoryAgent,
            ResearchAgent,
            OpenAIGenerator,
            VLLMGenerator,
            OpenAIGeneratorConfig,
            VLLMGeneratorConfig,
            InMemoryMemoryStore,
            InMemoryPageStore,
            IndexRetriever,
            BM25Retriever,
            DenseRetriever,
            IndexRetrieverConfig,
            BM25RetrieverConfig,
            DenseRetrieverConfig,
        )
        
        # Create Generator
        if self.config.generator_type == "openai":
            gen_config = OpenAIGeneratorConfig(
                model=self.config.model_name,
                api_key=self.config.api_key,
                base_url=self.config.api_base,
            )
            generator = OpenAIGenerator(gen_config)
        elif self.config.generator_type == "vllm":
            gen_config = VLLMGeneratorConfig(
                model_path=self.config.model_name,
            )
            generator = VLLMGenerator(gen_config)
        else:
            raise ValueError(f"Unknown generator type: {self.config.generator_type}")
        
        # Create stores
        memory_store = InMemoryMemoryStore()
        page_store = InMemoryPageStore()

        # Create retriever
        if self.config.retriever_type == "index":
            retriever_config = IndexRetrieverConfig()
            retriever = IndexRetriever(retriever_config, memory_store, page_store)
        elif self.config.retriever_type == "bm25":
            retriever_config = BM25RetrieverConfig()
            retriever = BM25Retriever(retriever_config, memory_store, page_store)
        elif self.config.retriever_type == "dense":
            retriever_config = DenseRetrieverConfig(
                model_path=self.config.embedding_model or "BAAI/bge-base-en-v1.5"
            )
            retriever = DenseRetriever(retriever_config, memory_store, page_store)
        else:
            raise ValueError(f"Unknown retriever type: {self.config.retriever_type}")
        
        # Create Agent
        memory_agent = MemoryAgent(
            generator=generator,
            memory_store=memory_store,
            page_store=page_store,
        )
        
        research_agent = ResearchAgent(
            generator=generator,
            retriever=retriever,
        )
        
        return memory_agent, research_agent
    
    def _save_results(self):
        """Save evaluation results"""
        import os
        import json
        from datetime import datetime
        
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = os.path.join(
            self.config.output_dir,
            f"{self.__class__.__name__}_{timestamp}.json"
        )
        
        output = {
            "config": {
                "data_path": self.config.data_path,
                "generator_type": self.config.generator_type,
                "model_name": self.config.model_name,
                "retriever_type": self.config.retriever_type,
                "num_samples": len(self.data),
            },
            "metrics": self.results,
            "predictions": [
                {
                    "prediction": pred,
                    "ground_truth": self.extract_ground_truth(sample),
                }
                for pred, sample in zip(self.predictions, self.data)
            ]
        }
        
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        
        print(f"\nResults saved to: {result_file}")

