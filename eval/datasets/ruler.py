# -*- coding: utf-8 -*-
"""
RULER Dataset Evaluation

RULER (Rule-based Understanding of Long-context Evaluation Resource)
is a dataset for testing long-context understanding capabilities
"""

import json
from typing import Any, Dict, List
from eval.datasets.base import BaseBenchmark, BenchmarkConfig
from eval.utils import chunk_text_smartly, compute_metrics, normalize_answer


class RULERBenchmark(BaseBenchmark):
    """RULER Evaluation Benchmark"""

    # Dataset-specific system prompts
    SYSTEM_PROMPTS = {
        "vt": "Memorize and track the chain(s) of variable assignment hidden in the following text.",
        "qa_1": "",
        "qa_2": "",
        "niah_single_1": "A special magic number is hidden within the following text. Make sure to memorize it. I will quiz you about the number afterwards.",
        "niah_single_2": "A special magic number is hidden within the following text. Make sure to memorize it. I will quiz you about the number afterwards.",
        "niah_single_3": "A special magic uuid is hidden within the following text. Make sure to memorize it. I will quiz you about the uuid afterwards.",
        "niah_multivalue": "",
        "niah_multiquery": "Some special magic numbers are hidden within the following text. You only need to memorize the special magic numbers. I will quiz you about the numbers afterwards.",
        "niah_multikey_1": "",
        "niah_multikey_2": "",
        "niah_multikey_3": "",
        "cwe": "Below is a numbered list of words. You only need to memorize the numbers that all words appear rather then make a abstract. I will quiz you about the numbers afterwards. Ignore the prompt below that asks you to summarize.",
        "fwe": "Read the following coded text and track the frequency of each coded word. Memorize the numbers that the words appear, I will quiz you about the numbers afterwards.",
    }
    
    def __init__(self, config: BenchmarkConfig, dataset_name: str = ""):
        super().__init__(config)
        self.dataset_name = dataset_name
    
    def load_data(self) -> List[Dict[str, Any]]:
        """Load RULER JSONL data"""
        data_all = []
        
        with open(self.config.data_path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    item = json.loads(line)
                    data_all.append({
                        "index": idx,
                        "context": item.get("context", ""),
                        "example": item.get("example", ""),
                        "instruction": item.get("instruction", ""),
                        "question": item.get("question", ""),
                        "outputs": item.get("outputs", []),
                    })
                except json.JSONDecodeError as e:
                    print(f"Skipping invalid JSON line {idx}: {e}")
                    continue
        
        return data_all
    
    def prepare_chunks(self, sample: Dict[str, Any]) -> List[str]:
        """
        Prepare context chunks
        For RULER, context is usually long and requires intelligent chunking
        """
        context = sample.get("context", "")
        if not context:
            return []

        # Add system prompt (if available)
        system_prompt = self._get_system_prompt()
        if system_prompt:
            context = f"{system_prompt}\n\n{context}"
        
        return chunk_text_smartly(
            context,
            max_tokens=self.config.chunk_size
        )
    
    def extract_question(self, sample: Dict[str, Any]) -> str:
        """
        Extract question
        RULER questions may contain both example and question
        """
        example = sample.get("example", "")
        question = sample.get("question", "")
        
        if example and question:
            return f"{example}\n\n{question}"
        elif question:
            return question
        else:
            return ""
    
    def extract_ground_truth(self, sample: Dict[str, Any]) -> List[str]:
        """Extract ground truth answer"""
        outputs = sample.get("outputs", [])
        if isinstance(outputs, list):
            return [str(o) for o in outputs if o]
        return [str(outputs)] if outputs else []
    
    def compute_metrics(
        self,
        predictions: List[str],
        ground_truths: List[List[str]]
    ) -> Dict[str, float]:
        """
        Compute Accuracy
        RULER primarily uses exact matching
        """
        if len(predictions) != len(ground_truths):
            raise ValueError("Number of predictions does not match number of ground truths")
        
        correct = 0
        for pred, gts in zip(predictions, ground_truths):
            normalized_pred = normalize_answer(pred)
            for gt in gts:
                if normalized_pred == normalize_answer(gt):
                    correct += 1
                    break
        
        accuracy = correct / len(predictions) if predictions else 0.0
        
        return {
            "accuracy": accuracy,
            "correct": correct,
            "total": len(predictions),
        }
    
    def _get_system_prompt(self) -> str:
        """Get dataset-specific system prompt"""
        if not self.dataset_name:
            return ""

        # Try exact match
        if self.dataset_name in self.SYSTEM_PROMPTS:
            return self.SYSTEM_PROMPTS[self.dataset_name]

        # Try fuzzy match (remove numeric suffix)
        base_name = self.dataset_name.split('_')[0]
        for key, prompt in self.SYSTEM_PROMPTS.items():
            if key.startswith(base_name):
                return prompt

        return ""

