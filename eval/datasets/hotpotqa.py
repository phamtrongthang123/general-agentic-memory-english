# -*- coding: utf-8 -*-
"""
HotpotQA dataset evaluation

HotpotQA is a multi-hop question answering dataset
"""

import json
from typing import Any, Dict, List
from eval.datasets.base import BaseBenchmark, BenchmarkConfig
from eval.utils import chunk_text_smartly, compute_metrics


class HotpotQABenchmark(BaseBenchmark):
    """HotpotQA evaluation benchmark"""

    def load_data(self) -> List[Dict[str, Any]]:
        """Load HotpotQA JSON data"""
        with open(self.config.data_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        
        data_all = [
            {
                "index": item.get("index", idx),
                "context": item.get("context", ""),
                "input": item.get("input", ""),
                "answers": item.get("answers", []),
                "_id": f"hotpotqa-{item.get('index', idx)}"
            }
            for idx, item in enumerate(dataset)
        ]
        
        return data_all
    
    def prepare_chunks(self, sample: Dict[str, Any]) -> List[str]:
        """Chunk the context"""
        context = sample.get("context", "")
        if not context:
            return []
        
        return chunk_text_smartly(
            context,
            max_tokens=self.config.chunk_size
        )
    
    def extract_question(self, sample: Dict[str, Any]) -> str:
        """Extract question"""
        return sample.get("input", "")
    
    def extract_ground_truth(self, sample: Dict[str, Any]) -> List[str]:
        """Extract ground truth answers"""
        answers = sample.get("answers", [])
        if isinstance(answers, list):
            return [str(a) for a in answers]
        return [str(answers)]
    
    def compute_metrics(
        self,
        predictions: List[str],
        ground_truths: List[List[str]]
    ) -> Dict[str, float]:
        """Compute F1 metrics"""
        return compute_metrics(predictions, ground_truths, metrics=["f1"])

