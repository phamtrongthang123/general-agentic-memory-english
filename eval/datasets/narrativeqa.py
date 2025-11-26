# -*- coding: utf-8 -*-
"""
NarrativeQA dataset evaluation

NarrativeQA is a narrative reading comprehension dataset containing long documents and related questions
"""

from typing import Any, Dict, List
from eval.datasets.base import BaseBenchmark, BenchmarkConfig
from eval.utils import chunk_text_by_sentences, compute_metrics


class NarrativeQABenchmark(BaseBenchmark):
    """NarrativeQA evaluation benchmark"""

    def load_data(self) -> List[Dict[str, Any]]:
        """Load NarrativeQA dataset"""
        from datasets import load_dataset

        # Load dataset
        dataset = load_dataset(self.config.data_path, split="test")
        
        data_all = []
        for idx, item in enumerate(dataset):
            document = item.get("document", {})
            question = item.get("question", {})
            answers = item.get("answers", [])
            
            data_all.append({
                "index": idx,
                "document_id": document.get("id", f"doc-{idx}"),
                "document_text": document.get("text", ""),
                "question_text": question.get("text", ""),
                "answers": [ans.get("text", "") for ans in answers if isinstance(ans, dict)],
            })
        
        return data_all
    
    def prepare_chunks(self, sample: Dict[str, Any]) -> List[str]:
        """Chunk document text (split by sentences to maintain coherence)"""
        document_text = sample.get("document_text", "")
        if not document_text:
            return []
        
        return chunk_text_by_sentences(
            document_text,
            max_tokens=self.config.chunk_size
        )
    
    def extract_question(self, sample: Dict[str, Any]) -> str:
        """Extract question"""
        return sample.get("question_text", "")
    
    def extract_ground_truth(self, sample: Dict[str, Any]) -> List[str]:
        """Extract ground truth answers"""
        answers = sample.get("answers", [])
        return [str(a) for a in answers if a]
    
    def compute_metrics(
        self,
        predictions: List[str],
        ground_truths: List[List[str]]
    ) -> Dict[str, float]:
        """Compute F1 metrics"""
        return compute_metrics(
            predictions, 
            ground_truths, 
            metrics=["f1"]
        )

