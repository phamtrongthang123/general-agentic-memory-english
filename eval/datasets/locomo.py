# -*- coding: utf-8 -*-
"""
LoCoMo Dataset Evaluation

LoCoMo is a long conversation memory dataset that tests memory capabilities in multi-turn conversations
"""

import re
import json
from typing import Any, Dict, List, Optional, Tuple
from eval.datasets.base import BaseBenchmark, BenchmarkConfig
from eval.utils.metrics import compute_locomo_metrics


class LoCoMoBenchmark(BaseBenchmark):
    """LoCoMo Evaluation Benchmark"""

    def load_data(self) -> List[Dict[str, Any]]:
        """Load LoCoMo JSON data"""
        with open(self.config.data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, dict) and "samples" in data:
            return data["samples"]
        elif isinstance(data, list):
            return data
        else:
            raise ValueError("Unrecognized LoCoMo JSON format")
    
    def prepare_chunks(self, sample: Dict[str, Any]) -> List[str]:
        """
        Split conversation into multiple session chunks
        Each session serves as an independent memory unit
        """
        conv = sample.get("conversation", {})
        sessions = self._extract_sessions(conv)
        
        chunks = []
        for idx, timestamp, turns, session_summary in sessions:
            chunk = self._session_to_text(idx, timestamp, turns, session_summary)
            chunks.append(chunk)
        
        return chunks
    
    def extract_question(self, sample: Dict[str, Any]) -> str:
        """
        LoCoMo has multiple questions, here we return the first question
        In practice, you may need to iterate through all questions
        """
        qa_items = sample.get("qa", [])
        if qa_items:
            return qa_items[0].get("question", "")
        return ""
    
    def extract_ground_truth(self, sample: Dict[str, Any]) -> List[str]:
        """Extract ground truth answer"""
        qa_items = sample.get("qa", [])
        if qa_items:
            answer = qa_items[0].get("answer", "")
            return [answer] if answer else []
        return []
    
    def compute_metrics(
        self,
        predictions: List[str],
        ground_truths: List[List[str]]
    ) -> Dict[str, float]:
        """Compute F1 and BLEU-1 metrics (LoCoMo specific)"""
        return compute_locomo_metrics(predictions, ground_truths)
    
    def run(self) -> Dict[str, float]:
        """
        Override run method to handle multiple QA pairs
        Each LoCoMo sample may have multiple questions
        """
        print(f"Loading dataset: {self.config.data_path}")
        self.data = self.load_data()

        if self.config.max_samples:
            self.data = self.data[:self.config.max_samples]

        print(f"Loaded {len(self.data)} samples")

        # Initialize agents
        print("Initializing GAM Agent...")
        memory_agent, research_agent = self._setup_agents()

        # Run evaluation
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

                # Process all QA pairs
                qa_items = sample.get("qa", [])
                for qa in qa_items:
                    question = qa.get("question", "")
                    answer = qa.get("answer", "")

                    if not question:
                        continue

                    # Research
                    research_output = research_agent.research(
                        question=question,
                        top_k=self.config.top_k
                    )

                    prediction = research_output.final_answer
                    self.predictions.append(prediction)
                    ground_truths.append([answer] if answer else [""])

                    if self.config.verbose:
                        print(f"Question: {question[:80]}...")
                        print(f"Prediction: {prediction[:80]}...")
                        print(f"Ground truth: {answer[:80]}...")

            except Exception as e:
                print(f"Error processing sample {idx}: {e}")
                continue

        # Compute metrics
        print("\nComputing evaluation metrics...")
        self.results = self.compute_metrics(self.predictions, ground_truths)

        # Save results
        if self.config.save_predictions:
            self._save_results()

        return self.results
    
    def _extract_sessions(
        self,
        conv_obj: Dict[str, Any]
    ) -> List[Tuple[int, str, List[Dict[str, Any]], Optional[str]]]:
        """Extract session information"""
        sessions = []
        for k, v in conv_obj.items():
            m = re.match(r'^session_(\d+)$', k)
            if not (m and isinstance(v, list)):
                continue
            
            idx = int(m.group(1))
            timestamp = conv_obj.get(f"session_{idx}_date_time", "")
            summary = conv_obj.get(f"session_{idx}_summary", None)
            
            if summary and not isinstance(summary, str):
                summary = None
            
            sessions.append((idx, timestamp, v, summary))
        
        sessions.sort(key=lambda x: x[0])
        return sessions
    
    def _session_to_text(
        self,
        idx: int,
        timestamp: str,
        turns: List[Dict[str, Any]],
        session_summary: Optional[str]
    ) -> str:
        """Convert session to text"""
        lines = [
            f"=== SESSION {idx} - Dialogue Time: {timestamp} ===",
            ""
        ]
        
        for turn in turns:
            speaker = turn.get("speaker", "Unknown")
            dia_id = turn.get("dia_id", "")
            text = turn.get("text", "")
            lines.append(f"{speaker} ({dia_id}): {text}")
        
        if session_summary:
            lines.append("")
            lines.append(f"Session {idx} summary: {session_summary}")
        
        return "\n".join(lines)

