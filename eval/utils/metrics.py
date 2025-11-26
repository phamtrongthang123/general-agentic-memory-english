# -*- coding: utf-8 -*-
"""
Evaluation Metrics Utilities Module

Provides common NLP evaluation metrics:
- Exact Match (EM)
- F1 Score
- BLEU-1
- ROUGE
"""

import re
import string
import math
from collections import Counter
from typing import List, Dict, Any, Union


def normalize_answer(s: str) -> str:
    """
    Normalize answer text

    - Convert to lowercase
    - Remove punctuation
    - Remove articles
    - Normalize whitespace
    """
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    
    def white_space_fix(text):
        return ' '.join(text.split())
    
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    
    def lower(text):
        return text.lower()
    
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def exact_match_score(prediction: str, ground_truths: List[str]) -> float:
    """
    Compute Exact Match score

    Args:
        prediction: Predicted answer
        ground_truths: List of ground truth answers

    Returns:
        1.0 if matches any ground truth answer, otherwise 0.0
    """
    normalized_pred = normalize_answer(prediction)
    for gt in ground_truths:
        if normalized_pred == normalize_answer(gt):
            return 1.0
    return 0.0


def f1_score(prediction: str, ground_truths: List[str]) -> float:
    """
    Compute F1 score (word-level)

    Args:
        prediction: Predicted answer
        ground_truths: List of ground truth answers

    Returns:
        Maximum F1 score
    """
    normalized_pred = normalize_answer(prediction)
    pred_tokens = normalized_pred.split()
    
    max_f1 = 0.0
    for gt in ground_truths:
        normalized_gt = normalize_answer(gt)
        gt_tokens = normalized_gt.split()
        
        common = Counter(pred_tokens) & Counter(gt_tokens)
        num_same = sum(common.values())
        
        if num_same == 0:
            f1 = 0.0
        else:
            precision = num_same / len(pred_tokens)
            recall = num_same / len(gt_tokens)
            f1 = 2 * precision * recall / (precision + recall)
        
        max_f1 = max(max_f1, f1)
    
    return max_f1


def rouge_score(
    prediction: str,
    ground_truths: List[str],
    rouge_type: str = "rouge-l"
) -> float:
    """
    Compute ROUGE score

    Args:
        prediction: Predicted answer
        ground_truths: List of ground truth answers
        rouge_type: ROUGE type ("rouge-1", "rouge-2", "rouge-l")

    Returns:
        Maximum ROUGE F1 score
    """
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer([rouge_type], use_stemmer=True)

        max_score = 0.0
        for gt in ground_truths:
            scores = scorer.score(gt, prediction)
            score = scores[rouge_type].fmeasure
            max_score = max(max_score, score)

        return max_score
    except ImportError:
        # If rouge_score is not available, return F1 score as fallback
        return f1_score(prediction, ground_truths)


def compute_metrics(
    predictions: List[str],
    ground_truths_list: List[List[str]],
    metrics: List[str] = ["em", "f1"]
) -> Dict[str, float]:
    """
    Batch compute evaluation metrics

    Args:
        predictions: List of predicted answers
        ground_truths_list: List of ground truth answer lists
        metrics: List of metrics to compute ["em", "f1", "rouge"]

    Returns:
        Average scores for each metric
    """
    if len(predictions) != len(ground_truths_list):
        raise ValueError("Number of predictions does not match number of ground truths")
    
    results = {metric: [] for metric in metrics}
    
    for pred, gts in zip(predictions, ground_truths_list):
        if not isinstance(gts, list):
            gts = [gts]
        
        if "em" in metrics:
            results["em"].append(exact_match_score(pred, gts))
        
        if "f1" in metrics:
            results["f1"].append(f1_score(pred, gts))
        
        if "rouge" in metrics or "rouge-l" in metrics:
            results.setdefault("rouge-l", []).append(rouge_score(pred, gts, "rouge-l"))
        
        if "rouge-1" in metrics:
            results.setdefault("rouge-1", []).append(rouge_score(pred, gts, "rouge-1"))
        
        if "rouge-2" in metrics:
            results.setdefault("rouge-2", []).append(rouge_score(pred, gts, "rouge-2"))

    # Compute averages
    avg_results = {
        metric: sum(scores) / len(scores) if scores else 0.0
        for metric, scores in results.items()
    }
    
    return avg_results


def compute_accuracy(predictions: List[str], ground_truths: List[str]) -> float:
    """
    Compute accuracy (strict matching)

    Args:
        predictions: List of predicted answers
        ground_truths: List of ground truth answers

    Returns:
        Accuracy
    """
    if len(predictions) != len(ground_truths):
        raise ValueError("Number of predictions does not match number of ground truths")
    
    correct = sum(
        1 for pred, gt in zip(predictions, ground_truths)
        if normalize_answer(pred) == normalize_answer(gt)
    )
    
    return correct / len(predictions) if predictions else 0.0


# ========== LoCoMo-Specific Evaluation Metrics ==========

def normalize_text_locomo(s: str) -> str:
    """
    LoCoMo dataset-specific text normalization
    Consistent with normalize_text in original eval/evalution_locomo.py
    """
    if s is None:
        return ""
    s = str(s)
    s = s.lower().strip()
    s = re.sub(r"[^\w\s]", " ", s)  # remove punctuation
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"(^|\s)(a|an|the)(\s|$)", " ", s)  # drop english articles
    s = re.sub(r"\s+", " ", s).strip()
    return s


def tokens_locomo(s: str):
    """LoCoMo tokenization"""
    s = normalize_text_locomo(s)
    return s.split() if s else []


def f1_score_locomo(pred: str, gold: str) -> float:
    """
    LoCoMo dataset-specific F1 computation
    Consistent with f1_score in original eval/evalution_locomo.py
    """
    gtoks = tokens_locomo(gold)
    ptoks = tokens_locomo(pred)
    if not gtoks and not ptoks:
        return 1.0
    if not gtoks or not ptoks:
        return 0.0
    gcount = Counter(gtoks)
    pcount = Counter(ptoks)
    overlap = sum(min(pcount[t], gcount[t]) for t in pcount)
    if overlap == 0:
        return 0.0
    precision = overlap / len(ptoks)
    recall = overlap / len(gtoks)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def bleu1_score(pred: str, gold: str) -> float:
    """
    Compute BLEU-1 score
    Consistent with bleu1_score in original eval/evalution_locomo.py
    """
    gtoks = tokens_locomo(gold)
    ptoks = tokens_locomo(pred)
    if len(ptoks) == 0:
        return 0.0
    gcount = Counter(gtoks)
    pcount = Counter(ptoks)
    clipped = sum(min(pcount[t], gcount[t]) for t in pcount)
    precision = clipped / len(ptoks) if ptoks else 0.0
    if ptoks and gtoks:
        bp = 1.0 if len(ptoks) >= len(gtoks) else math.exp(1 - len(gtoks)/len(ptoks))
    else:
        bp = 0.0
    return bp * precision


def compute_locomo_metrics(
    predictions: List[str],
    ground_truths_list: List[List[str]]
) -> Dict[str, float]:
    """
    Compute LoCoMo-specific evaluation metrics (F1 and BLEU-1)

    Args:
        predictions: List of predicted answers
        ground_truths_list: List of ground truth answer lists

    Returns:
        Dictionary of evaluation metrics
    """
    if len(predictions) != len(ground_truths_list):
        raise ValueError("Number of predictions does not match number of ground truths")

    f1_scores = []
    bleu1_scores = []

    for pred, gts in zip(predictions, ground_truths_list):
        if not isinstance(gts, list):
            gts = [gts]

        # Compute score for each ground truth, take maximum
        max_f1 = max([f1_score_locomo(pred, gt) for gt in gts]) if gts else 0.0
        max_bleu1 = max([bleu1_score(pred, gt) for gt in gts]) if gts else 0.0
        
        f1_scores.append(max_f1)
        bleu1_scores.append(max_bleu1)
    
    return {
        "f1": sum(f1_scores) / len(f1_scores) if f1_scores else 0.0,
        "bleu1": sum(bleu1_scores) / len(bleu1_scores) if bleu1_scores else 0.0,
    }

