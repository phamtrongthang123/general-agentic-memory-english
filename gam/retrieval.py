import math
from collections import Counter
from typing import List, Dict, Tuple

from .utils import tokenize


class BM25Sessions:
    """轻量 BM25（对 sessions 文本做检索）"""
    
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_ids: List[int] = []
        self.doc_tokens: List[List[str]] = []
        self.df: Counter = Counter()
        self.idf: Dict[str, float] = {}
        self.avgdl: float = 0.0
        self.N: int = 0
        self.dl: List[int] = []

    def build(self, documents: List[Tuple[int, str]]):
        """
        documents: [(session_id, text), ...]
        """
        self.doc_ids.clear()
        self.doc_tokens.clear()
        self.df.clear()
        self.idf.clear()
        self.dl.clear()

        for sid, text in documents:
            toks = tokenize(text or "")
            self.doc_ids.append(sid)
            self.doc_tokens.append(toks)
            self.dl.append(len(toks))
            for term in set(toks):
                self.df[term] += 1

        self.N = len(self.doc_ids)
        self.avgdl = (sum(self.dl) / self.N) if self.N else 0.0

        # 经典 BM25 idf
        for term, dfi in self.df.items():
            # 加 0.5 做平滑
            self.idf[term] = math.log(1 + (self.N - dfi + 0.5) / (dfi + 0.5))

    def _score_query_doc(self, q_terms: List[str], doc_idx: int) -> float:
        if not q_terms or doc_idx >= len(self.doc_tokens):
            return 0.0
        toks = self.doc_tokens[doc_idx]
        dl = self.dl[doc_idx] if self.dl else 0
        tf = Counter(toks)
        score = 0.0
        for t in q_terms:
            if t not in self.idf:
                continue
            f = tf.get(t, 0)
            if f == 0:
                continue
            idf = self.idf[t]
            denom = f + self.k1 * (1 - self.b + self.b * (dl / (self.avgdl + 1e-9)))
            score += idf * (f * (self.k1 + 1)) / (denom + 1e-9)
        return score

    def search(self, query_terms: List[str], topk: int = 10) -> List[int]:
        if not self.N or not query_terms:
            return []
        q_terms = [t for t in query_terms if t in self.idf]
        if not q_terms:
            # 若所有 query 词在索引外，直接返回空
            return []
        scored = []
        for i in range(self.N):
            s = self._score_query_doc(q_terms, i)
            if s > 0:
                scored.append((s, self.doc_ids[i]))
        scored.sort(reverse=True)
        return [sid for s, sid in scored[:topk]]
