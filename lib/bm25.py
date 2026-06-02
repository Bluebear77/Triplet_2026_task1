from dataclasses import dataclass
from typing import Dict, List
from collections import Counter
import math

from .utils import tokenize, normalize_text

# -------------------------
# Simple BM25 implementation
# -------------------------
@dataclass
class BM25Index:
    doc_ids: List[str]
    doc_len: List[int]
    tf: List[Counter]
    df: Counter
    avgdl: float
    N: int

    def idf(self, term: str) -> float:
        # BM25+ style idf (classic BM25)
        n_qi = self.df.get(term, 0)
        return math.log((self.N - n_qi + 0.5) / (n_qi + 0.5) + 1.0)

def build_bm25(docs: Dict[str, str]) -> BM25Index:
    doc_ids = []
    tf_list = []
    df = Counter()
    doc_len = []

    for doc_id, text in docs.items():
        toks = tokenize(text)
        c = Counter(toks)
        doc_ids.append(doc_id)
        tf_list.append(c)
        doc_len.append(len(toks))
        for term in c.keys():
            df[term] += 1

    N = len(doc_ids)
    avgdl = sum(doc_len) / N if N > 0 else 0.0
    return BM25Index(doc_ids=doc_ids, doc_len=doc_len, tf=tf_list, df=df, avgdl=avgdl, N=N)

def bm25_score(index: BM25Index, query: str, k1: float = 1.2, b: float = 0.75) -> Dict[str, float]:
    q_terms = tokenize(query)
    q_tf = Counter(q_terms)

    scores = {doc_id: 0.0 for doc_id in index.doc_ids}
    for i, doc_id in enumerate(index.doc_ids):
        dl = index.doc_len[i]
        denom_norm = k1 * (1 - b + b * (dl / (index.avgdl + 1e-9)))
        doc_tf = index.tf[i]

        s = 0.0
        for term, qf in q_tf.items():
            f = doc_tf.get(term, 0)
            if f == 0:
                continue
            idf = index.idf(term)
            numer = f * (k1 + 1.0)
            denom = f + denom_norm
            s += idf * (numer / (denom + 1e-9))
        scores[doc_id] = s

    return scores
