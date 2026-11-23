"""Retrieval: dense -> (optional) BM25 hybrid via RRF -> (optional) CrossEncoder rerank.

Every stage is toggled from config.yaml so ablations.py can switch them off
one at a time and measure the delta. search() always returns plain dicts, so
the eval code never depends on LangChain internals.
"""
from functools import lru_cache

from .config import load_config
from .vectorstore import get_vectorstore


@lru_cache(maxsize=2)
def _get_reranker(model_name):
    from sentence_transformers import CrossEncoder
    return CrossEncoder(model_name)


def reciprocal_rank_fusion(ranked_lists, k=60):
    """Standard RRF: score = sum(1 / (k + rank)). Rank-based, so it needs no
    score normalisation between a cosine score and a BM25 score."""
    scores, payload = {}, {}
    for lst in ranked_lists:
        for rank, item in enumerate(lst, start=1):
            key = item["id"]
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            payload.setdefault(key, item)
    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    out = []
    for key, s in ordered:
        item = dict(payload[key])
        item["score"] = s
        out.append(item)
    return out


class Retriever:
    def __init__(self, cfg=None):
        self.cfg = cfg or load_config()
        self.vs = get_vectorstore(self.cfg)
        self._bm25 = None
        self._bm25_docs = None

    # --- stages ---------------------------------------------------------
    def _dense(self, query, k):
        hits = []
        for d, dist in self.vs.similarity_search_with_score(query, k=k):
            hits.append({
                "id": f"{d.metadata['doc']}::p{d.metadata['page']}::{hash(d.page_content) & 0xffff}",
                "text": d.page_content,
                "doc": d.metadata["doc"],
                "page": d.metadata["page"],
                "score": 1 - dist,
            })
        return hits

    def _ensure_bm25(self):
        if self._bm25 is not None:
            return
        from rank_bm25 import BM25Okapi
        raw = self.vs.get(include=["documents", "metadatas"])
        self._bm25_docs = [
            {
                "id": f"{m['doc']}::p{m['page']}::{hash(t) & 0xffff}",
                "text": t, "doc": m["doc"], "page": m["page"],
            }
            for t, m in zip(raw["documents"], raw["metadatas"])
        ]
        self._bm25 = BM25Okapi([d["text"].lower().split() for d in self._bm25_docs])

    def _sparse(self, query, k):
        self._ensure_bm25()
        scores = self._bm25.get_scores(query.lower().split())
        top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [dict(self._bm25_docs[i], score=float(scores[i])) for i in top]

    def _rerank(self, query, hits, k):
        if not hits:
            return hits
        model = _get_reranker(self.cfg["retrieval"]["rerank_model"])
        scores = model.predict([(query, h["text"]) for h in hits])
        for h, s in zip(hits, scores):
            h["rerank_score"] = float(s)
        return sorted(hits, key=lambda h: h["rerank_score"], reverse=True)[:k]

    # --- public ---------------------------------------------------------
    def probe(self, query, k=5):
        """Absolute-relevance scoring, for coverage analysis.

        search() may return Reciprocal Rank Fusion scores, which are purely
        rank-based: the top hit scores ~1/(k+1) whether it is a perfect match
        or the least-bad of a bad lot. That is fine for ordering and useless
        for "does the corpus cover this topic at all". So probe() uses raw
        dense cosine similarity, optionally refined by the CrossEncoder, both
        of which are absolute.
        """
        hits = self._dense(query, k)
        if self.cfg["retrieval"].get("rerank"):
            hits = self._rerank(query, hits, k)
        return hits

    def search(self, query, k=None):
        r = self.cfg["retrieval"]
        k = k or r["top_k"]
        cand_k = max(r.get("candidate_k", k), k)

        dense = self._dense(query, cand_k)
        if r.get("hybrid"):
            hits = reciprocal_rank_fusion([dense, self._sparse(query, cand_k)], r.get("rrf_k", 60))
        else:
            hits = dense

        if r.get("rerank"):
            return self._rerank(query, hits[:cand_k], k)
        return hits[:k]
