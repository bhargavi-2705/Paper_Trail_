"""Retrieval metrics. Plain Python on purpose: these numbers go on a resume,
so there should be nothing between me and the arithmetic.

A hit = a retrieved chunk whose (doc, page) is in the question's gold set.
Pages are matched with a +/- tolerance because a chunk near a page boundary
can legitimately answer a question labelled on the neighbouring page.
"""


def _is_hit(hit, gold_doc, gold_pages, page_tol=0):
    if gold_doc and hit["doc"] != gold_doc:
        return False
    return any(abs(hit["page"] - gp) <= page_tol for gp in gold_pages)


def hit_at_k(hits, gold_doc, gold_pages, k, page_tol=0):
    """1.0 if any gold page appears in the top-k, else 0.0."""
    return float(any(_is_hit(h, gold_doc, gold_pages, page_tol) for h in hits[:k]))


def recall_at_k(hits, gold_doc, gold_pages, k, page_tol=0):
    """Fraction of distinct gold pages covered by the top-k."""
    if not gold_pages:
        return 0.0
    found = {gp for gp in gold_pages
             for h in hits[:k] if _is_hit(h, gold_doc, [gp], page_tol)}
    return len(found) / len(gold_pages)


def precision_at_k(hits, gold_doc, gold_pages, k, page_tol=0):
    if not hits[:k]:
        return 0.0
    rel = sum(_is_hit(h, gold_doc, gold_pages, page_tol) for h in hits[:k])
    return rel / len(hits[:k])


def reciprocal_rank(hits, gold_doc, gold_pages, page_tol=0):
    for i, h in enumerate(hits, start=1):
        if _is_hit(h, gold_doc, gold_pages, page_tol):
            return 1.0 / i
    return 0.0


def aggregate(per_question, ks=(1, 3, 5, 10)):
    n = len(per_question) or 1
    out = {}
    for k in ks:
        out[f"hit@{k}"] = sum(q[f"hit@{k}"] for q in per_question) / n
        out[f"recall@{k}"] = sum(q[f"recall@{k}"] for q in per_question) / n
    out["precision@5"] = sum(q["precision@5"] for q in per_question) / n
    out["mrr"] = sum(q["mrr"] for q in per_question) / n
    out["n"] = len(per_question)
    return out


def score_question(hits, gold_doc, gold_pages, ks=(1, 3, 5, 10), page_tol=0):
    row = {"mrr": reciprocal_rank(hits, gold_doc, gold_pages, page_tol),
           "precision@5": precision_at_k(hits, gold_doc, gold_pages, 5, page_tol)}
    for k in ks:
        row[f"hit@{k}"] = hit_at_k(hits, gold_doc, gold_pages, k, page_tol)
        row[f"recall@{k}"] = recall_at_k(hits, gold_doc, gold_pages, k, page_tol)
    return row
