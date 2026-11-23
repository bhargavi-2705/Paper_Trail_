"""Topic coverage analysis.

For each syllabus topic, probe the index and measure how well the corpus
supports it: best retrieval score, how many distinct documents respond, and
(optionally) an LLM verdict on whether the top excerpts *substantively* cover it.

A detector is only worth reporting if it passes a test you set up in advance:
topics you KNOW are present and topics you KNOW are absent (corpus.yaml). The
score is judged by how well it separates them (AUC), not by accuracy alone --
with one absent topic and five present ones, "everything is present" is already
83% accurate.
"""
from statistics import mean

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from .judge import parse_json

RELEVANCE = ChatPromptTemplate.from_messages([
    ("system",
     "Decide whether the excerpts substantively cover the given topic -- enough to study "
     "from. A passing mention, a citation, or a one-line aside does NOT count.\n"
     "Write the reason in plain text: no LaTeX and no backslashes.\n"
     'Reply with JSON only: {{"covers": true|false, "why": "<one short sentence>"}}'),
    ("human", "Topic: {topic}\n\nExcerpts:\n{context}"),
])


def topic_query(topic, queries=None):
    """Natural-language query for a topic slug. Bare acronyms ('peft') retrieve
    badly, so corpus.yaml can supply a fuller phrase."""
    if queries and topic in queries:
        return queries[topic]
    return topic.replace("-", " ")


def probe_topic(topic, retriever, k=5, queries=None):
    # probe() not search(): fused ranks carry no absolute relevance (see DECISIONS D1)
    probe = getattr(retriever, "probe", retriever.search)
    hits = probe(topic_query(topic, queries), k=k)
    if not hits:
        return {"topic": topic, "best_score": 0.0, "best_cosine": 0.0, "mean_score": 0.0,
                "n_docs": 0, "n_pages": 0, "hits": []}
    scores = [h.get("rerank_score", h["score"]) for h in hits]
    cos = [h["score"] for h in hits]
    return {
        "topic": topic,
        "best_score": max(scores),
        "best_cosine": max(cos),
        "mean_score": mean(scores),
        "n_docs": len({h["doc"] for h in hits}),
        "n_pages": len({(h["doc"], h["page"]) for h in hits}),
        "hits": hits,
    }


def llm_covers(topic_text, hits, llm):
    ctx = "\n\n".join(f"[{i}] ({h['doc']} p.{h['page']})\n{h['text'][:800]}"
                      for i, h in enumerate(hits, 1))
    try:
        raw = (RELEVANCE | llm | StrOutputParser()).invoke({"topic": topic_text, "context": ctx})
    except Exception as e:      # quota / network: record it, keep the run alive
        return None, f"error: {type(e).__name__}: {str(e)[:100]}"
    d = parse_json(raw, None)
    if isinstance(d, dict):
        return bool(d.get("covers")), d.get("why", "")
    return None, raw[:120]


def calibrate_threshold(probes, known_present, known_absent, field="best_score"):
    """Pick the cut-off that maximises correct separation of known topics.
    Returns (threshold, accuracy, n_labelled). Compare the accuracy with the
    majority-class baseline before believing it."""
    labelled = [(p[field], 1) for p in probes if p["topic"] in known_present]
    labelled += [(p[field], 0) for p in probes if p["topic"] in known_absent]
    if not labelled:
        return None, None, 0
    candidates = sorted({v for v, _ in labelled})
    best, best_acc = None, -1.0
    for c in candidates:
        acc = mean([float((v >= c) == bool(lbl)) for v, lbl in labelled])
        if acc > best_acc:
            best, best_acc = c, acc
    return best, best_acc, len(labelled)


def separability(probes, known_present, known_absent, field="best_score"):
    """AUC: the chance that a random known-present topic outscores a random
    known-absent one. 1.0 = perfect, 0.5 = coin flip, below 0.5 = backwards."""
    pos = [p[field] for p in probes if p["topic"] in known_present]
    neg = [p[field] for p in probes if p["topic"] in known_absent]
    if not pos or not neg:
        return None
    wins = sum((x > y) + 0.5 * (x == y) for x in pos for y in neg)
    return wins / (len(pos) * len(neg))


def verdict_accuracy(rows, known_present, known_absent):
    """How often the LLM verdict matches what you know. A missing verdict counts as wrong."""
    labelled = [(r, 1 if r["topic"] in known_present else 0)
                for r in rows if r["topic"] in known_present or r["topic"] in known_absent]
    if not labelled:
        return None, 0
    ok = sum(1 for r, lbl in labelled
             if r.get("llm_covers") is not None and int(bool(r["llm_covers"])) == lbl)
    return ok / len(labelled), len(labelled)


def analyse(syllabus, retriever, threshold=None, llm=None, k=5, queries=None, field="best_score"):
    probes = [probe_topic(t, retriever, k, queries) for t in syllabus]
    if threshold is None:
        threshold = mean([p[field] for p in probes]) if probes else 0.0
    rows = []
    for p in probes:
        row = {kk: vv for kk, vv in p.items() if kk != "hits"}
        row["field"] = field
        row["score_used"] = p[field]
        row["flagged_gap"] = p[field] < threshold
        if llm is not None:
            covers, why = llm_covers(topic_query(p["topic"], queries), p["hits"], llm)
            row["llm_covers"], row["llm_why"] = covers, why
        row["top_sources"] = [f"{h['doc']} p.{h['page']}" for h in p["hits"][:3]]
        rows.append(row)
    return {"threshold": threshold, "field": field, "topics": rows}
