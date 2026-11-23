"""Adaptive quizzing over weak topics.

Questions are generated from retrieved excerpts, so each one has a known
source page. Grading is LLM-based and should be validated against hand grades
with judge.validate_judge() before any accuracy number is reported.
"""
import json

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

GEN = ChatPromptTemplate.from_messages([
    ("system",
     "Write {n} short exam questions on the topic, answerable purely from the excerpts. "
     "Mix recall and reasoning. Each needs a concise model answer.\n"
     'Reply with JSON only: [{{"question": "...", "model_answer": "...", "excerpt": <index int>}}]'),
    ("human", "Topic: {topic}\n\nExcerpts:\n{context}"),
])

GRADE = ChatPromptTemplate.from_messages([
    ("system",
     "Grade a student's answer against the model answer and the source excerpt. "
     "Be strict about factual errors, lenient about wording.\n"
     'Reply with JSON only: {{"correct": true|false, "score": 0-1, "feedback": "<one or two sentences>"}}'),
    ("human", "Question: {question}\n\nModel answer: {model_answer}\n\n"
              "Source excerpt: {excerpt}\n\nStudent answer: {student_answer}"),
])


def _parse(text, fallback):
    text = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        return json.loads(text)
    except Exception:
        return fallback


def generate_quiz(topic, retriever, llm, n=3, k=4):
    hits = retriever.search(topic.replace("-", " "), k=k)
    if not hits:
        return []
    ctx = "\n\n".join(f"[{i}] ({h['doc']} p.{h['page']})\n{h['text'][:1000]}"
                      for i, h in enumerate(hits, 1))
    raw = (GEN | llm | StrOutputParser()).invoke({"topic": topic, "n": n, "context": ctx})
    items = _parse(raw, [])
    out = []
    for it in items if isinstance(items, list) else []:
        idx = it.get("excerpt")
        src = hits[idx - 1] if isinstance(idx, int) and 1 <= idx <= len(hits) else hits[0]
        out.append({
            "topic": topic,
            "question": it.get("question", ""),
            "model_answer": it.get("model_answer", ""),
            "source": {"doc": src["doc"], "page": src["page"]},
            "excerpt": src["text"][:1200],
        })
    return out


def grade(item, student_answer, llm):
    raw = (GRADE | llm | StrOutputParser()).invoke({
        "question": item["question"], "model_answer": item["model_answer"],
        "excerpt": item["excerpt"], "student_answer": student_answer,
    })
    d = _parse(raw, {"correct": None, "score": None, "feedback": raw[:200]})
    return d if isinstance(d, dict) else {"correct": None, "score": None, "feedback": str(d)[:200]}


def next_topics(coverage_rows, quiz_history, limit=3):
    """Adaptive bit: prioritise topics that are covered by the corpus (so
    questions are answerable) but that you have scored badly on."""
    scores = {}
    for h in quiz_history:
        scores.setdefault(h["topic"], []).append(float(h.get("score") or 0))
    ranked = []
    for row in coverage_rows:
        if row.get("flagged_gap"):
            continue                       # can't quiz what the corpus lacks
        hist = scores.get(row["topic"])
        avg = sum(hist) / len(hist) if hist else None
        ranked.append((avg if avg is not None else 0.5, row["topic"]))
    ranked.sort(key=lambda t: t[0])
    return [t for _, t in ranked[:limit]]
