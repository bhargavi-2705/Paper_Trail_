"""LLM-as-judge for faithfulness, plus the validation harness that keeps it honest.

faithfulness = share of the answer's claims supported by the retrieved excerpts.
No gold answers needed, which is why it's usable on a self-built corpus.

validate_judge() compares the judge against your own hand grades and reports
raw agreement + Cohen's kappa. Report those alongside any faithfulness number;
an unvalidated judge score is just a vibe.
"""
import json
import re

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

FAITHFULNESS = ChatPromptTemplate.from_messages([
    ("system",
     "You check whether an answer is grounded in the excerpts it was given.\n"
     "Split the answer into atomic factual claims. For each, decide if the excerpts "
     "support it. Ignore hedges and statements about what is missing.\n"
     "Write claims in plain text: no LaTeX and no backslashes.\n"
     'Reply with JSON only: {{"claims": [{{"claim": "...", "supported": true|false}}], '
     '"score": <supported/total as a float 0-1>}}'),
    ("human", "Excerpts:\n{context}\n\nAnswer:\n{answer}"),
])

# A backslash is only legal in JSON as part of  \"  \\  \/  \b  \f  \n  \r  \t
# or \uXXXX. Models quoting maths write things like \( \sqrt{d_k} \), which
# makes json.loads raise -- and a silent fallback then drops exactly the
# formula-heavy answers from the average. So double any backslash that is NOT
# already part of a valid escape. Valid pairs are matched first and left alone
# (otherwise already-correct JSON such as \\( would get mangled).
_ESCAPE_OR_BACKSLASH = re.compile(r'\\(?:["\\/bfnrt]|u[0-9a-fA-F]{4})|\\')


def _fix_backslashes(s):
    return _ESCAPE_OR_BACKSLASH.sub(lambda m: m.group(0) if len(m.group(0)) > 1 else "\\\\", s)


def parse_json(text, fallback):
    """Pull the first JSON object/array out of an LLM reply, tolerating code
    fences, chatter around it, and LaTeX backslashes. Returns `fallback` if
    nothing parses."""
    starts = [i for i in (text.find("{"), text.find("[")) if i != -1]
    if not starts:
        return fallback
    start = min(starts)
    closer = "}" if text[start] == "{" else "]"
    end = text.rfind(closer)
    if end <= start:
        return fallback
    try:
        return json.loads(_fix_backslashes(text[start:end + 1]))
    except Exception:
        return fallback


def faithfulness(answer_text, hits, llm):
    ctx = "\n\n".join(f"[{i}] {h['text']}" for i, h in enumerate(hits, 1))
    raw = (FAITHFULNESS | llm | StrOutputParser()).invoke({"context": ctx, "answer": answer_text})
    data = parse_json(raw, {"claims": [], "score": None})
    if not isinstance(data, dict):
        data = {"claims": [], "score": None}
    claims = data.get("claims") or []
    score = data.get("score")
    if score is None and claims:
        score = sum(bool(c.get("supported")) for c in claims) / len(claims)
    return {"score": score, "claims": claims}


def cohens_kappa(a, b):
    """Agreement corrected for chance. Both lists are 0/1 labels."""
    n = len(a)
    if n == 0:
        return None
    po = sum(x == y for x, y in zip(a, b)) / n
    pa1, pb1 = sum(a) / n, sum(b) / n
    pe = pa1 * pb1 + (1 - pa1) * (1 - pb1)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def validate_judge(records, threshold=0.7):
    """records: [{"judge_score": float, "human_label": 0|1}, ...]
    human_label = 1 if you judged the answer faithful.
    """
    pairs = [(r, 1 if (r["judge_score"] or 0) >= threshold else 0) for r in records
             if r.get("judge_score") is not None and r.get("human_label") is not None]
    if not pairs:
        return {"n": 0, "agreement": None, "kappa": None}
    human = [r["human_label"] for r, _ in pairs]
    judge = [j for _, j in pairs]
    return {
        "n": len(pairs),
        "agreement": sum(h == j for h, j in zip(human, judge)) / len(pairs),
        "kappa": cohens_kappa(human, judge),
        "threshold": threshold,
    }
