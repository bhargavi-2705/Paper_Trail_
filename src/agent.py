"""LangGraph agent: decompose -> retrieve -> critic -> (re-query) -> synthesize.

Each stage is a config flag so ablations.py can report what the agent actually
buys over the baseline chain -- including when the answer is "nothing".
"""
import json
import operator
from typing import Annotated, TypedDict

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, StateGraph

from .config import load_config
from .judge import parse_json as _parse_json
from .llm import get_llm
from .retrieve import Retriever

DECOMPOSE = ChatPromptTemplate.from_messages([
    ("system",
     "Break the user's question into at most {max_sub} standalone sub-questions that a "
     "search engine over ML papers could answer. If the question is already simple, return "
     "it unchanged as a single item. Reply with a JSON array of strings and nothing else."),
    ("human", "{question}"),
])

CRITIC = ChatPromptTemplate.from_messages([
    ("system",
     "You judge whether retrieved excerpts are sufficient to answer a question. "
     'Reply with JSON only: {{"sufficient": true|false, "missing": "<what is missing, or empty>", '
     '"followup_query": "<a better search query, or empty>"}}'),
    ("human", "Question: {question}\n\nExcerpts:\n{context}"),
])

SYNTH = ChatPromptTemplate.from_messages([
    ("system",
     "You answer questions using ONLY the numbered excerpts provided. Cite excerpts you use "
     "like [1], [2]. If the excerpts do not contain the answer, say so plainly instead of guessing."),
    ("human", "Excerpts:\n\n{context}\n\nQuestion: {question}"),
])


def format_context(hits):
    return "\n\n".join(
        f"[{i}] ({h['doc']}, p.{h['page']})\n{h['text']}" for i, h in enumerate(hits, 1)
    )


def _merge(a, b):
    """Dedupe hits by (doc, page, first 60 chars) while preserving order."""
    seen, out = set(), []
    for h in list(a) + list(b):
        key = (h["doc"], h["page"], h["text"][:60])
        if key not in seen:
            seen.add(key)
            out.append(h)
    return out


class AgentState(TypedDict, total=False):
    question: str
    subquestions: list
    hits: Annotated[list, _merge]
    critique: dict
    requeries: int
    answer: str
    trace: Annotated[list, operator.add]   # reducer: nodes append, not overwrite




def build_graph(cfg=None, retriever=None, llm=None):
    cfg = cfg or load_config()
    acfg = cfg["agent"]
    retriever = retriever or Retriever(cfg)
    llm = llm or get_llm(cfg)

    def decompose(state):
        if not acfg.get("decompose", True):
            return {"subquestions": [state["question"]], "trace": ["decompose: skipped"]}
        raw = (DECOMPOSE | llm | StrOutputParser()).invoke(
            {"question": state["question"], "max_sub": acfg.get("max_subquestions", 3)}
        )
        subs = _parse_json(raw, [state["question"]])
        if not isinstance(subs, list) or not subs:
            subs = [state["question"]]
        subs = [str(s) for s in subs][: acfg.get("max_subquestions", 3)]
        return {"subquestions": subs, "trace": [f"decompose: {len(subs)} sub-question(s)"]}

    def retrieve(state):
        hits = []
        for q in state.get("subquestions") or [state["question"]]:
            hits.extend(retriever.search(q))
        return {"hits": hits, "trace": [f"retrieve: {len(hits)} hits"]}

    def critic(state):
        if not acfg.get("critic", True):
            return {"critique": {"sufficient": True}, "trace": ["critic: skipped"]}
        raw = (CRITIC | llm | StrOutputParser()).invoke(
            {"question": state["question"], "context": format_context(state["hits"])}
        )
        c = _parse_json(raw, {"sufficient": True})
        if not isinstance(c, dict):
            c = {"sufficient": True}
        return {"critique": c, "trace": [f"critic: sufficient={c.get('sufficient')}"]}

    def requery(state):
        q = (state.get("critique") or {}).get("followup_query") or state["question"]
        return {
            "hits": retriever.search(q),
            "requeries": state.get("requeries", 0) + 1,
            "trace": [f"requery: {q[:60]}"],
        }

    def synthesize(state):
        hits = state["hits"][: cfg["retrieval"]["top_k"] * 2]
        text = (SYNTH | llm | StrOutputParser()).invoke(
            {"question": state["question"], "context": format_context(hits)}
        )
        return {"answer": text, "trace": ["synthesize"]}

    def route(state):
        c = state.get("critique") or {}
        if c.get("sufficient", True):
            return "synthesize"
        if state.get("requeries", 0) >= acfg.get("max_requeries", 1):
            return "synthesize"
        return "requery"

    g = StateGraph(AgentState)
    g.add_node("decompose", decompose)
    g.add_node("retrieve", retrieve)
    g.add_node("critic", critic)
    g.add_node("requery", requery)
    g.add_node("synthesize", synthesize)
    g.set_entry_point("decompose")
    g.add_edge("decompose", "retrieve")
    g.add_edge("retrieve", "critic")
    g.add_conditional_edges("critic", route, {"requery": "requery", "synthesize": "synthesize"})
    g.add_edge("requery", "critic")
    g.add_edge("synthesize", END)
    return g.compile()


def agent_answer(question, cfg=None, retriever=None, llm=None, graph=None):
    cfg = cfg or load_config()
    graph = graph or build_graph(cfg, retriever, llm)
    out = graph.invoke({"question": question, "hits": [], "requeries": 0, "trace": []})
    # return only the excerpts the synthesizer saw, so the faithfulness judge
    # checks the answer against the same evidence the answer was written from
    used = out["hits"][: cfg["retrieval"]["top_k"] * 2]
    return {"answer": out["answer"], "sources": used, "trace": out.get("trace", [])}
