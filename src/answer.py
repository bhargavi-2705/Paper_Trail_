"""Entry point for answering. Routes to the baseline LCEL chain or the
LangGraph agent depending on config (agent.enabled).

    python -m src.answer "what problem does LoRA solve?"
"""
import sys

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from .config import load_config
from .llm import get_llm
from .retrieve import Retriever

SYSTEM = (
    "You answer questions using ONLY the numbered excerpts provided. Cite excerpts you use "
    "like [1], [2]. If the excerpts do not contain the answer, say so plainly instead of guessing."
)
PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM),
    ("human", "Excerpts:\n\n{context}\n\nQuestion: {question}"),
])


def format_context(hits):
    return "\n\n".join(f"[{i}] ({h['doc']}, p.{h['page']})\n{h['text']}" for i, h in enumerate(hits, 1))


def baseline_answer(question, cfg=None, retriever=None, llm=None):
    cfg = cfg or load_config()
    retriever = retriever or Retriever(cfg)
    hits = retriever.search(question)
    chain = PROMPT | (llm or get_llm(cfg)) | StrOutputParser()
    return {"answer": chain.invoke({"context": format_context(hits), "question": question}),
            "sources": hits, "trace": ["baseline"]}


def answer(question, cfg=None, retriever=None, llm=None):
    cfg = cfg or load_config()
    if cfg.get("agent", {}).get("enabled"):
        from .agent import agent_answer
        return agent_answer(question, cfg, retriever, llm)
    return baseline_answer(question, cfg, retriever, llm)


if __name__ == "__main__":
    out = answer(" ".join(sys.argv[1:]))
    print(out["answer"], "\n\nSources:")
    for i, h in enumerate(out["sources"], 1):
        print(f"  [{i}] {h['doc']} p.{h['page']}")
