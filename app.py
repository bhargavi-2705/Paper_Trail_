"""Streamlit demo: ask, inspect sources, see coverage, take a quiz.

    streamlit run app.py
"""
import json
from pathlib import Path

import streamlit as st

from src.config import load_config
from src.llm import get_llm
from src.quiz import generate_quiz, grade
from src.retrieve import Retriever

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="Notes RAG KB", layout="wide")


@st.cache_resource
def boot(agent_enabled, hybrid, rerank):
    cfg = load_config()
    cfg["agent"]["enabled"] = agent_enabled
    cfg["retrieval"]["hybrid"] = hybrid
    cfg["retrieval"]["rerank"] = rerank
    return cfg, Retriever(cfg), get_llm(cfg)


st.sidebar.header("Pipeline")
agent_enabled = st.sidebar.checkbox("Agent loop (decompose + critic)", value=False)
hybrid = st.sidebar.checkbox("Hybrid retrieval (dense + BM25)", value=True)
rerank = st.sidebar.checkbox("CrossEncoder rerank", value=True)
st.sidebar.caption("Toggles mirror the ablation flags, so the demo shows the same "
                   "configurations the results table reports.")

try:
    cfg, retriever, llm = boot(agent_enabled, hybrid, rerank)
except Exception as e:
    st.error(f"Could not load the index: {e}\n\nRun `python -m src.build_index` first.")
    st.stop()

tab_ask, tab_cov, tab_quiz, tab_res = st.tabs(["Ask", "Coverage", "Quiz", "Results"])

with tab_ask:
    q = st.text_input("Question", placeholder="What problem does LoRA solve?")
    if q:
        from src.answer import answer
        with st.spinner("retrieving + answering..."):
            out = answer(q, cfg, retriever, llm)
        st.markdown(out["answer"])
        if out.get("trace"):
            st.caption(" → ".join(out["trace"]))
        st.subheader("Sources")
        for i, h in enumerate(out["sources"], 1):
            score = h.get("rerank_score", h.get("score"))
            with st.expander(f"[{i}] {h['doc']} p.{h['page']} (score {score:.3f})"):
                st.write(h["text"])

with tab_cov:
    p = ROOT / "eval" / "results" / "coverage.json"
    if not p.exists():
        st.info("Run `python scripts/run_coverage.py` to generate the report.")
    else:
        rep = json.loads(p.read_text())
        st.caption(f"Gap threshold {rep['threshold']:.3f}, calibrated against "
                   f"{rep['calibration']['n_labelled']} known-present/absent topics "
                   f"(accuracy {rep['calibration']['accuracy']:.2f}).")
        rows = sorted(rep["topics"], key=lambda r: r.get("score_used", r["best_score"]))
        st.dataframe([{"topic": r["topic"], "score": round(r.get("score_used", r["best_score"]), 3),
                       "docs": r["n_docs"], "gap": "yes" if r["flagged_gap"] else "",
                       "top sources": ", ".join(r["top_sources"][:2])} for r in rows],
                     use_container_width=True)

with tab_quiz:
    topic = st.text_input("Quiz topic", value="attention")
    if st.button("Generate questions") and topic:
        st.session_state.quiz = generate_quiz(topic, retriever, llm, n=3)
    for i, item in enumerate(st.session_state.get("quiz", [])):
        st.markdown(f"**Q{i+1}.** {item['question']}")
        ans = st.text_area("Your answer", key=f"a{i}")
        if st.button("Grade", key=f"g{i}") and ans.strip():
            g = grade(item, ans, llm)
            st.write(f"{'Correct' if g.get('correct') else 'Not quite'} — {g.get('feedback')}")
            st.caption(f"source: {item['source']['doc']} p.{item['source']['page']}")

with tab_res:
    p = ROOT / "eval" / "results" / "ablations.md"
    st.markdown(p.read_text() if p.exists()
                else "Run `python scripts/ablations.py` to produce the comparison table.")
