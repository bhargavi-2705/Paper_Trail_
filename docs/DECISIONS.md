# Decisions log

One entry per experiment: what changed, why, what the number did. Entries marked
**TODO** need a real run on the real corpus — fill them in rather than guessing.

---

## D0 — Baseline (established)

- **Chunking**: 1200 chars / 200 overlap via `RecursiveCharacterTextSplitter`,
  applied per page so a chunk never crosses a page boundary. Gives exact
  `(doc, page)` citations and makes gold labelling possible at page granularity.
- **Embeddings**: `bge-small-en-v1.5` (~130MB, CPU-friendly). Query prefix as the
  model card specifies.
- **Store**: Chroma, cosine.
- **Generation**: top-5 chunks → cited answer, instructed to say when the
  excerpts don't contain the answer.
- **Framework**: LangChain for loaders/splitters/store/LCEL, LangGraph for the
  agent. Eval and metrics are plain Python on purpose — these numbers end up on a
  CV, so nothing sits between me and the arithmetic.

## D1 — Rank-fusion scores cannot drive coverage analysis *(found during build)*

First version of the coverage analyser read `search()` scores. With hybrid
retrieval on, those are Reciprocal Rank Fusion scores: `1/(k+rank)` summed across
lists. They encode **rank only**. Every topic scored ~0.033 — including a topic
deliberately absent from the corpus — because the top hit always scores
`~1/(k+1)` whether it is a perfect match or the least-bad of a bad lot.

Fix: `Retriever.probe()` bypasses fusion and returns raw dense cosine (optionally
CrossEncoder-refined), both of which are absolute. Topic scores then spread
across a real range instead of collapsing to a constant.

Lesson worth stating in an interview: *rank-based fusion is the right tool for
ordering results and the wrong tool for any question of the form "is this good
enough in absolute terms".*

## D2 — Hybrid retrieval (dense + BM25 via RRF) — **TODO**

Hypothesis: BM25 helps on queries with rare exact tokens (`AdamW`, `d_k`,
`in-batch negatives`) where a small dense model is weak. Run:
`python scripts/ablations.py --only dense_only hybrid`

| variant | hit@1 | hit@5 | mrr | latency |
|---|---|---|---|---|
| dense_only | | | | |
| hybrid | | | | |

Record which queries flipped, not just the aggregate.

## D3 — CrossEncoder reranking — **TODO**

Hypothesis: the largest single gain, because the bi-encoder's job becomes recall
over 20 candidates rather than precision at 5. Cost: latency, and a second model
to load. Run `--only dense_rerank hybrid_rerank`. If the gain is under ~3 points
on a 32-question set, say plainly that it's inside the noise.

## D4 — Chunk size and page-bounded chunking — **TODO**

`chunk_600` / `chunk_2000` / `per_page`. Page-bounded chunking is a deliberate
constraint (exact citations); this measures what it costs on prose papers whose
arguments run across page breaks.

## D5 — Agent loop vs baseline chain — **TODO**

`--only baseline_chain agent_no_critic agent_full --generation`. Record
faithfulness, latency, and LLM calls per question. **A negative result here is a
good result** — "decompose + critic cost 3× the latency for +2 points of
faithfulness, inside the noise on 32 questions" is a stronger interview answer
than an unexamined agent.

## D6 — Judge validation — **TODO**

Hand-grade ≥30 answers (`python scripts/run_quiz.py --validate-judge`), then
report n, agreement and Cohen's kappa. Kappa matters: if most answers are
faithful, raw agreement looks high by chance alone.

## D7 — Coverage threshold calibration — **TODO**

`python scripts/run_coverage.py`. Report the calibrated threshold, the
calibration accuracy, and how many topics it was fitted on. With ~6 labelled
topics this is a sanity check, not a validated classifier — say so.
