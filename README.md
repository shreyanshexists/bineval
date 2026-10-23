# BinEval

Working implementation of **BinEval**, the evaluation framework from:

> Cho et al. *Ask, Don’t Judge: Binary Questions for Interpretable LLM Evaluation and Self-Improvement.* [arXiv:2606.27226](https://arxiv.org/abs/2606.27226)

Instead of one opaque Likert score, BinEval:

1. Decomposes a task prompt into atomic **yes/no questions** (grouped by dimension).
2. Answers each question independently and aggregates **per-dimension** and **overall** scores in `[0, 1]`.
3. Uses failing questions as feedback for **self prompt update** and **cross-model evaluator update**.

This repo is a runnable library plus CLI. The bundled `heuristic` provider is deterministic so the project works without an API key. Point `--provider openai` at any OpenAI-compatible endpoint for the paper’s LLM-as-evaluator setting.

## Install

```bash
cd bineval
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Quick start

```bash
bineval demo
```

Evaluate the sample summaries (faithful / hallucinated / garbled):

```bash
bineval evaluate --input examples/cases.json --bank summeval --provider heuristic
```

Decompose a task prompt, then optimize a generation prompt from failures:

```bash
bineval generate-questions --task examples/task_prompt.txt --out questions.json
bineval optimize --input examples/cases.json --prompt examples/generation_prompt.txt --out updated_prompt.txt
```

Python API:

```python
from bineval import BinEval
from bineval.banks import load_bank
from bineval.providers import get_provider

evaluator = BinEval(load_bank("summeval"), get_provider("heuristic"))
result = evaluator.evaluate(source_article, summary)
print(result.overall, result.scores, result.failing())
```

## Scoring (paper §3.2)

For evaluator \(E\), input \(x\), output \(y\), and question \(q_i\):

\[
f_E(x,y,q_i)\in\{0,1\}
\]

Dimension score \(S_d\) is the mean of answers in that dimension. Overall score \(S\) is the mean over all \(N\) questions. Optional affine map \(S' = S\cdot(b-a)+a\) (default Likert 1–5).

The built-in `summeval` bank is Appendix E of the paper: 8 coherence, 7 consistency, 7 fluency, and 5 relevance questions.

## LLM backend

```bash
export OPENAI_API_KEY=...
# optional: export OPENAI_BASE_URL=https://api.openai.com/v1
# optional: export BINEVAL_MODEL=gpt-4o-mini
bineval evaluate --input examples/cases.json --provider openai
```

Temperature defaults to `0`, matching the paper.

## Tests

```bash
pytest
```

## Layout

```
src/bineval/     library + CLI
examples/        sample task, cases, API snippet
tests/           scoring, evaluation, optimization, CLI
```

## License

MIT. The method and SummEval question wording follow the paper cited above.
