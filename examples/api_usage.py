"""Minimal Python API usage for BinEval."""

from bineval import BinEval
from bineval.banks import load_bank
from bineval.providers import get_provider

source = (
    "The city council voted 7-2 on Monday to expand overnight bus service. "
    "The change starts in March and adds two routes in the east district."
)
summary = (
    "The city council voted 7-2 on Monday to expand overnight bus service, "
    "adding two east-district routes beginning in March."
)

evaluator = BinEval(load_bank("summeval"), get_provider("heuristic"))
result = evaluator.evaluate(source, summary)
print(f"overall={result.overall:.3f} scaled={result.scaled:.2f}")
for name, score in sorted(result.scores.items()):
    print(f"  {name}: {score:.3f}")
for verdict in result.failing():
    print(f"fail {verdict.question_id}: {verdict.explanation}")
