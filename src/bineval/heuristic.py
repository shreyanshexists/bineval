from __future__ import annotations

import re
from collections import Counter

from bineval.types import BinaryQuestion, Verdict

_STOP = {
    "a", "an", "the", "and", "or", "of", "to", "in", "on", "for", "with", "by",
    "from", "at", "as", "is", "are", "was", "were", "be", "been", "that", "this",
    "it", "its", "they", "their", "he", "she", "his", "her", "but", "not",
}

_COMMON = {
    "the", "and", "for", "that", "with", "from", "this", "were", "have", "has",
    "been", "after", "before", "during", "said", "says", "according", "officials",
    "people", "years", "year", "today", "monday", "summary", "article", "about",
}

_TYPO_MARKERS = ("teh", "recieve", "seperate", "occured", "definately", "goverment")


def tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9']+", text.lower())


def content_tokens(text: str) -> list[str]:
    return [tok for tok in tokenize(text) if tok not in _STOP and len(tok) > 2]


def sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p for p in parts if p]


def numbers(text: str) -> set[str]:
    return set(re.findall(r"\b\d+(?:\.\d+)?\b", text))


def entities(text: str) -> set[str]:
    return {m.group(0) for m in re.finditer(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", text)}


def overlap(a: list[str], b: list[str]) -> float:
    if not a:
        return 0.0
    bset = set(b)
    return sum(1 for tok in a if tok in bset) / len(a)


class HeuristicJudge:
    """Deterministic yes/no judge so BinEval runs without an API key.

    It is not a substitute for an LLM evaluator. It implements cheap probes
    that discriminate the bundled examples and unit tests.
    """

    def answer(self, source: str, output: str, question: BinaryQuestion) -> Verdict:
        yes, why = self._decide(source, output, question)
        return Verdict(
            question_id=question.id,
            answer=1 if yes else 0,
            explanation=why,
            dimension=question.dimension,
        )

    def _decide(self, source: str, output: str, question: BinaryQuestion) -> tuple[bool, str]:
        q = question.text.lower()
        src_toks = content_tokens(source)
        out_toks = content_tokens(output)
        src_set = set(src_toks)
        out_sents = sentences(output)
        cov = overlap(out_toks, src_toks) if out_toks else 0.0
        src_cov = overlap(src_toks[:40], out_toks)

        invented = [tok for tok in out_toks if tok not in src_set and tok not in _COMMON and len(tok) > 4]
        extra_numbers = numbers(output) - numbers(source)
        extra_entities = {e for e in entities(output) if e.lower() not in source.lower()}
        has_url = bool(re.search(r"https?://", output)) and "http" not in source.lower()
        repeats = any(c >= 3 for c in Counter(out_sents).values()) if out_sents else False
        if not repeats and out_toks:
            window = [" ".join(out_toks[i : i + 4]) for i in range(max(0, len(out_toks) - 3))]
            repeats = any(c >= 3 for c in Counter(window).values())

        grammar_broken = bool(
            re.search(r"\b(is are|are is|the the|a a)\b", output.lower())
            or re.search(r"\b(he|she|they|it)\s+(go|do|have)\s", output.lower())
            or output.count("  ") > 2
        )
        spelling = any(m in output.lower() for m in _TYPO_MARKERS) or bool(
            re.search(r"\b[A-Za-z]{12,}[0-9]+\b", output)
        )
        punct = bool(re.search(r"[!?]{2,}|,{3,}|\.\.\.\.|;;", output) or (output and output[-1] not in ".!?\""))
        garbled = bool(re.search(r"\b[bcdfghjklmnpqrstvwxyz]{6,}\b", output.lower()))
        short = len(out_sents) <= 1 and len(out_toks) < 8
        heap = len(out_sents) >= 3 and src_cov < 0.15
        ordered = self._source_order_ok(source, output)
        focused = src_cov >= 0.12 or overlap(content_tokens(" ".join(sentences(source)[:2])), out_toks) >= 0.25
        main_covered = overlap(content_tokens(" ".join(sentences(source)[:2]) or source[:240]), out_toks) >= 0.2
        key_points = src_cov >= 0.18 or (focused and len(out_toks) >= 12)
        trivial = focused is False and len(out_toks) >= 8
        fluent = not grammar_broken and not spelling and not garbled

        if "well-defined structure" in q or "beginning, middle" in q:
            ok = len(out_sents) >= 1 and not heap and focused
            return ok, "structure follows a readable sequence" if ok else "output lacks a clear focused structure"
        if "logical order" in q and "arranged" in q:
            ok = ordered and not heap
            return ok, "sentence order tracks the source" if ok else "events appear out of source order"
        if "heap of loosely related" in q:
            return (not heap, "facts are tied to one topic" if not heap else "facts are loosely piled")
        if "flow logically" in q or "transitions" in q:
            ok = len(out_sents) <= 1 or (ordered and focused)
            return ok, "adjacent sentences stay on-topic" if ok else "sentence flow is weakly connected"
        if "unified focus" in q or "drifting" in q:
            ok = focused and not heap
            return ok, "stays on one topic" if ok else "drifts off the source topic"
        if "cover the main topic" in q:
            return (main_covered, "central event is mentioned" if main_covered else "main topic is missing")
        if "key points" in q:
            return (key_points, "several source content words appear" if key_points else "key points are missing")
        if "clear manner" in q:
            ok = fluent and focused
            return ok, "wording is followable" if ok else "wording is hard to follow"
        if "entailed" in q or "supported by the source" in q:
            return (cov >= 0.55 and not extra_numbers and not has_url, "most claims overlap the source" if cov >= 0.55 else "statements are not well supported")
        if "factual errors" in q:
            return (not extra_numbers and len(extra_entities) <= 1, "no clear numeric/entity clash" if not extra_numbers else f"unsupported numbers {sorted(extra_numbers)}")
        if "hallucin" in q or "fabricat" in q:
            return (not has_url and len(invented) <= 2, "no fabricated extras detected" if not has_url and len(invented) <= 2 else f"possible hallucinations: {invented[:5]} / url={has_url}")
        if "named entities" in q:
            return (len(extra_entities) == 0, "entities match the source" if not extra_entities else f"unseen entities {sorted(extra_entities)}")
        if "numerical" in q:
            return (not extra_numbers, "numbers match the source" if not extra_numbers else f"new numbers {sorted(extra_numbers)}")
        if "causal" in q or "event sequences" in q:
            return (ordered, "event order is consistent" if ordered else "event order looks reversed")
        if "misrepresenting" in q or "distort" in q:
            return (cov >= 0.5 and len(invented) <= 2, "meaning appears undistorted" if cov >= 0.5 else "meaning may be distorted")
        if "grammatical" in q:
            return (not grammar_broken and not garbled, "no obvious grammar breaks" if not grammar_broken else "grammar looks broken")
        if "spelling" in q:
            return (not spelling, "no obvious spelling issues" if not spelling else "spelling issues found")
        if "punctuation" in q:
            return (not punct, "punctuation looks fine" if not punct else "punctuation looks off")
        if "word choices" in q:
            return (not garbled and fluent, "word choice looks natural" if not garbled else "word choice looks unnatural")
        if "well-structured sentences" in q:
            ok = fluent and not garbled
            return ok, "sentences are structured" if ok else "sentences are poorly structured"
        if "read smoothly" in q or "sound natural" in q:
            return (fluent and not heap, "reads smoothly" if fluent else "does not read smoothly")
        if "re-reading" in q or "language issues" in q:
            return (fluent, "understandable without re-reading" if fluent else "language issues force re-reading")
        if "main topic or central event" in q:
            return (main_covered, "addresses the central event" if main_covered else "misses the central event")
        if "key points or important" in q:
            return (key_points, "includes important details" if key_points else "important details missing")
        if "redundancy" in q or "repeating" in q:
            return (not repeats, "little redundancy" if not repeats else "the same point is repeated")
        if "trivial or unimportant" in q:
            return (not trivial, "not dominated by trivia" if not trivial else "trivial details crowd out the news")
        if "newsworthy" in q or "tangential" in q:
            return (main_covered and not trivial, "leads with newsworthy content" if main_covered else "focuses on tangential content")
        return (cov >= 0.45 and fluent, "generic overlap/fluency check")

    def _source_order_ok(self, source: str, output: str) -> bool:
        src_sents = sentences(source)
        out_sents = sentences(output)
        if len(out_sents) <= 1 or len(src_sents) <= 1:
            return True
        positions = []
        for sent in out_sents:
            toks = content_tokens(sent)
            if not toks:
                continue
            best_i, best = 0, -1.0
            for i, src in enumerate(src_sents):
                score = overlap(toks, content_tokens(src))
                if score > best:
                    best, best_i = score, i
            if best >= 0.2:
                positions.append(best_i)
        if len(positions) < 2:
            return True
        inversions = sum(1 for a, b in zip(positions, positions[1:]) if a > b)
        return inversions <= 1
