"""Built-in SummEval question bank from the paper (Appendix E, Tables 9–12)."""

from __future__ import annotations

from bineval.types import BinaryQuestion, Requirement

SUMMEVAL_DIMENSIONS = ("coherence", "consistency", "fluency", "relevance")

SUMMEVAL_REQUIREMENTS = [
    Requirement("R-coh", "The summary is structured, ordered, and focused.", "coherence"),
    Requirement("R-con", "The summary is factually supported by the source.", "consistency"),
    Requirement("R-flu", "The summary is grammatical and natural.", "fluency"),
    Requirement("R-rel", "The summary covers the main, newsworthy content.", "relevance"),
]

_COHERENCE = [
    ("Q-coh-1", "Does the summary have a well-defined structure (e.g., a clear beginning, middle, and/or end) rather than appearing randomly assembled?", "a jumble of unrelated fragments with no beginning or end"),
    ("Q-coh-2", "Are the sentences in the summary arranged in a sensible and logical order?", "later events are narrated before the events that cause them"),
    ("Q-coh-3", "Does the summary avoid being a mere heap of loosely related facts or information?", "disconnected bullet-like facts with no unifying thread"),
    ("Q-coh-4", "Do the sentences in the summary flow logically from one to the next, with clear transitions or connections between them?", "abrupt topic jumps with no implicit or explicit connection"),
    ("Q-coh-5", "Does the summary maintain a unified focus on a single main topic rather than drifting across multiple unrelated subjects?", "the summary switches between unrelated stories"),
    ("Q-coh-6", "Does the summary cover the main topic of the news article?", "the summary discusses a side issue and never states the central event"),
    ("Q-coh-7", "Does the summary cover the key points of the news article?", "only a trivial detail is mentioned while the core conflict is missing"),
    ("Q-coh-8", "Does the summary present information in a clear manner that is easy to follow and understand?", "pronouns and events are so tangled that the reader must re-parse the text"),
]

_CONSISTENCY = [
    ("Q-con-1", "Are all statements in the summary entailed by or supported by the source article?", "a claim that cannot be inferred from the article"),
    ("Q-con-2", "Is the summary free of factual errors when compared to the source article?", "an actor, number, or outcome that contradicts the source"),
    ("Q-con-3", "Is the summary free of hallucinated facts (i.e., information that is fabricated and not present in the source article)?", "a URL, quote, or event invented by the summarizer"),
    ("Q-con-4", "Are all named entities (people, organizations, locations) in the summary accurately represented as they appear in the source article?", "a name swapped onto the wrong actor"),
    ("Q-con-5", "Are all numerical claims (dates, statistics, quantities, amounts) in the summary consistent with the source article?", "a count or date that does not match the article"),
    ("Q-con-6", "Are the causal relationships and event sequences described in the summary consistent with those in the source article?", "cause and effect reversed relative to the article"),
    ("Q-con-7", "Does the summary avoid misrepresenting or distorting the meaning of information from the source article?", "two parties' accounts are conflated into one"),
]

_FLUENCY = [
    ("Q-flu-1", "Is the summary free of grammatical errors?", "subject-verb disagreement or broken clauses"),
    ("Q-flu-2", "Is the summary free of spelling errors?", "obvious misspellings of common words"),
    ("Q-flu-3", "Is the summary free of punctuation errors?", "run-on punctuation or missing sentence boundaries"),
    ("Q-flu-4", "Does the summary use appropriate and natural word choices?", "nonsensical or clearly wrong lexical substitutions"),
    ("Q-flu-5", "Does the summary have well-structured sentences that are easy to follow?", "sentence fragments stacked without a predicate"),
    ("Q-flu-6", "Does the summary read smoothly and sound natural overall?", "choppy, machine-garbled phrasing"),
    ("Q-flu-7", "Is the summary easy to understand without requiring re-reading due to language issues?", "the wording forces the reader to decode the sentence"),
]

_RELEVANCE = [
    ("Q-rel-1", "Does the summary address the main topic or central event of the source article?", "the summary never mentions the central event"),
    ("Q-rel-2", "Does the summary cover at least some of the key points or important details of the source article?", "only a peripheral aside is retained"),
    ("Q-rel-3", "Is the summary free from significant redundancy, such as repeating the same point multiple times in different words?", "the same fact is restated in every sentence"),
    ("Q-rel-4", "Is the summary free from excessive trivial or unimportant details that dilute the coverage of main points?", "the summary dwells on a minor color or weather note"),
    ("Q-rel-5", "Does the summary prioritize the most newsworthy or significant information from the source rather than focusing on minor or tangential aspects?", "background flavor replaces the headline fact"),
]


def _pack(rows: list[tuple[str, str, str]], dimension: str, requirement_id: str) -> list[BinaryQuestion]:
    return [
        BinaryQuestion(
            id=qid,
            text=text,
            dimension=dimension,
            requirement_id=requirement_id,
            violation_example=violation,
        )
        for qid, text, violation in rows
    ]


SUMMEVAL_QUESTIONS: list[BinaryQuestion] = (
    _pack(_COHERENCE, "coherence", "R-coh")
    + _pack(_CONSISTENCY, "consistency", "R-con")
    + _pack(_FLUENCY, "fluency", "R-flu")
    + _pack(_RELEVANCE, "relevance", "R-rel")
)

BANKS = {
    "summeval": SUMMEVAL_QUESTIONS,
}


def load_bank(name: str) -> list[BinaryQuestion]:
    key = name.lower().strip()
    if key not in BANKS:
        known = ", ".join(sorted(BANKS))
        raise KeyError(f"unknown question bank {name!r}; available: {known}")
    return list(BANKS[key])
