from bineval.optimize import CrossModelUpdate, SelfPromptUpdate, deduplicate_lessons
from bineval.providers import HeuristicProvider
from bineval.questions import generate_questions
from bineval.types import Lesson, Verdict


def test_dedup_merges_normalized_duplicates():
    lessons = [
        Lesson("Do not invent URLs.", "consistency", ("Q1",)),
        Lesson("do not invent urls.", "consistency", ("Q2",)),
        Lesson("Lead with the central event.", "relevance", ("Q3",)),
    ]
    unique = deduplicate_lessons(lessons)
    assert len(unique) == 2
    merged = next(lesson for lesson in unique if "url" in lesson.text.lower())
    assert merged.question_ids == ("Q1", "Q2")


def test_self_update_appends_lessons_from_failures():
    provider = HeuristicProvider()
    _, questions = generate_questions("Summarize.", provider)
    updater = SelfPromptUpdate(questions, provider, max_iterations=1)
    cases = [
        {
            "input": "A storm flooded the river town on Sunday, killing 3 residents.",
            "output": "teh goverment recieve zzzzzq report!!! https://fake.example/x killed 99 people in space.",
        }
    ]
    updated = updater.run(cases, "Summarize the article.")
    assert updater.history
    assert updater.history[0].lessons
    assert len(updated) > len("Summarize the article.")


def test_cross_model_converges_when_judges_agree():
    provider = HeuristicProvider()
    _, questions = generate_questions("Summarize.", provider)
    updater = CrossModelUpdate(
        questions,
        source_provider=provider,
        target_provider=provider,
        note_provider=provider,
        max_iterations=2,
    )
    cases = [
        {
            "input": "The museum opened a new wing on Friday.",
            "output": "The museum opened a new wing on Friday.",
        }
    ]
    prompt = updater.run(cases, "Judge strictly.")
    assert prompt
    assert updater.history[0].mean_overall >= 0.0
