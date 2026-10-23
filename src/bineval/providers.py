from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from typing import Any, Protocol

from bineval.heuristic import HeuristicJudge
from bineval.types import BinaryQuestion, Verdict


class Provider(Protocol):
    def complete(self, prompt: str) -> str: ...


def parse_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    if fence:
        text = fence.group(1)
    try:
        value = json.loads(text)
        if isinstance(value, dict):
            return value
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        value = json.loads(text[start : end + 1])
        if isinstance(value, dict):
            return value
    raise ValueError(f"model did not return a JSON object: {text[:240]!r}")


class OpenAICompatibleProvider:
    """Minimal OpenAI-compatible chat client (no extra dependencies)."""

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.0,
        timeout: float = 60.0,
    ) -> None:
        self.model = model or os.environ.get("BINEVAL_MODEL", "gpt-4o-mini")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.base_url = (
            base_url
            or os.environ.get("OPENAI_BASE_URL")
            or "https://api.openai.com/v1"
        ).rstrip("/")
        self.temperature = temperature
        self.timeout = timeout

    def complete(self, prompt: str) -> str:
        if not self.api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Use --provider heuristic for an offline run."
            )
        payload = json.dumps(
            {
                "model": self.model,
                "temperature": self.temperature,
                "messages": [{"role": "user", "content": prompt}],
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"LLM HTTP {exc.code}: {detail[:400]}") from exc
        return body["choices"][0]["message"]["content"]


class HeuristicProvider:
    """Offline provider: structured JSON for generation/update, heuristic verdicts for eval."""

    def __init__(self, judge: HeuristicJudge | None = None) -> None:
        self.judge = judge or HeuristicJudge()
        self._eval_context: tuple[str, str] | None = None

    def bind_example(self, input_text: str, output_text: str) -> None:
        self._eval_context = (input_text, output_text)

    def complete(self, prompt: str) -> str:
        if '"answer"' in prompt and "Question" in prompt:
            return self._answer_eval(prompt)
        if "decompose a generation task" in prompt.lower() or "Step 1 – Summarize" in prompt:
            return self._fallback_questions()
        if "extract generalized lessons" in prompt.lower() or "Failing questions" in prompt:
            return self._lessons_from_failures(prompt)
        if "evaluator disagreements" in prompt.lower():
            return json.dumps(
                {
                    "lessons": [
                        {
                            "text": "Do not treat omission of a source detail as a contradiction.",
                            "dimension": "consistency",
                        }
                    ]
                }
            )
        if "rewrite a prompt" in prompt.lower():
            return self._rewrite(prompt)
        return json.dumps({"lessons": []})

    def _answer_eval(self, prompt: str) -> str:
        question = BinaryQuestion(id="adhoc", text="", dimension="unknown")
        q_match = re.search(
            r"Question \(([^,]+), dimension=([^)]+)\):\n(.*?)(?:\n\nViolation|\nViolation)",
            prompt,
            re.DOTALL,
        )
        if q_match:
            question = BinaryQuestion(
                id=q_match.group(1).strip(),
                text=q_match.group(3).strip(),
                dimension=q_match.group(2).strip(),
            )
        input_text, output_text = "", ""
        if self._eval_context:
            input_text, output_text = self._eval_context
        else:
            in_match = re.search(r"Input \(x\):\n(.*?)\n\nOutput \(y\):\n(.*?)\n\nQuestion", prompt, re.DOTALL)
            if in_match:
                input_text, output_text = in_match.group(1).strip(), in_match.group(2).strip()
        verdict = self.judge.answer(input_text, output_text, question)
        return json.dumps({"answer": verdict.answer, "explanation": verdict.explanation})

    def _fallback_questions(self) -> str:
        from bineval.banks import SUMMEVAL_QUESTIONS, SUMMEVAL_REQUIREMENTS

        return json.dumps(
            {
                "requirements": [
                    {"id": r.id, "text": r.text, "dimension": r.dimension}
                    for r in SUMMEVAL_REQUIREMENTS
                ],
                "questions": [q.to_dict() for q in SUMMEVAL_QUESTIONS],
            }
        )

    def _lessons_from_failures(self, prompt: str) -> str:
        lessons = []
        if "hallucin" in prompt.lower() or "fabricat" in prompt.lower() or "q-con-3" in prompt.lower():
            lessons.append(
                {
                    "text": "State only facts that appear in the source; never invent URLs, quotes, or extra events.",
                    "dimension": "consistency",
                }
            )
        if "grammatical" in prompt.lower() or "q-flu-1" in prompt.lower():
            lessons.append(
                {
                    "text": "Write complete grammatical sentences with correct subject-verb agreement.",
                    "dimension": "fluency",
                }
            )
        if "main topic" in prompt.lower() or "q-rel-1" in prompt.lower() or "q-coh-6" in prompt.lower():
            lessons.append(
                {
                    "text": "Lead with the central event of the source before any background detail.",
                    "dimension": "relevance",
                }
            )
        if not lessons:
            lessons.append(
                {
                    "text": "Satisfy every binary evaluation question; prefer supported, focused, grammatical output.",
                    "dimension": "overall",
                }
            )
        return json.dumps({"lessons": lessons})

    def _rewrite(self, prompt: str) -> str:
        lesson_match = re.search(r"Lesson:\n(.*)\n\nIdentify", prompt, re.DOTALL)
        current_match = re.search(r"Current prompt:\n(.*)\n\nLesson:", prompt, re.DOTALL)
        lesson = lesson_match.group(1).strip() if lesson_match else "Follow all evaluation criteria."
        current = current_match.group(1).strip() if current_match else ""
        addition = f" Additionally: {lesson.rstrip('.')}."
        if current:
            return json.dumps({"old": current, "new": current.rstrip() + addition})
        return json.dumps({"old": "", "new": lesson})


def get_provider(name: str) -> HeuristicProvider | OpenAICompatibleProvider:
    key = name.lower().strip()
    if key in {"heuristic", "offline", "mock", "local"}:
        return HeuristicProvider()
    if key in {"openai", "llm", "api"}:
        return OpenAICompatibleProvider()
    raise ValueError(f"unknown provider {name!r}; use heuristic or openai")
