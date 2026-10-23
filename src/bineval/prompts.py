"""Task-agnostic meta-prompts from BinEval §3.1–3.4."""

QUESTION_GENERATION = """You decompose a generation task into atomic binary evaluation questions.

Task prompt T:
{task_prompt}

Dimensions to cover: {dimensions}

Work in two steps.

Step 1 – Summarize.
Extract an explicit set of requirements. Each requirement is one distinct evaluation
criterion (content that must appear, a format constraint, or a quality property).

Step 2 – Decompose.
For each requirement, write one or more binary questions such that:
- answering YES means the output satisfies the requirement
- answering NO means a violation
- compound requirements become separate questions
- each question includes a short violation example

Return JSON only:
{{
  "requirements": [
    {{"id": "R1", "text": "...", "dimension": "consistency"}}
  ],
  "questions": [
    {{
      "id": "Q1",
      "text": "Does ...?",
      "dimension": "consistency",
      "requirement_id": "R1",
      "violation_example": "..."
    }}
  ]
}}
"""

EVALUATION = """You are a strict BinEval evaluator. Answer ONE binary question about an
input/output pair. Answer 1 only when the evidence clearly supports yes.
When evidence is absent, weak, or ambiguous, answer 0.

Evaluator guidance:
{evaluator_prompt}

Input (x):
{input_text}

Output (y):
{output_text}

Question ({question_id}, dimension={dimension}):
{question_text}

Violation example (clarifies the negative case):
{violation_example}

Return JSON only:
{{"answer": 0 or 1, "explanation": "one or two sentences citing evidence from the output"}}
"""

DEFAULT_EVALUATOR_PROMPT = (
    "Judge each question independently. Do not give partial credit. "
    "Do not conflate omission with contradiction: missing a source detail is "
    "not a factual error unless the output states something unsupported. "
    "Implicit logical connections between sentences are acceptable. "
    "Do not penalize brevity when the core claim is clear."
)

NOTE_TAKER_FAILURES = """You extract generalized lessons from BinEval failures.

Task context:
{context}

Failing questions and explanations:
{failures}

Write 1-4 reusable lessons that would help a prompt avoid these failures.
Do not mention this specific example's names or numbers.

Return JSON only:
{{"lessons": [{{"text": "...", "dimension": "..."}}]}}
"""

NOTE_TAKER_DISAGREEMENT = """You extract lessons from evaluator disagreements.

A source evaluator is the reference. A target evaluator disagreed on some
binary questions. Diagnose the target's systematic error.

Input:
{input_text}

Output:
{output_text}

Disagreements (source_answer vs target_answer):
{disagreements}

Return JSON only:
{{"lessons": [{{"text": "...", "dimension": "..."}}]}}
"""

PROMPT_UPDATER = """You rewrite a prompt by incorporating one lesson.

Current prompt:
{prompt}

Lesson:
{lesson}

Identify a substring to replace and the revised substring that absorbs the lesson.
Keep the rest of the prompt intact. If no substring fits, append a short instruction.

Return JSON only:
{{"old": "...", "new": "..."}}
"""
