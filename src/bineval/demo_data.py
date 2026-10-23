TASK_PROMPT = """Summarize the news article in a short paragraph.
The summary must be coherent, factually consistent with the source, fluent,
and focused on the most newsworthy information. Do not invent facts, URLs,
quotes, or numbers. Do not dwell on trivia."""

GENERATION_PROMPT = "Summarize the article in a few sentences."

SOURCE = (
    "The Pentagon said a Russian Su-27 fighter intercepted a U.S. RC-135 "
    "reconnaissance aircraft over the Baltic Sea on Tuesday. U.S. officials "
    "described the intercept as unsafe because the Russian jet came within "
    "20 feet of the American plane. Moscow said its aircraft was identifying "
    "an unknown aircraft in international airspace and denied any unsafe "
    "maneuver. No injuries were reported and both planes returned to base."
)

CASES = [
    {
        "id": "faithful",
        "input": SOURCE,
        "output": (
            "A Russian Su-27 intercepted a U.S. RC-135 over the Baltic Sea on Tuesday. "
            "U.S. officials called the intercept unsafe after the jet came within 20 feet "
            "of the American plane, while Moscow said it was identifying an unknown aircraft "
            "and denied any unsafe maneuver. No injuries were reported."
        ),
    },
    {
        "id": "hallucinated",
        "input": SOURCE,
        "output": (
            "The Pentagon posted https://pentagon.example/intercept claiming Russia fired "
            "missiles at a NATO summit in Berlin, killing 14 people on Friday."
        ),
    },
    {
        "id": "garbled",
        "input": SOURCE,
        "output": (
            "plane plane plane the the su27 go boom recieve teh goverment "
            "zzzzzq report!!!"
        ),
    },
]
