import json
from pathlib import Path

from bineval.cli import main


def test_demo_runs(capsys):
    assert main(["demo"]) == 0
    out = capsys.readouterr().out
    assert "overall=" in out
    assert "faithful" in out
    assert "Ranking" in out


def test_evaluate_cli_writes_json(tmp_path: Path, capsys):
    cases = tmp_path / "cases.json"
    cases.write_text(
        json.dumps(
            [
                {
                    "id": "ok",
                    "input": "The board approved a 2 percent tax cut on Tuesday.",
                    "output": "The board approved a 2 percent tax cut on Tuesday.",
                }
            ]
        )
    )
    out = tmp_path / "scores.json"
    assert main(["evaluate", "--input", str(cases), "--out", str(out), "--provider", "heuristic"]) == 0
    payload = json.loads(out.read_text())
    assert payload[0]["overall"] > 0.5
    assert "overall=" in capsys.readouterr().out


def test_generate_and_optimize_cli(tmp_path: Path):
    task = tmp_path / "task.txt"
    task.write_text("Summarize the news article faithfully.")
    questions = tmp_path / "q.json"
    assert main(
        [
            "generate-questions",
            "--task",
            str(task),
            "--out",
            str(questions),
            "--provider",
            "heuristic",
        ]
    ) == 0
    bank = json.loads(questions.read_text())
    assert len(bank["questions"]) >= 5

    cases = tmp_path / "cases.json"
    cases.write_text(
        json.dumps(
            [
                {
                    "id": "bad",
                    "input": "Farmers harvested wheat after the drought ended in June.",
                    "output": "teh goverment recieve https://x.example missiles in Berlin.",
                }
            ]
        )
    )
    prompt = tmp_path / "p.txt"
    prompt.write_text("Summarize the article.")
    updated = tmp_path / "p2.txt"
    assert (
        main(
            [
                "optimize",
                "--input",
                str(cases),
                "--prompt",
                str(prompt),
                "--out",
                str(updated),
                "--questions",
                str(questions),
                "--iterations",
                "1",
            ]
        )
        == 0
    )
    assert updated.read_text().strip()
