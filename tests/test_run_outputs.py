import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run


def test_default_output_root_is_centralized():
    args = run.parse_args([])

    assert args.output_dir == run._REPO_ROOT / "output"
    assert run._aspect_output_dir(args.output_dir, "novelty") == (
        run._REPO_ROOT / "output" / "novelty"
    )


def test_constructiveness_uses_central_output_env(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(run, "_python", lambda *args, **kwargs: calls.append(kwargs) or 0)

    result = run.run_constructiveness(
        conferences=["iclr2024"],
        reviewers=["human"],
        output_dir=tmp_path / "constructiveness",
    )

    assert result == 0
    assert calls[0]["env"]["CONSTRUCTIVENESS_OUTPUT_ROOT"] == str(
        tmp_path / "constructiveness"
    )


def test_flaw_identification_separates_reviewer_outputs(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(
        run,
        "_python",
        lambda *args, **kwargs: calls.append(args) or 0,
    )

    result = run.run_flaw_identification(
        conferences=["iclr2024"],
        reviewers=["sea"],
        output_dir=tmp_path / "flaw_identification",
    )

    assert result == 0
    assert "--output-dir" in calls[0]
    output_index = calls[0].index("--output-dir") + 1
    assert Path(calls[0][output_index]) == (
        tmp_path / "flaw_identification" / "iclr2024" / "sea"
    )
