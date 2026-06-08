#!/usr/bin/env python3
"""
PRISM Unified Experiment Runner.

Single entry point for all PRISM experiments — evaluation and review generation.

Usage:
    python run.py                          # all enabled items
    python run.py --list                   # list available
    python run.py --profile quick          # smoke test
    python run.py --only constructiveness   # specific aspect
    python run.py --skip novelty            # skip an aspect
    python run.py --limit 10               # quick subset (first N papers)

Run --help on any sub-module for detailed pipeline flags:
    python run.py -- --help                # forward to default pipeline
    python run.py --only constructiveness -- --help

Examples:
    # Full paper experiment (all 4 aspects, all 5 conferences, human + 5 LLMs)
    python run.py

    # Just depth of analysis on ICLR 2024
    python run.py --only depth_of_analysis

    # Quick test: constructiveness on 10 papers
    python run.py --only constructiveness --limit 10

    # List all available pipelines
    python run.py --list
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

_REPO_ROOT = Path(__file__).parent
sys.path.insert(0, str(_REPO_ROOT))

from dotenv import load_dotenv

load_dotenv(_REPO_ROOT / ".env", override=False)

# ── Config system from llm_client.py ──────────────────────────────────────
from llm_client import (
    PRISMLLMClient,
    get_aspect_config,
    get_reviewer_config,
    get_profile_names,
    is_enabled,
    list_all,
    list_disabled,
    list_enabled,
    resolve_items,
)

logger = logging.getLogger("prism.run")


# ═══════════════════════════════════════════════════════════════════════════
# Pipeline dispatchers
# ═══════════════════════════════════════════════════════════════════════════

ASPECT_DISPATCH: Dict[str, str] = {
    "constructiveness": "run_constructiveness",
    "depth_of_analysis": "run_depth_of_analysis",
    "flaw_identification": "run_flaw_identification",
    "novelty": "run_novelty",
}

REVIEWER_DISPATCH: Dict[str, str] = {
    "sea": "generate_sea",
    "reviewer2": "generate_reviewer2",
    "treereview": "generate_treereview",
    "deepreview": "generate_deepreview",
    "cyclereview": "generate_cyclereview",
}

CONFERENCE_NAMES = ["iclr2024", "iclr2025", "iclr2026", "icml2025", "neurips2025"]
CONFERENCE_ENV = ["ICLR2024", "ICLR2025", "ICLR2026", "ICML2025", "NeurIPS2025"]

SHORT_CONF = {
    "iclr2024": "iclr2024",
    "iclr2025": "iclr2025",
    "iclr2026": "iclr2026",
    "icml2025": "icml2025",
    "neurips2025": "neurips2025",
}


# ═══════════════════════════════════════════════════════════════════════════
# Shared helpers
# ═══════════════════════════════════════════════════════════════════════════


def _python(*args: str, cwd: Optional[str] = None, **kwargs) -> int:
    """Run a Python script via subprocess, returning the exit code."""
    cmd = [sys.executable] + list(args)
    print(f"\n[RUN] {' '.join(cmd)}")
    sys.stdout.flush()
    result = subprocess.run(cmd, cwd=cwd, **kwargs)
    return result.returncode


def _aspect_dir(aspect: str) -> Path:
    """Return pipeline root for an aspect."""
    mapping = {
        "constructiveness": "Aspects_benchmarking/constructiveness",
        "depth_of_analysis": "Aspects_benchmarking/depth_of_analysis",
        "flaw_identification": "Aspects_benchmarking/flaw_identification",
        "novelty": "Aspects_benchmarking/novelty_vefification",
    }
    return _REPO_ROOT / mapping[aspect]


def _read_paper_ids(conference: str) -> List[str]:
    """Read paper IDs from a conference's paper_ids_200 file."""
    try:
        from Aspects_benchmarking.env_loader import conf_path

        env_conf = conference.upper()
        if env_conf == "NEURIPS2025":
            env_conf = "NeurIPS2025"
        base = conf_path(env_conf)
        conf_lower = conference.lower().replace("neurips2025", "neurlps2025")
        ids_file = os.path.join(base, f"paper_ids_200_{conf_lower}.txt")
        if not os.path.exists(ids_file):
            return []
        with open(ids_file) as f:
            return [line.strip() for line in f if line.strip()]
    except Exception:
        return []


# ═══════════════════════════════════════════════════════════════════════════
# Depth of Analysis
# ═══════════════════════════════════════════════════════════════════════════

LLM_TYPE_TO_DOA_FORMAT = {
    "sea": "txt",
    "tree": "tree_json",
    "reviewer2": "reviewer2_txt",
    "deepreview": "deepreview_json",
    "cyclereview": "cyclereview_json",
}


def run_depth_of_analysis(
    conferences: List[str],
    reviewers: List[str],
    limit: Optional[int] = None,
    extra_args: Optional[List[str]] = None,
) -> int:
    """Run DoA evaluation for specified conferences and reviewer types."""
    cwd = _aspect_dir("depth_of_analysis")
    exit_code = 0

    for conf in conferences:
        conf_env = conf.upper()
        if conf_env == "NEURIPS2025":
            conf_env = "NeurIPS2025"

        # 1. Evaluate human reviews
        print(f"\n{'=' * 60}")
        print(f"  Depth of Analysis — Human — {conf_env}")
        print(f"{'=' * 60}")
        script = cwd / "run_human.py"
        if script.exists():
            code = _python(
                str(script),
                "--conference",
                conf_env,
                *(extra_args or []),
                cwd=str(cwd),
            )
            if code != 0:
                exit_code = code

        # 2. Evaluate each LLM reviewer type
        for rev in reviewers:
            source_name = f"{rev}_{conf_env.lower()}"
            if conf_env.lower() == "neurips2025":
                source_name = source_name.replace("neurips2025", "neurlps2025")

            print(f"\n{'=' * 60}")
            print(f"  Depth of Analysis — {rev} — {conf_env}")
            print(f"{'=' * 60}")
            script = cwd / "run_llm.py"
            if script.exists():
                code = _python(
                    str(script),
                    "--source",
                    source_name,
                    *(extra_args or []),
                    cwd=str(cwd),
                )
                if code != 0:
                    exit_code = code

        # 3. Compute metrics
        print(f"\n  Computing DoA metrics for {conf_env}...")
        metrics_script = cwd / "evaluate_all.py"
        if metrics_script.exists():
            code = _python(
                str(metrics_script),
                "--conference",
                conf.lower(),
                *(extra_args or []),
                cwd=str(cwd),
            )
            if code != 0:
                exit_code = code

    return exit_code


# ═══════════════════════════════════════════════════════════════════════════
# Constructiveness
# ═══════════════════════════════════════════════════════════════════════════

REVIEWER_TO_CONSTRUCT_MODE = {
    "human": "human",
    "sea": "sea",
    "reviewer2": "reviewer2",
    "tree": "tree",
    "deepreview": "deepreview",
    "cyclereview": "cyclereview",
}


def run_constructiveness(
    conferences: List[str],
    reviewers: List[str],
    limit: Optional[int] = None,
    extra_args: Optional[List[str]] = None,
) -> int:
    """Run Constructiveness evaluation for specified conferences and reviewers."""
    cwd = _aspect_dir("constructiveness")
    exit_code = 0
    script = cwd / "run_constructiveness.py"

    if not script.exists():
        print(f"  [ERROR] Script not found: {script}")
        return 1

    for conf in conferences:
        for rev in reviewers:
            mode = REVIEWER_TO_CONSTRUCT_MODE.get(rev, rev)
            print(f"\n{'=' * 60}")
            print(f"  Constructiveness — {rev} — {conf}")
            print(f"{'=' * 60}")

            cmd = [str(script), "--mode", mode, "--conf", conf]
            if limit:
                cmd += ["--limit", str(limit)]
            if extra_args:
                cmd += extra_args

            code = _python(*cmd, cwd=str(cwd))
            if code != 0:
                exit_code = code

    return exit_code


# ═══════════════════════════════════════════════════════════════════════════
# Flaw Identification
# ═══════════════════════════════════════════════════════════════════════════


def run_flaw_identification(
    conferences: List[str],
    reviewers: List[str],
    limit: Optional[int] = None,
    extra_args: Optional[List[str]] = None,
) -> int:
    """Run Flaw ID evaluation for specified conferences and reviewers."""
    cwd = _aspect_dir("flaw_identification")
    exit_code = 0

    for conf in conferences:
        # Map conference name to the main_cfi script
        conf_lower = conf.lower().replace("neurips2025", "neurlps2025")
        script = cwd / f"main_cfi_{conf_lower}.py"
        if not script.exists():
            # Try without the neurIPS mapping
            script = cwd / f"main_cfi_{conf.lower()}.py"
        if not script.exists():
            print(f"  [WARN] No flaw ID script for {conf}: {script}")
            continue

        for rev in reviewers:
            print(f"\n{'=' * 60}")
            print(f"  Flaw ID — {rev} — {conf}")
            print(f"{'=' * 60}")

            cmd = [str(script), "--mode", "all", "--llm-type", rev]
            if limit:
                cmd += [f"--limit={limit}"]
            if extra_args:
                cmd += extra_args

            code = _python(*cmd, cwd=str(cwd))
            if code != 0:
                exit_code = code

    return exit_code


# ═══════════════════════════════════════════════════════════════════════════
# Novelty Assessment
# ═══════════════════════════════════════════════════════════════════════════

REVIEWER_TO_NOVELTY_TYPE = {
    "human": "human",
    "sea": "sea",
    "tree": "tree",
    "deepreview": "deepreview",
    "cyclereview": "cyclereview",
    "reviewer2": "reviewer2",
}


def run_novelty(
    conferences: List[str],
    reviewers: List[str],
    limit: Optional[int] = None,
    extra_args: Optional[List[str]] = None,
) -> int:
    """Run Novelty Assessment for specified conferences and reviewers.

    The novelty pipeline has its own pyproject.toml and dependencies.
    It requires a --data-root pointing to the dataset.
    """
    cwd = _aspect_dir("novelty")
    script = cwd / "scripts" / "run_pipeline.py"
    if not script.exists():
        print(f"  [ERROR] Novelty script not found: {script}")
        print("  (novelty pipeline may need 'uv sync' in its directory)")
        return 1

    from Aspects_benchmarking.env_loader import DATA_ROOT

    data_root = DATA_ROOT
    if not data_root:
        print("  [ERROR] DATA_ROOT not set in .env. Cannot run novelty pipeline.")
        return 1

    exit_code = 0
    for conf in conferences:
        # Map conf name to novelty's naming convention
        conf_map = {
            "iclr2024": "ICLR_2024",
            "iclr2025": "ICLR_2025",
            "iclr2026": "ICLR_2026",
            "icml2025": "ICML_2025",
            "neurips2025": "NeurIPS_2025",
        }
        novelty_conf = conf_map.get(conf.lower(), conf)

        for rev in reviewers:
            review_type = REVIEWER_TO_NOVELTY_TYPE.get(rev, rev)
            print(f"\n{'=' * 60}")
            print(f"  Novelty — {rev} — {conf}")
            print(f"{'=' * 60}")

            cmd = [
                str(script),
                "--data-root",
                data_root,
                "--conferences",
                novelty_conf,
                "--review-types",
                review_type,
            ]
            if extra_args:
                cmd += extra_args

            code = _python(*cmd, cwd=str(cwd))
            if code != 0:
                exit_code = code

    return exit_code


# ═══════════════════════════════════════════════════════════════════════════
# Review generation (LLM_reviewer)
# ═══════════════════════════════════════════════════════════════════════════


def generate_sea(
    conferences: List[str],
    limit: Optional[int] = None,
    extra_args: Optional[List[str]] = None,
) -> int:
    cwd = _REPO_ROOT / "LLM_reviewer" / "SEA"
    script = cwd / "generate_reviews.py"
    if not script.exists():
        print(f"  [ERROR] SEA script not found: {script}")
        return 1
    exit_code = 0
    for conf in conferences:
        print(f"\n{'=' * 60}")
        print(f"  SEA generation — {conf}")
        print(f"{'=' * 60}")
        code = _python(str(script), *(extra_args or []), cwd=str(cwd))
        if code != 0:
            exit_code = code
    return exit_code


def generate_reviewer2(
    conferences: List[str],
    limit: Optional[int] = None,
    extra_args: Optional[List[str]] = None,
) -> int:
    cwd = _REPO_ROOT / "LLM_reviewer" / "Reviewer2"
    exit_code = 0
    for conf in conferences:
        script_name = f"demo_{conf.lower()}_vllm.py" if limit else "demo.py"
        script = cwd / script_name
        if not script.exists():
            script = cwd / "demo.py"
        if not script.exists():
            print(f"  [WARN] No Reviewer2 script for {conf}")
            continue
        print(f"\n{'=' * 60}")
        print(f"  Reviewer2 generation — {conf}")
        print(f"{'=' * 60}")
        code = _python(str(script), *(extra_args or []), cwd=str(cwd))
        if code != 0:
            exit_code = code
    return exit_code


def generate_treereview(
    conferences: List[str],
    limit: Optional[int] = None,
    extra_args: Optional[List[str]] = None,
) -> int:
    cwd = _REPO_ROOT / "LLM_reviewer" / "TreeReview" / "treereview_batch_pipeline"
    script = cwd / "run_batch_treereview.py"
    if not script.exists():
        print(f"  [ERROR] TreeReview script not found: {script}")
        return 1
    exit_code = 0
    for conf in conferences:
        print(f"\n{'=' * 60}")
        print(f"  TreeReview generation — {conf}")
        print(f"{'=' * 60}")
        cmd = [str(script), *(extra_args or [])]
        code = _python(*cmd, cwd=str(cwd))
        if code != 0:
            exit_code = code
    return exit_code


def generate_deepreview(
    conferences: List[str],
    limit: Optional[int] = None,
    extra_args: Optional[List[str]] = None,
) -> int:
    cwd = _REPO_ROOT / "LLM_reviewer" / "Deepreview_CycleReview" / "run"
    exit_code = 0
    for conf in conferences:
        script = cwd / f"run_deepreviewer_{conf.lower()}.py"
        if not script.exists():
            script = cwd / "run_deepreviewer.py"
        if not script.exists():
            print(f"  [WARN] No DeepReview script for {conf}")
            continue
        print(f"\n{'=' * 60}")
        print(f"  DeepReview generation — {conf}")
        print(f"{'=' * 60}")
        code = _python(str(script), *(extra_args or []), cwd=str(cwd))
        if code != 0:
            exit_code = code
    return exit_code


def generate_cyclereview(
    conferences: List[str],
    limit: Optional[int] = None,
    extra_args: Optional[List[str]] = None,
) -> int:
    cwd = _REPO_ROOT / "LLM_reviewer" / "Deepreview_CycleReview" / "run"
    exit_code = 0
    for conf in conferences:
        script = cwd / f"run_cyclereviewer_{conf.lower()}.py"
        if not script.exists():
            script = cwd / "run_cyclereviewer.py"
        if not script.exists():
            print(f"  [WARN] No CycleReview script for {conf}")
            continue
        print(f"\n{'=' * 60}")
        print(f"  CycleReview generation — {conf}")
        print(f"{'=' * 60}")
        code = _python(str(script), *(extra_args or []), cwd=str(cwd))
        if code != 0:
            exit_code = code
    return exit_code


# ═══════════════════════════════════════════════════════════════════════════
# Main orchestrator
# ═══════════════════════════════════════════════════════════════════════════


def resolve_aspects(args) -> List[str]:
    """Resolve which aspects to run based on CLI flags."""
    if args.only:
        items = [x.strip() for x in args.only.split(",")]
        return [i for i in items if i in ASPECT_DISPATCH]
    profile_items = None
    if args.profile:
        from llm_client import get_profile_items as _gpi

        profile_items = _gpi(args.profile)
    base = list_enabled("aspects")
    if profile_items is not None:
        base = [i for i in base if i in profile_items]
    if args.skip:
        skip_set = set(x.strip() for x in args.skip.split(","))
        base = [i for i in base if i not in skip_set]
    return base


def resolve_reviewers(args) -> List[str]:
    """Resolve which reviewer types to evaluate (human + LLMs)."""
    # Always include human
    result = ["human"]
    if args.only:
        items = [x.strip() for x in args.only.split(",")]
        llm_items = [i for i in items if i in REVIEWER_DISPATCH]
        result.extend(llm_items)
        return list(dict.fromkeys(result))  # dedup, preserve order
    profile_items = None
    if args.profile:
        from llm_client import get_profile_items as _gpi

        profile_items = _gpi(args.profile)
    base = list_enabled("reviewers")
    if profile_items is not None:
        base = [i for i in base if i in profile_items]
    if args.skip:
        skip_set = set(x.strip() for x in args.skip.split(","))
        base = [i for i in base if i not in skip_set]
    result.extend(base)
    return list(dict.fromkeys(result))


def resolve_conferences(args) -> List[str]:
    """Resolve which conferences to run on."""
    if args.conference:
        return [x.strip().lower() for x in args.conference.split(",")]
    return list(CONFERENCE_NAMES)


def validate_config(dry_run: bool = False) -> bool:
    """Check that essential config is present."""
    from Aspects_benchmarking.env_loader import DATA_ROOT

    if not DATA_ROOT:
        print("[ERROR] DATA_ROOT is not set in .env")
        print("  Copy .env.example to .env and set DATA_ROOT")
        return False
    if not os.path.isdir(DATA_ROOT):
        print(f"[WARN] DATA_ROOT directory not found: {DATA_ROOT}")
        print("  Run: python Data/setup_aspect_benchmark.py --write-env")
        if not dry_run:
            try:
                ans = input("  Continue anyway? [y/N] ")
                if ans.lower() != "y":
                    return False
            except (EOFError, OSError):
                return False
    return True


def print_list() -> None:
    """Print available aspects, reviewers, and profiles."""
    aspects = list_all("aspects")
    reviewers = list_all("reviewers")
    profiles = get_profile_names()

    print("\nAVAILABLE ASPECTS:")
    for a in aspects:
        enabled = "✓" if is_enabled("aspects", a) else "✗"
        cfg = get_aspect_config(a) or {}
        prov = cfg.get("provider", "?")
        model = cfg.get("model", "?")
        print(f"  {enabled} {a:<25} ({prov} / {model})")

    print("\nAVAILABLE REVIEWERS:")
    for r in reviewers:
        enabled = "✓" if is_enabled("reviewers", r) else "✗"
        cfg = get_reviewer_config(r) or {}
        rtype = cfg.get("type", "?")
        if rtype == "local":
            detail = cfg.get("model_path") or cfg.get("model_size", "") or "?"
        else:
            detail = f"{cfg.get('provider', '?')} / {cfg.get('model', '?')}"
        print(f"  {enabled} {r:<25} ({rtype}: {detail})")

    print("\nCONFERENCES:")
    for c in CONFERENCE_NAMES:
        n_papers = len(_read_paper_ids(c))
        status = (
            f"({n_papers} papers)"
            if n_papers
            else "(no data — run Data/setup_aspect_benchmark.py)"
        )
        print(f"    {c:<20} {status}")

    print("\nPROFILES:")
    for p in profiles:
        print(f"    {p}")

    print(
        "\nUsage:  python run.py [--profile <name>] [--only <items>] [--skip <items>]"
    )
    print("        python run.py --conference iclr2024")
    print("        python run.py --limit 10")
    print()


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="PRISM Unified Experiment Runner",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument(
        "--list",
        action="store_true",
        help="List available aspects, reviewers, and profiles",
    )
    p.add_argument(
        "--setup-data",
        action="store_true",
        help="Download and setup the PRISM benchmark dataset",
    )
    p.add_argument(
        "--profile",
        default=None,
        help="Run profile (quick, aspects, reviewers, heavy, etc.)",
    )
    p.add_argument(
        "--only",
        default=None,
        help="Comma-separated whitelist (e.g. 'constructiveness,novelty')",
    )
    p.add_argument(
        "--skip",
        default=None,
        help="Comma-separated blacklist (e.g. 'flaw_identification')",
    )
    p.add_argument(
        "--conference",
        default=None,
        help="Comma-separated conferences (e.g. 'iclr2024,icml2025')",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most N papers per pipeline (for quick tests)",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of concurrent workers for parallel paper execution and dispatching (default: 1)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would run without executing",
    )
    p.add_argument(
        "forward_args",
        nargs=argparse.REMAINDER,
        help="Extra args forwarded to each pipeline (use -- to separate)",
    )
    return p.parse_args(argv)


def _setup_data() -> int:
    """Download and prepare the PRISM dataset."""
    data_script = _REPO_ROOT / "Data" / "setup_aspect_benchmark.py"
    if not data_script.exists():
        print(f"[ERROR] Data setup script not found: {data_script}")
        return 1
    print("\n[SETUP] Downloading PRISM benchmark dataset...\n")
    return _python(str(data_script), "--write-env", cwd=str(_REPO_ROOT))


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    if args.list:
        print_list()
        return 0

    if args.setup_data:
        return _setup_data()

    if not validate_config(dry_run=args.dry_run):
        return 1

    aspects = resolve_aspects(args)
    conferences = resolve_conferences(args)
    reviewers = resolve_reviewers(args)
    # Resolve worker count
    workers = args.workers
    if workers is None:
        env_workers = os.getenv("PRISM_MAX_WORKERS")
        if env_workers:
            try:
                workers = int(env_workers)
            except ValueError:
                workers = 1
        else:
            workers = 1

    extra_args = (
        args.forward_args[1:]
        if args.forward_args and args.forward_args[0] == "--"
        else []
    )

    if workers > 1:
        if "--workers" not in extra_args and not any(arg.startswith("--workers=") for arg in extra_args):
            extra_args.extend(["--workers", str(workers)])
        if "--max-workers" not in extra_args and not any(arg.startswith("--max-workers=") for arg in extra_args):
            extra_args.extend(["--max-workers", str(workers)])

    # Distinguish aspect pipelines from reviewer generation pipelines
    aspect_names = [a for a in aspects if a in ASPECT_DISPATCH]
    reviewer_gen_names = [
        r for r in reviewers if r in REVIEWER_DISPATCH and r != "human"
    ]

    if not aspect_names and not reviewer_gen_names:
        print("[WARN] Nothing to run. Use --list to see available items.")
        return 0

    print(f"\n{'=' * 60}")
    print(f"  PRISM Experiment Runner")
    print(f"{'=' * 60}")
    print(f"  Conferences : {', '.join(conferences)}")
    print(f"  Aspects     : {', '.join(aspect_names) if aspect_names else '(none)'}")
    print(f"  Reviewers   : {', '.join(reviewers) if reviewers else '(none)'}")
    if args.limit:
        print(f"  Limit       : {args.limit} papers")
    if workers > 1:
        print(f"  Workers     : {workers}")
    print()

    if args.dry_run:
        print("[DRY RUN] No commands executed.\n")
        return 0

    exit_code = 0

    # ── Run aspect evaluations ────────────────────────────────────
    for aspect in aspect_names:
        print(f"\n{'#' * 60}")
        print(f"#  ASPECT: {aspect}")
        print(f"{'#' * 60}")

        dispatch_fn = {
            "constructiveness": run_constructiveness,
            "depth_of_analysis": run_depth_of_analysis,
            "flaw_identification": run_flaw_identification,
            "novelty": run_novelty,
        }[aspect]

        code = dispatch_fn(
            conferences=conferences,
            reviewers=reviewers,
            limit=args.limit,
            extra_args=extra_args,
        )
        if code != 0:
            exit_code = code
            print(f"  [WARN] {aspect} exited with code {code}")

    # ── Run reviewer generation ───────────────────────────────────
    for rev in reviewer_gen_names:
        print(f"\n{'#' * 60}")
        print(f"#  REVIEWER GENERATION: {rev}")
        print(f"{'#' * 60}")

        dispatch_fn = {
            "sea": generate_sea,
            "reviewer2": generate_reviewer2,
            "treereview": generate_treereview,
            "deepreview": generate_deepreview,
            "cyclereview": generate_cyclereview,
        }[rev]

        code = dispatch_fn(
            conferences=conferences,
            limit=args.limit,
            extra_args=extra_args,
        )
        if code != 0:
            exit_code = code
            print(f"  [WARN] {rev} generation exited with code {code}")

    print(f"\n{'=' * 60}")
    print(f"  Done! Exit code: {exit_code}")
    print(f"{'=' * 60}\n")

    return exit_code


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    sys.exit(main())
