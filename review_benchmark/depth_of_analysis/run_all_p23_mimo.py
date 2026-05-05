# -*- coding: utf-8 -*-
"""
run_all_p23_mimo.py
===================
Chay Phase 2+3 Mimo tren tat ca sources (Human + LLM Reviewers x 5 conferences).
Phase 1 ADUs duoc lay tu Gemini output co san.

Cach chay:
    # Tat ca sources (Human + 5 LLMs x 5 conferences = 30 sources)
    python pipeline/run_all_p23_mimo.py

    # Chi Human sources
    python pipeline/run_all_p23_mimo.py --type human

    # Chi LLM sources
    python pipeline/run_all_p23_mimo.py --type llm

    # Chi 1 conference
    python pipeline/run_all_p23_mimo.py --conference ICLR2024
    python pipeline/run_all_p23_mimo.py --conference ICML2025
    python pipeline/run_all_p23_mimo.py --conference NeurIPS2025

    # Chi 1 LLM system
    python pipeline/run_all_p23_mimo.py --llm sea
    python pipeline/run_all_p23_mimo.py --llm deepreview

    # Tiep tuc neu loi
    python pipeline/run_all_p23_mimo.py --continue

Output: pipeline/output/p23mimo_{source_name}/{paper_id}.json
"""

import sys
import os
import subprocess
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR  = os.path.dirname(PIPELINE_DIR)

# ================================================================
#  Source definitions
# ================================================================

CONFERENCES = ["ICLR2024", "ICLR2025", "ICLR2026", "ICML2025", "NeurIPS2025"]

LLM_FOLDER_BY_CONF = {
    "sea": {
        "ICLR2024": "sea_iclr2024", "ICLR2025": "sea_iclr2025", "ICLR2026": "sea_iclr2026",
        "ICML2025": "sea_icml2025", "NeurIPS2025": "sea_neurlps2025",
    },
    "tree": {
        "ICLR2024": "tree_iclr2024", "ICLR2025": "tree_iclr2025", "ICLR2026": "tree_iclr2026",
        "ICML2025": "tree_icml2025", "NeurIPS2025": "tree_neurips2025",
    },
    "reviewer2": {
        "ICLR2024": "reviewer2_iclr2024", "ICLR2025": "reviewer2_iclr2025", "ICLR2026": "reviewer2_iclr2026",
        "ICML2025": "reviewer2_icml2025", "NeurIPS2025": "reviewer2_neurips2025",
    },
    "deepreview": {
        "ICLR2024": "deepreview_iclr2024", "ICLR2025": "deepreview_iclr2025", "ICLR2026": "deepreview_iclr2026",
        "ICML2025": "deepreview_icml2025", "NeurIPS2025": "deepreview_neurips2025",
    },
    "cyclereview": {
        "ICLR2024": "cyclereview_iclr2024", "ICLR2025": "cyclereview_iclr2025", "ICLR2026": "cyclereview_iclr2026",
        "ICML2025": "cyclereview_icml2025", "NeurIPS2025": "cyclereview_neurlps2025",
    },
}

HUMAN_FOLDER_BY_CONF = {
    "ICLR2024":    "human_iclr2024",
    "ICLR2025":    "human_iclr2025",
    "ICLR2026":    "human_iclr2026",
    "ICML2025":    "human_icml2025",
    "NeurIPS2025": "human_neurips2025",
}

LLM_SYSTEMS = list(LLM_FOLDER_BY_CONF.keys())


def get_all_sources(source_type: str = "all", conference: str = None, llm: str = None) -> list:
    """Lay danh sach (source_folder_name) theo filter."""
    confs = [conference] if conference else CONFERENCES
    sources = []

    if source_type in ("all", "human"):
        for conf in confs:
            folder = HUMAN_FOLDER_BY_CONF.get(conf)
            if folder:
                sources.append(folder)

    if source_type in ("all", "llm"):
        llm_systems = [llm] if llm else LLM_SYSTEMS
        for llm_name in llm_systems:
            for conf in confs:
                folder = LLM_FOLDER_BY_CONF.get(llm_name, {}).get(conf)
                if folder:
                    sources.append(folder)

    return sources


# ================================================================
#  Run 1 source
# ================================================================

def run_source(source_name: str, run_all: bool = False):
    cmd = ["python", "pipeline/run_p23_mimo.py", "--source", source_name]
    if run_all:
        cmd.append("--all")

    print(f"\n{'=' * 65}")
    print(f">> [{source_name}]")
    print(f"   cmd: {' '.join(cmd)}")
    print(f"{'=' * 65}")

    try:
        subprocess.run(cmd, check=True, cwd=PROJECT_DIR)
        print(f"[OK] {source_name} done.\n")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERR] {source_name} failed (code {e.returncode})\n")
        return False


# ================================================================
#  Entry Point
# ================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Phase 2+3 Mimo on all Gemini-segmented sources."
    )
    parser.add_argument(
        "--type", type=str, default="all", choices=["all", "human", "llm"],
        help="Loai source: all / human / llm (default: all)"
    )
    parser.add_argument(
        "--conference", type=str, default=None, choices=CONFERENCES,
        help="Chi chay 1 conference."
    )
    parser.add_argument(
        "--llm", type=str, default=None, choices=LLM_SYSTEMS,
        help=f"Chi chay 1 LLM system: {LLM_SYSTEMS}"
    )
    parser.add_argument(
        "--all", action="store_true", dest="run_all",
        help="Chay tat ca papers (bo qua filter 50 IDs)."
    )
    parser.add_argument(
        "--continue", action="store_true", dest="skip_errors",
        help="Tiep tuc khi gap loi."
    )
    args = parser.parse_args()

    sources = get_all_sources(
        source_type = args.type,
        conference  = args.conference,
        llm         = args.llm,
    )

    if not sources:
        print("[ERR] Khong co source nao.")
        sys.exit(1)

    print(f"\n{'=' * 65}")
    print(f"[P23-MIMO] Gemini ADUs -> Mimo Phase 2+3")
    print(f"  Total sources: {len(sources)}")
    print(f"  Sources: {sources}")
    print(f"{'=' * 65}")

    ok_count   = 0
    fail_list  = []

    for src in sources:
        ok = run_source(src, run_all=args.run_all)
        if ok:
            ok_count += 1
        else:
            fail_list.append(src)
            if not args.skip_errors:
                print(f"[STOP] Dung lai vi loi. Dung --continue de tiep tuc.")
                break

    print(f"\n{'=' * 65}")
    print(f"[DONE] {ok_count}/{len(sources)} sources hoan tat")
    if fail_list:
        print(f"[FAIL] {len(fail_list)} sources that bai: {', '.join(fail_list)}")
    print(f"{'=' * 65}")

    sys.exit(0 if ok_count == len(sources) else 1)

