# -*- coding: utf-8 -*-
"""
run_mimo_all_sources.py -- Helper script de chay Mimo cho tat ca LLM sources.

Cach chay:
    # Chay tat ca 30 sources tren tat ca 5 conferences
    # (tu dong dung paper_ids_50 tuong ung cho tung conference)
    python pipeline/run_mimo_all_sources.py

    # Chi chay sources cua 1 conference (voi paper_ids_50 tuong ung)
    python pipeline/run_mimo_all_sources.py --conference ICLR2025
    python pipeline/run_mimo_all_sources.py --conference ICML2025
    python pipeline/run_mimo_all_sources.py --conference NeurIPS2025

    # Chi chay nhung source cu the (tu dong dung paper_ids_50 cua conference tuong ung)
    python pipeline/run_mimo_all_sources.py --sources sea_iclr2025 tree_iclr2025 reviewer2_iclr2025

    # Chay tat ca khong filter IDs
    python pipeline/run_mimo_all_sources.py --all

Output: pipeline/output/mimo_{source_name}/{paper_id}.json

Paper IDs Mapping:
  ICLR2024    -> DATA_ROOT/ICLR2024/paper_ids_50_iclr2024.txt
  ICLR2025    -> DATA_ROOT/ICLR2025/paper_ids_50_iclr2025.txt
  ICLR2026    -> DATA_ROOT/ICLR2026/paper_ids_50_iclr2026.txt
  ICML2025    -> DATA_ROOT/ICML2025/paper_ids_50_icml2025.txt
  NeurIPS2025 -> DATA_ROOT/Neurlps2025/paper_ids_50_neurips2025.txt
"""

import sys
import os
import subprocess
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pipeline.config as config


# ================================================================
#  Danh sach tat ca LLM sources theo tung conference
# ================================================================

SOURCES_BY_CONFERENCE = {
    "ICLR2024": [
        "sea_iclr2024",
        "tree_iclr2024",
        "reviewer2_iclr2024",
        "deepreview_iclr2024",
        "cyclereview_iclr2024",
    ],
    "ICLR2025": [
        "sea_iclr2025",
        "tree_iclr2025",
        "reviewer2_iclr2025",
        "deepreview_iclr2025",
        "cyclereview_iclr2025",
    ],
    "ICLR2026": [
        "sea_iclr2026",
        "tree_iclr2026",
        "reviewer2_iclr2026",
        "deepreview_iclr2026",
        "cyclereview_iclr2026",
    ],
    "ICML2025": [
        "sea_icml2025",
        "tree_icml2025",
        "reviewer2_icml2025",
        "deepreview_icml2025",
        "cyclereview_icml2025",
    ],
    "NeurIPS2025": [
        "sea_neurlps2025",
        "tree_neurips2025",
        "reviewer2_neurips2025",
        "deepreview_neurips2025",
        "cyclereview_neurlps2025",
    ],
}

# Paper IDs file mapping cho tung conference
PAPER_IDS_50_BY_CONFERENCE = {
    conf: config.paper_ids_file(conf, 50)
    for conf in ["ICLR2024", "ICLR2025", "ICLR2026", "ICML2025", "NeurIPS2025"]
}

ALL_SOURCES = []
for conf_sources in SOURCES_BY_CONFERENCE.values():
    ALL_SOURCES.extend(conf_sources)

# Reverse mapping: source_name  conference (e tu ong chon paper IDs)
SOURCE_TO_CONFERENCE = {}
for conf_name, conf_sources in SOURCES_BY_CONFERENCE.items():
    for src in conf_sources:
        SOURCE_TO_CONFERENCE[src] = conf_name


def get_paper_ids_for_source(source_name: str) -> str:
    """Tu ong lay paper_ids_50 tuong ung voi conference cua source."""
    conf = SOURCE_TO_CONFERENCE.get(source_name)
    if conf and conf in PAPER_IDS_50_BY_CONFERENCE:
        return PAPER_IDS_50_BY_CONFERENCE[conf]
    return config.PAPER_IDS_50_FILE  # fallback ICLR2024


# ================================================================
#  Helper: Run single source
# ================================================================

def run_source(source_name: str, paper_ids_file: str = None, run_all: bool = False):
    """Run mot LLM source qua Mimo evaluator.
    Neu paper_ids_file khong chi inh, tu ong dung file 50 IDs cua conference tuong ung.
    """
    cmd = ["python", "pipeline/run_llm_mimo.py", "--source", source_name]

    if run_all:
        cmd.append("--all")
    else:
        # Tu ong chon paper IDs theo conference neu chua chi inh
        ids_file = paper_ids_file or get_paper_ids_for_source(source_name)
        cmd.extend(["--paper_ids", ids_file])

    print(f"\n{'=' * 70}")
    print(f">> Chay [{source_name}]")
    print(f"{'=' * 70}")
    print(f"   Command: {' '.join(cmd)}")
    print()

    try:
        result = subprocess.run(cmd, check=True, cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        print(f"[OK] [{source_name}] hoan tat thanh cong!\n")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERR] [{source_name}] Loi! Ma loi: {e.returncode}\n")
        return False


# ================================================================
#  Main: Run multiple sources
# ================================================================

def run_all_sources(sources: list, paper_ids_file: str = None, conference: str = None, run_all: bool = False, skip_errors: bool = True):
    """Run danh sach sources. Moi source tu ong dung paper_ids_50 tuong ung voi conference cua no."""
    total = len(sources)
    successful = 0
    failed_sources = []

    print(f"\n{'=' * 70}")
    print(f"[MIMO] MIMO Pipeline  Run {total} source(s)")
    if conference:
        print(f"   Conference: {conference}")
        ids_display = paper_ids_file or PAPER_IDS_50_BY_CONFERENCE.get(conference, config.PAPER_IDS_50_FILE)
        print(f"   Paper IDs : {ids_display}")
    else:
        if paper_ids_file:
            print(f"   Paper IDs : {paper_ids_file} (override cho tat ca sources)")
        else:
            print(f"   Paper IDs : auto  moi source dung file 50 IDs cua conference tuong ung")
    print(f"Sources: {sources}")
    print(f"{'=' * 70}\n")

    for source_name in sources:
        if source_name not in config.LLM_SOURCES:
            print(f"[ERR] Source '{source_name}' khong ton tai. Bo qua.\n")
            failed_sources.append(source_name)
            continue

        # Hien thi conference va paper IDs se dung
        inferred_conf = SOURCE_TO_CONFERENCE.get(source_name, "?")
        ids_used = paper_ids_file or get_paper_ids_for_source(source_name)
        if not run_all:
            print(f"  [NOTE] [{source_name}]  conference: {inferred_conf}  | IDs: {ids_used}")

        success = run_source(source_name, paper_ids_file=paper_ids_file, run_all=run_all)
        if success:
            successful += 1
        else:
            failed_sources.append(source_name)
            if not skip_errors:
                break

    # Summary
    print(f"\n{'=' * 70}")
    print(f"[STATS] TONG KET")
    print(f"{'=' * 70}")
    print(f"[OK] Thanh cong: {successful}/{total} sources")
    if failed_sources:
        print(f"[ERR] That bai: {len(failed_sources)} sources")
        print(f"   {', '.join(failed_sources)}")
    print(f"{'=' * 70}\n")

    return successful == total


# ================================================================
#  Entry Point
# ================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Mimo Pipeline  Run LLM sources for all conferences"
    )
    parser.add_argument(
        "--sources", nargs="*", default=None,
        help="Danh sach sources can chay. Mac inh: ALL sources"
    )
    parser.add_argument(
        "--conference", type=str, default=None,
        help=f"Chi chay sources cua conference (se dung paper_ids_50 tuong ung). Tuy chon: {list(SOURCES_BY_CONFERENCE.keys())}"
    )
    parser.add_argument(
        "--paper_ids", type=str, default=None,
        help="File paper IDs custom. Neu khong chi inh, se dung paper_ids_50 tuong ung voi conference."
    )
    parser.add_argument(
        "--all", action="store_true", dest="run_all",
        help="Chay tat ca papers (khong filter theo IDs)."
    )
    parser.add_argument(
        "--continue", action="store_true", dest="skip_errors",
        help="Tiep tuc chay cac source tiep theo neu mot source bi loi."
    )
    args = parser.parse_args()

    # Xac inh sources can chay
    conference_selected = None
    sources_to_run = []
    run_by_conference = False

    if args.conference:
        if args.conference not in SOURCES_BY_CONFERENCE:
            print(f"[ERR] Conference '{args.conference}' khong ton tai.")
            print(f"   Cac conference: {list(SOURCES_BY_CONFERENCE.keys())}")
            sys.exit(1)
        sources_to_run = SOURCES_BY_CONFERENCE[args.conference]
        conference_selected = args.conference
    elif args.sources and len(args.sources) > 0:
        sources_to_run = args.sources
    else:
        # Chay tat ca conferences voi paper IDs tuong ung
        run_by_conference = True

    # Run
    if run_by_conference:
        # Chay tung conference mot lan voi paper IDs tuong ung
        total_success = 0
        total_sources = sum(len(v) for v in SOURCES_BY_CONFERENCE.values())

        for conf_name, conf_sources in SOURCES_BY_CONFERENCE.items():
            success = run_all_sources(
                sources        = conf_sources,
                paper_ids_file = PAPER_IDS_50_BY_CONFERENCE.get(conf_name),
                conference     = conf_name,
                run_all        = args.run_all,
                skip_errors    = args.skip_errors,
            )
            if success:
                total_success += 1

        print(f"\n{'=' * 70}")
        print(f"[DONE] TAT CA CONFERENCES  {total_success}/{len(SOURCES_BY_CONFERENCE)} conferences hoan tat")
        print(f"{'=' * 70}\n")
        sys.exit(0 if total_success == len(SOURCES_BY_CONFERENCE) else 1)
    else:
        success = run_all_sources(
            sources        = sources_to_run,
            paper_ids_file = args.paper_ids,
            conference     = conference_selected,
            run_all        = args.run_all,
            skip_errors    = args.skip_errors,
        )
        sys.exit(0 if success else 1)











