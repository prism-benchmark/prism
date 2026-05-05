"""
evaluate.py — Phase 3: Merge Human + LLM results và tính Metric.

Yêu cầu: Phase 1 (run_human.py) và Phase 2 (run_llm.py --source X) đã hoàn tất.

Cách chạy:
    # -- Tính riêng từng phía --------------------------------------
    python pipeline/evaluate.py --mode human                    ← chỉ Human
    python pipeline/evaluate.py --mode llm   --source sea       ← chỉ LLM (sea)

    # -- So sánh Human vs LLM -------------------------------------
    python pipeline/evaluate.py --mode compare --source sea     ← so sánh 1 source
    python pipeline/evaluate.py --mode compare --source all     ← so sánh tất cả sources
    python pipeline/evaluate.py --source sea                    ← backward-compat (= compare)

Output:
    pipeline/output/metrics/human_Metrics.csv               ← human standalone
    pipeline/output/metrics/human_Overall.csv
    pipeline/output/metrics/{source}_Metrics.csv            ← llm standalone
    pipeline/output/metrics/{source}_Overall.csv
    pipeline/output/metrics/{source}_Final_Metrics.csv      ← compare: chi tiết
    pipeline/output/metrics/{source}_Overall_Average.csv    ← compare: trung bình
    pipeline/output/metrics/comparison_summary.csv          ← (nếu --source all)
"""

import sys
import os
import json
import argparse
import pandas as pd

# Thêm root project vào sys.path để import src/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.evaluate import calculate_review_metrics
import pipeline.config as config


# ================================================================
#  Data Loaders
# ================================================================

def load_human_results() -> dict:
    """
    Load toàn bộ kết quả Human từ output/human/*.json
    Returns: { paper_id: {...result dict...} }
    """
    results = {}
    if not os.path.isdir(config.OUTPUT_HUMAN_DIR):
        print(f"[!]  Chưa có kết quả Human tại: {config.OUTPUT_HUMAN_DIR}")
        print(f"   → Hãy chạy: python pipeline/run_batch.py --target human")
        return results

    for filename in os.listdir(config.OUTPUT_HUMAN_DIR):
        if not filename.endswith(".json"):
            continue
        filepath = os.path.join(config.OUTPUT_HUMAN_DIR, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        results[data["paper_id"]] = data

    return results


def load_llm_results(source_name: str) -> dict:
    """
    Load toàn bộ kết quả LLM từ output/{source_name}/*.json
    Returns: { paper_id: {...result dict...} }
    """
    results  = {}
    llm_dir  = config.get_llm_output_dir(source_name)

    if not os.path.isdir(llm_dir):
        print(f"[!]  Chưa có kết quả LLM '{source_name}' tại: {llm_dir}")
        print(f"   → Hãy chạy: python pipeline/run_batch.py --target {source_name}")
        return results

    for filename in os.listdir(llm_dir):
        if not filename.endswith(".json"):
            continue
        filepath = os.path.join(llm_dir, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        results[data["paper_id"]] = data

    return results


# ================================================================
#  Standalone Metric Computation (Human-only / LLM-only)
# ================================================================

def compute_human_standalone(filter_ids: set = None) -> pd.DataFrame:
    """
    Tính metric riêng cho Human reviews, không cần LLM.
    Mỗi paper: tính từng reviewer rồi lấy trung bình.

    Output columns:
        paper_id | n_reviewers | R_premise | S_depth |
        DoA_score | DoA_score_HM | Total_Claims | Total_Premises |
        Ratio_All_* | Ratio_Prem__*
    """
    print(f"\n{'=' * 60}")
    print(f"[=] Tính metric riêng: HUMAN (standalone)")
    print(f"{'=' * 60}")

    human_results = load_human_results()
    if not human_results:
        return pd.DataFrame()

    total_all = len(human_results)
    if filter_ids:
        human_results = {k: v for k, v in human_results.items() if k in filter_ids}
        print(f"  Human papers : {len(human_results)}  (lọc từ {total_all} → {len(human_results)})")
    else:
        print(f"  Human papers : {len(human_results)}")

    rows = []
    for paper_id in sorted(human_results.keys()):
        h_data = human_results[paper_id]

        per_reviewer = [
            calculate_review_metrics(args)
            for reviewer_id, args in h_data["reviews_analysis"].items()
            if reviewer_id.startswith("Human")
        ]
        if not per_reviewer:
            continue

        mean_m = pd.DataFrame(per_reviewer).mean().to_dict()

        row = {
            "paper_id":       paper_id,
            "n_reviewers":    len(per_reviewer),
            "R_premise":      mean_m["r_premise"],
            "S_depth":        mean_m["s_depth"],
            "DoA_score":      mean_m["doa_score"],
            "DoA_score_HM":   mean_m["doa_score_hm"],
            "Total_Claims":   mean_m["total_claims"],
            "Total_Premises": mean_m["total_premises"],
        }
        for k, v in mean_m.items():
            if k.startswith("Ratio_"):
                row[k] = v

        rows.append(row)

    df = pd.DataFrame(rows)
    print(f"  [OK] Xong {len(df)} papers")
    return df


def compute_llm_standalone(source_name: str, filter_ids: set = None) -> pd.DataFrame:
    """
    Tính metric riêng cho LLM reviews, không cần Human.
    Mỗi paper: 1 review duy nhất của LLM.

    Output columns:
        paper_id | LLM_source | Token_Total | R_premise | S_depth |
        DoA_score | DoA_score_HM | Total_Claims | Total_Premises |
        Ratio_All_* | Ratio_Prem__*
    """
    print(f"\n{'=' * 60}")
    print(f"[=] Tính metric riêng: {source_name.upper()} (standalone)")
    print(f"{'=' * 60}")

    llm_results = load_llm_results(source_name)
    if not llm_results:
        return pd.DataFrame()

    total_all = len(llm_results)
    if filter_ids:
        llm_results = {k: v for k, v in llm_results.items() if k in filter_ids}
        print(f"  LLM papers : {len(llm_results)}  (lọc từ {total_all} → {len(llm_results)})")
    else:
        print(f"  LLM papers : {len(llm_results)}")
    llm_key = source_name.upper()

    rows = []
    for paper_id in sorted(llm_results.keys()):
        l_data    = llm_results[paper_id]
        llm_args  = l_data["reviews_analysis"].get(llm_key, [])
        if not llm_args:
            continue

        m     = calculate_review_metrics(llm_args)
        usage = l_data.get("usage_stats", {})

        row = {
            "paper_id":       paper_id,
            "LLM_source":     source_name,
            "Token_Total":    usage.get("total_tokens", 0),
            "R_premise":      m["r_premise"],
            "S_depth":        m["s_depth"],
            "DoA_score":      m["doa_score"],
            "DoA_score_HM":   m["doa_score_hm"],
            "Total_Claims":   m["total_claims"],
            "Total_Premises": m["total_premises"],
        }
        for k, v in m.items():
            if k.startswith("Ratio_"):
                row[k] = v

        rows.append(row)

    df = pd.DataFrame(rows)
    print(f"  [OK] Xong {len(df)} papers")
    return df


# ================================================================
#  Compare Metric Computation (Human vs LLM)
# ================================================================

def compute_metrics_for_source(source_name: str, filter_ids: set = None) -> pd.DataFrame:
    """
    Tính metric so sánh Human vs 1 LLM source.
    Chỉ tính những paper có đủ cả 2 phía.
    """
    print(f"\n{'=' * 60}")
    print(f"[=] So sánh metric: Human  vs  {source_name.upper()}")
    print(f"{'=' * 60}")

    human_results = load_human_results()
    llm_results   = load_llm_results(source_name)

    if not human_results or not llm_results:
        return pd.DataFrame()

    common_ids = sorted(set(human_results.keys()) & set(llm_results.keys()))
    if filter_ids:
        common_ids = sorted(pid for pid in common_ids if pid in filter_ids)

    print(f"  Human papers : {len(human_results)}")
    print(f"  LLM papers   : {len(llm_results)}")
    print(f"  Papers khớp  : {len(common_ids)}" + (f"  (sau lọc)" if filter_ids else ""))

    if not common_ids:
        print("  [X] Không có paper_id nào khớp!")
        return pd.DataFrame()

    final_rows = []

    for paper_id in common_ids:
        h_data = human_results[paper_id]
        l_data = llm_results[paper_id]

        # -- Human metrics (trung bình tất cả reviewers) ----------
        human_metrics_list = [
            calculate_review_metrics(args)
            for reviewer_id, args in h_data["reviews_analysis"].items()
            if reviewer_id.startswith("Human")
        ]
        if not human_metrics_list:
            continue
        human_mean = pd.DataFrame(human_metrics_list).mean().to_dict()

        # -- LLM metrics ------------------------------------------
        llm_key  = source_name.upper()
        llm_args = l_data["reviews_analysis"].get(llm_key, [])
        if not llm_args:
            continue
        llm_metrics = calculate_review_metrics(llm_args)

        # -- Token usage ------------------------------------------
        h_usage = h_data.get("usage_stats", {})
        l_usage = l_data.get("usage_stats", {})

        # -- Tổng hợp thành 1 Row ---------------------------------
        row = {
            "paper_id":   paper_id,
            "LLM_source": source_name,

            "Human_Token_Total": h_usage.get("total_tokens", 0),
            "LLM_Token_Total":   l_usage.get("total_tokens", 0),

            # LLM metrics
            "LLM_R_premise":    llm_metrics["r_premise"],
            "LLM_S_depth":      llm_metrics["s_depth"],
            "LLM_DoA_score":    llm_metrics["doa_score"],
            "LLM_DoA_score_HM": llm_metrics["doa_score_hm"],
            "LLM_Claims":       llm_metrics["total_claims"],
            "LLM_Premises":     llm_metrics["total_premises"],

            # Human mean metrics
            "Human_Mean_R_premise":    human_mean["r_premise"],
            "Human_Mean_S_depth":      human_mean["s_depth"],
            "Human_Mean_DoA_score":    human_mean["doa_score"],
            "Human_Mean_DoA_score_HM": human_mean["doa_score_hm"],
            "Human_Mean_Claims":       human_mean["total_claims"],
            "Human_Mean_Premises":     human_mean["total_premises"],
        }

        for k, v in llm_metrics.items():
            if k.startswith("Ratio_"):
                row[f"LLM_{k}"] = v
        for k, v in human_mean.items():
            if k.startswith("Ratio_"):
                row[f"Human_{k}"] = v

        final_rows.append(row)

    return pd.DataFrame(final_rows)


# ================================================================
#  Save & Print helpers
# ================================================================

def save_and_print_standalone(df: pd.DataFrame, name: str):
    """Lưu CSV và in tổng kết cho standalone mode (human-only / llm-only)."""
    os.makedirs(config.OUTPUT_METRICS_DIR, exist_ok=True)

    detail_path = os.path.join(config.OUTPUT_METRICS_DIR, f"{name}_Metrics.csv")
    df.to_csv(detail_path, index=False)
    print(f"\n  [>] Chi tiết từng paper : {detail_path}")

    overall = df.mean(numeric_only=True)
    overall_path = os.path.join(config.OUTPUT_METRICS_DIR, f"{name}_Overall.csv")
    overall.to_frame(name="Overall_Mean").T.to_csv(overall_path, index=False)
    print(f"  [>] Trung bình tổng thể : {overall_path}")

    key_cols = ["R_premise", "S_depth", "DoA_score", "DoA_score_HM"]
    available = [c for c in key_cols if c in overall.index]
    print(f"\n  [=] Tổng kết [{name.upper()}]  (trung bình {len(df)} papers):")
    print(f"  {'-' * 40}")
    for col in available:
        print(f"  {col:<20}: {overall[col]:.4f}")


def save_and_print(df: pd.DataFrame, source_name: str):
    """Lưu CSV và in bảng tổng kết cho compare mode."""
    os.makedirs(config.OUTPUT_METRICS_DIR, exist_ok=True)

    detail_path = os.path.join(config.OUTPUT_METRICS_DIR, f"{source_name}_Final_Metrics.csv")
    df.to_csv(detail_path, index=False)
    print(f"\n  [>] Chi tiết từng paper : {detail_path}")

    overall = df.mean(numeric_only=True)
    overall_path = os.path.join(config.OUTPUT_METRICS_DIR, f"{source_name}_Overall_Average.csv")
    overall.to_frame(name="Overall_Mean").T.to_csv(overall_path, index=False)
    print(f"  [>] Trung bình tổng thể : {overall_path}")

    key_cols = [
        "LLM_R_premise", "LLM_S_depth", "LLM_DoA_score", "LLM_DoA_score_HM",
        "Human_Mean_R_premise", "Human_Mean_S_depth",
        "Human_Mean_DoA_score", "Human_Mean_DoA_score_HM",
    ]
    available = [c for c in key_cols if c in overall.index]
    print(f"\n  [=] Tổng kết (trung bình {len(df)} papers):")
    print(f"  {'-' * 48}")

    labels = {
        "LLM_R_premise":           "  LLM  R_premise     ",
        "LLM_S_depth":             "  LLM  S_depth       ",
        "LLM_DoA_score":           "  LLM  DoA_score (×) ",
        "LLM_DoA_score_HM":        "  LLM  DoA_score (HM)",
        "Human_Mean_R_premise":    "  HUM  R_premise     ",
        "Human_Mean_S_depth":      "  HUM  S_depth       ",
        "Human_Mean_DoA_score":    "  HUM  DoA_score (×) ",
        "Human_Mean_DoA_score_HM": "  HUM  DoA_score (HM)",
    }
    for col in available:
        print(f"  {labels.get(col, col)}: {overall[col]:.4f}")


def compare_all_sources(human_results: dict):
    """So sánh tất cả LLM sources trên cùng 1 bảng."""
    summary_rows = []

    for source_name in config.LLM_SOURCES:
        llm_results = load_llm_results(source_name)
        if not llm_results:
            continue
        if not (set(human_results.keys()) & set(llm_results.keys())):
            continue

        df = compute_metrics_for_source(source_name)
        if df.empty:
            continue

        save_and_print(df, source_name)

        overall = df.mean(numeric_only=True)
        row = {"LLM_source": source_name, "n_papers": len(df)}
        for col in ["LLM_R_premise", "LLM_S_depth", "LLM_DoA_score", "LLM_DoA_score_HM",
                    "Human_Mean_R_premise", "Human_Mean_S_depth",
                    "Human_Mean_DoA_score", "Human_Mean_DoA_score_HM"]:
            if col in overall.index:
                row[col] = round(overall[col], 4)
        summary_rows.append(row)

    if not summary_rows:
        print("[!]  Không có dữ liệu để so sánh.")
        return

    df_summary = pd.DataFrame(summary_rows).set_index("LLM_source")
    summary_path = os.path.join(config.OUTPUT_METRICS_DIR, "comparison_summary.csv")
    os.makedirs(config.OUTPUT_METRICS_DIR, exist_ok=True)
    df_summary.to_csv(summary_path)

    print(f"\n{'=' * 60}")
    print(f"[*] SO SÁNH TẤT CẢ LLM SOURCES vs HUMAN")
    print(f"{'=' * 60}")
    print(df_summary.to_string())
    print(f"\n  [>] Bảng so sánh: {summary_path}")


# ================================================================
#  Entry Point
# ================================================================

if __name__ == "__main__":
    valid_sources = list(config.LLM_SOURCES.keys())

    parser = argparse.ArgumentParser(
        description="DoA Pipeline — Phase 3: Compute Metrics",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--mode", type=str, default="compare",
        choices=["human", "llm", "compare"],
        help=(
            "Chế độ tính metric:\n"
            "  human   — chỉ tính Human (không cần LLM)\n"
            "  llm     — chỉ tính LLM   (không cần Human, yêu cầu --source)\n"
            "  compare — so sánh Human vs LLM (mặc định, yêu cầu --source)\n"
        ),
    )
    parser.add_argument(
        "--source", type=str, default=None,
        help=(
            f"Tên LLM source. Hiện có: {valid_sources}\n"
            f"Dùng 'all' (chỉ với --mode compare) để so sánh tất cả sources."
        ),
    )
    parser.add_argument(
        "--filter", type=str, default=None, metavar="FILE",
        help=(
            "Đường dẫn tới file .txt chứa danh sách paper_id (1 ID/dòng).\n"
            "Nếu cung cấp, chỉ tính metric cho các paper trong danh sách này.\n"
            "Ví dụ: --filter paper_ids_200.txt"
        ),
    )
    parser.add_argument(
        "--tag", type=str, default=None, metavar="TAG",
        help=(
            "Tên nhãn thêm vào tên file output (ví dụ: --tag 200).\n"
            "Nếu không cung cấp và có --filter, tự động dùng số lượng IDs."
        ),
    )
    args = parser.parse_args()

    # -- Load filter IDs nếu có ----------------------------------
    filter_ids = None
    filter_suffix = ""
    if args.filter:
        if not os.path.isfile(args.filter):
            print(f"[X] File filter không tồn tại: {args.filter}")
            sys.exit(1)
        with open(args.filter, "r", encoding="utf-8") as fh:
            filter_ids = {line.strip() for line in fh if line.strip()}
        tag = args.tag if args.tag else str(len(filter_ids))
        filter_suffix = f"_top{tag}"
        print(f"\n[~] Filter: {len(filter_ids)} paper IDs từ '{args.filter}'  (suffix='{filter_suffix}')")
    elif args.tag:
        filter_suffix = f"_{args.tag}"

    # Wrap save helpers để inject suffix vào tên file --------
    _orig_save_standalone = save_and_print_standalone
    _orig_save_compare    = save_and_print

    def _save_standalone_filtered(df, name):
        _orig_save_standalone(df, name + filter_suffix)

    def _save_compare_filtered(df, source_name):
        _orig_save_compare(df, source_name + filter_suffix)

    # -- Mode: human ----------------------------------------------
    if args.mode == "human":
        df = compute_human_standalone(filter_ids=filter_ids)
        if not df.empty:
            _save_standalone_filtered(df, "human")

    # -- Mode: llm ------------------------------------------------
    elif args.mode == "llm":
        if not args.source:
            print("[X] --mode llm yêu cầu --source <tên_source>")
            print(f"   Ví dụ: python pipeline/evaluate.py --mode llm --source sea")
            sys.exit(1)
        if args.source not in valid_sources:
            print(f"[X] Source '{args.source}' không tồn tại.")
            print(f"   Các source hiện có: {valid_sources}")
            sys.exit(1)
        df = compute_llm_standalone(args.source, filter_ids=filter_ids)
        if not df.empty:
            _save_standalone_filtered(df, args.source)

    # -- Mode: compare (default) ----------------------------------
    else:
        if not args.source:
            print("[X] --mode compare yêu cầu --source <tên_source> hoặc 'all'")
            print(f"   Ví dụ: python pipeline/evaluate.py --source sea")
            sys.exit(1)
        if args.source == "all":
            human_results = load_human_results()
            # Pass filter_ids via closure into compare_all_sources
            _orig_compute = compute_metrics_for_source
            import functools
            def _filtered_compute(sname):
                return _orig_compute(sname, filter_ids=filter_ids)
            # Temporarily patch this module's function (avoid cross-module aliasing).
            _self = sys.modules[__name__]
            _self.compute_metrics_for_source = _filtered_compute
            compare_all_sources(human_results)
            _self.compute_metrics_for_source = _orig_compute
        else:
            if args.source not in valid_sources:
                print(f"[X] Source '{args.source}' không tồn tại.")
                print(f"   Các source hiện có: {valid_sources} | 'all'")
                sys.exit(1)
            df = compute_metrics_for_source(args.source, filter_ids=filter_ids)
            if not df.empty:
                _save_compare_filtered(df, args.source)

