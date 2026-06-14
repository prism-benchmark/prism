"""
evaluate_all.py — Tính metric DoA ĐỘC LẬP cho TỪNG hội nghị, TỪNG Human & TỪNG LLM source.

Không so sánh Human vs LLM — mỗi bên tính hoàn toàn riêng biệt.

Cách chạy:
    python pipeline/evaluate_all.py                           ← tính TẤT CẢ (human + llm)
    python pipeline/evaluate_all.py --mode human              ← chỉ Human (mọi hội nghị)
    python pipeline/evaluate_all.py --mode llm                ← chỉ LLM   (mọi source)
    python pipeline/evaluate_all.py --source sea_iclr2024     ← 1 LLM source cụ thể
    python pipeline/evaluate_all.py --conference iclr2024     ← 1 hội nghị (human + llm của hội nghị đó)
    python pipeline/evaluate_all.py --filter paper_ids_200.txt ← chỉ tính subset paper

Output:
    pipeline/output/metrics/human_iclr2024_Metrics.csv     ← chi tiết từng paper
    pipeline/output/metrics/human_iclr2024_Overall.csv     ← trung bình tổng thể
    pipeline/output/metrics/sea_iclr2024_Metrics.csv
    pipeline/output/metrics/sea_iclr2024_Overall.csv
    ...
    pipeline/output/metrics/ALL_Human_Summary.csv          ← bảng tổng hợp tất cả human
    pipeline/output/metrics/ALL_LLM_Summary.csv            ← bảng tổng hợp tất cả llm
"""

import sys
import os
import json
import argparse
import pandas as pd

# Fix Unicode encoding on Windows
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Thêm root project vào sys.path để import src/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.evaluate import calculate_review_metrics
import pipeline.config as config

# ================================================================
#  Mapping: tên hội nghị → thư mục output Human
# ================================================================
PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_ROOT = os.environ.get("DOA_OUTPUT_ROOT") or os.path.join(PIPELINE_DIR, "output")

HUMAN_CONFERENCES = {
    "iclr2024":    os.path.join(OUTPUT_ROOT, "human"),
    "iclr2025":    os.path.join(OUTPUT_ROOT, "human_iclr2025"),
    "iclr2026":    os.path.join(OUTPUT_ROOT, "human_iclr2026"),
    "icml2025":    os.path.join(OUTPUT_ROOT, "human_icml2025"),
    "neurips2025": os.path.join(OUTPUT_ROOT, "human_neurips2025"),
}

OUTPUT_METRICS_DIR = os.path.join(OUTPUT_ROOT, "metrics")


# ================================================================
#  Helper: load JSON files từ một thư mục output
# ================================================================

def load_json_folder(folder: str) -> dict:
    """
    Đọc tất cả file *.json trong folder.
    Returns: { paper_id: data_dict }
    """
    results = {}
    if not os.path.isdir(folder):
        return results
    for fname in os.listdir(folder):
        if not fname.endswith(".json"):
            continue
        fpath = os.path.join(folder, fname)
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
            pid = data.get("paper_id", fname[:-5])
            results[pid] = data
        except Exception:
            pass
    return results


def get_llm_review_key(data: dict, source_name: str) -> list:
    """
    Lấy list arguments của LLM từ reviews_analysis.
    Ưu tiên: source_name.upper() → phần model (trước dấu _) → key đầu tiên không phải Human.
    """
    ra = data.get("reviews_analysis", {})
    if not ra:
        return []

    # 1. Thử key chính xác: SOURCE_NAME.upper()
    key_exact = source_name.upper()
    if key_exact in ra:
        return ra[key_exact]

    # 2. Thử lấy phần model (phần trước dấu _ đầu tiên, uppercase)
    #    VD: sea_iclr2024 → SEA
    model_part = source_name.split("_")[0].upper()
    if model_part in ra:
        return ra[model_part]

    # 3. Lấy key đầu tiên không phải Human
    for k, v in ra.items():
        if not k.startswith("Human"):
            return v

    return []


# ================================================================
#  Tính metric cho Human (standalone, per conference)
# ================================================================

def compute_human_metrics(conference: str, folder: str,
                          filter_ids: set = None) -> pd.DataFrame:
    """
    Tính metric cho Human reviews của 1 hội nghị.
    Mỗi paper: tính từng reviewer rồi lấy trung bình.
    """
    results = load_json_folder(folder)
    if not results:
        print(f"  [!] Không có dữ liệu tại: {folder}")
        return pd.DataFrame()

    if filter_ids:
        results = {k: v for k, v in results.items() if k in filter_ids}

    rows = []
    for paper_id in sorted(results.keys()):
        data = results[paper_id]
        ra   = data.get("reviews_analysis", {})

        per_reviewer = [
            calculate_review_metrics(args)
            for rev_id, args in ra.items()
            if rev_id.startswith("Human") and isinstance(args, list)
        ]
        if not per_reviewer:
            continue

        mean_m = pd.DataFrame(per_reviewer).mean().to_dict()

        row = {
            "paper_id":       paper_id,
            "conference":     conference,
            "n_reviewers":    len(per_reviewer),
            "R_premise":      round(mean_m["r_premise"],      4),
            "S_depth":        round(mean_m["s_depth"],        4),
            "DoA_score":      round(mean_m["doa_score"],      4),
            "DoA_score_HM":   round(mean_m["doa_score_hm"],   4),
            "Total_Claims":   round(mean_m["total_claims"],   2),
            "Total_Premises": round(mean_m["total_premises"], 2),
        }
        for k, v in mean_m.items():
            if k.startswith("Ratio_"):
                row[k] = round(v, 4)
        rows.append(row)

    return pd.DataFrame(rows)


# ================================================================
#  Tính metric cho LLM (standalone, per source)
# ================================================================

def compute_llm_metrics(source_name: str, folder: str,
                        filter_ids: set = None) -> pd.DataFrame:
    """
    Tính metric cho LLM reviews của 1 source.
    Mỗi paper: 1 review duy nhất của LLM.
    """
    results = load_json_folder(folder)
    if not results:
        print(f"  [!] Không có dữ liệu tại: {folder}")
        return pd.DataFrame()

    if filter_ids:
        results = {k: v for k, v in results.items() if k in filter_ids}

    rows = []
    for paper_id in sorted(results.keys()):
        data     = results[paper_id]
        llm_args = get_llm_review_key(data, source_name)
        if not llm_args:
            continue

        m     = calculate_review_metrics(llm_args)
        usage = data.get("usage_stats", {})

        row = {
            "paper_id":       paper_id,
            "llm_source":     source_name,
            "Token_Total":    usage.get("total_tokens", 0),
            "R_premise":      round(m["r_premise"],      4),
            "S_depth":        round(m["s_depth"],        4),
            "DoA_score":      round(m["doa_score"],      4),
            "DoA_score_HM":   round(m["doa_score_hm"],   4),
            "Total_Claims":   m["total_claims"],
            "Total_Premises": m["total_premises"],
        }
        for k, v in m.items():
            if k.startswith("Ratio_"):
                row[k] = round(v, 4)
        rows.append(row)

    return pd.DataFrame(rows)


# ================================================================
#  Save & Print helpers
# ================================================================

def save_metrics(df: pd.DataFrame, name: str, label: str):
    """Lưu CSV chi tiết + CSV trung bình tổng thể. In tóm tắt."""
    os.makedirs(OUTPUT_METRICS_DIR, exist_ok=True)

    detail_path  = os.path.join(OUTPUT_METRICS_DIR, f"{name}_Metrics.csv")
    overall_path = os.path.join(OUTPUT_METRICS_DIR, f"{name}_Overall.csv")

    df.to_csv(detail_path, index=False)

    overall = df.mean(numeric_only=True)
    overall.to_frame(name="Overall_Mean").T.to_csv(overall_path, index=False)

    key_cols = ["R_premise", "S_depth", "DoA_score", "DoA_score_HM"]
    available = [c for c in key_cols if c in overall.index]

    print(f"\n  [{label}]  {len(df)} papers")
    print(f"  {'-' * 44}")
    for col in available:
        print(f"  {col:<20}: {overall[col]:.4f}")
    print(f"  → {detail_path}")
    print(f"  → {overall_path}")

    return overall


# ================================================================
#  Run helpers
# ================================================================

def run_all_human(filter_ids: set = None):
    """Tính metric cho tất cả hội nghị Human."""
    summary_rows = []
    print(f"\n{'=' * 60}")
    print(f"[HUMAN] Tính metric độc lập cho TẤT CẢ hội nghị")
    print(f"{'=' * 60}")

    for conf, folder in HUMAN_CONFERENCES.items():
        if not os.path.isdir(folder):
            print(f"\n  [skip] {conf}: thư mục không tồn tại ({folder})")
            continue
        n_files = len([f for f in os.listdir(folder) if f.endswith(".json")])
        if n_files == 0:
            print(f"\n  [skip] {conf}: không có file JSON")
            continue

        print(f"\n  [Human / {conf}]  ({n_files} files)")
        df = compute_human_metrics(conf, folder, filter_ids=filter_ids)
        if df.empty:
            print(f"  [skip] {conf}: kết quả rỗng")
            continue

        overall = save_metrics(df, f"human_{conf}", f"Human / {conf}")

        row = {"source": f"human_{conf}", "type": "Human", "conference": conf,
               "n_papers": len(df)}
        for col in ["R_premise", "S_depth", "DoA_score", "DoA_score_HM"]:
            row[col] = round(overall[col], 4) if col in overall.index else None
        summary_rows.append(row)

    return summary_rows


def run_all_llm(filter_ids: set = None, target_sources: list = None):
    """Tính metric cho tất cả LLM sources."""
    summary_rows = []
    sources = target_sources or list(config.LLM_SOURCES.keys())

    print(f"\n{'=' * 60}")
    print(f"[LLM] Tính metric độc lập cho TẤT CẢ LLM sources ({len(sources)} sources)")
    print(f"{'=' * 60}")

    for source_name in sources:
        folder = config.get_llm_output_dir(source_name)
        if not os.path.isdir(folder):
            print(f"\n  [skip] {source_name}: thư mục output không tồn tại ({folder})")
            continue
        n_files = len([f for f in os.listdir(folder) if f.endswith(".json")])
        if n_files == 0:
            print(f"\n  [skip] {source_name}: không có file JSON")
            continue

        print(f"\n  [LLM / {source_name}]  ({n_files} files)")
        df = compute_llm_metrics(source_name, folder, filter_ids=filter_ids)
        if df.empty:
            print(f"  [skip] {source_name}: kết quả rỗng")
            continue

        # Xác định conference và model từ source_name
        parts = source_name.split("_", 1)
        model_name = parts[0]
        conference = parts[1] if len(parts) > 1 else source_name

        overall = save_metrics(df, source_name, f"LLM / {source_name}")

        row = {"source": source_name, "type": "LLM", "model": model_name,
               "conference": conference, "n_papers": len(df)}
        for col in ["R_premise", "S_depth", "DoA_score", "DoA_score_HM"]:
            row[col] = round(overall[col], 4) if col in overall.index else None
        summary_rows.append(row)

    return summary_rows


def save_summary(human_rows: list, llm_rows: list):
    """Lưu bảng tổng hợp cuối cùng."""
    os.makedirs(OUTPUT_METRICS_DIR, exist_ok=True)

    if human_rows:
        df_h = pd.DataFrame(human_rows)
        path = os.path.join(OUTPUT_METRICS_DIR, "ALL_Human_Summary.csv")
        df_h.to_csv(path, index=False)
        print(f"\n[>] Bảng tổng hợp Human: {path}")
        print(df_h.to_string(index=False))

    if llm_rows:
        df_l = pd.DataFrame(llm_rows)
        path = os.path.join(OUTPUT_METRICS_DIR, "ALL_LLM_Summary.csv")
        df_l.to_csv(path, index=False)
        print(f"\n[>] Bảng tổng hợp LLM  : {path}")
        print(df_l.to_string(index=False))

    # Kết hợp 2 bảng thành 1 bảng so sánh cross-conference (nếu có đủ cả 2)
    if human_rows and llm_rows:
        all_rows = human_rows + llm_rows
        df_all   = pd.DataFrame(all_rows)
        path_all = os.path.join(OUTPUT_METRICS_DIR, "ALL_Combined_Summary.csv")
        df_all.to_csv(path_all, index=False)
        print(f"\n[>] Bảng kết hợp tất cả: {path_all}")


# ================================================================
#  Entry Point
# ================================================================

if __name__ == "__main__":
    valid_sources = list(config.LLM_SOURCES.keys())
    valid_confs   = list(HUMAN_CONFERENCES.keys())

    parser = argparse.ArgumentParser(
        description="DoA Pipeline — Tính metric ĐỘC LẬP per Conference (Human & LLM)",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--mode", type=str, default="all",
        choices=["all", "human", "llm"],
        help=(
            "Chế độ tính:\n"
            "  all   — tính tất cả Human + LLM (mặc định)\n"
            "  human — chỉ tính Human\n"
            "  llm   — chỉ tính LLM\n"
        ),
    )
    parser.add_argument(
        "--source", type=str, default=None,
        metavar="SOURCE",
        help=(
            f"Tính riêng 1 LLM source. Hiện có:\n  {valid_sources}"
        ),
    )
    parser.add_argument(
        "--conference", type=str, default=None,
        metavar="CONF",
        help=(
            f"Tính riêng 1 hội nghị (Human + LLM của hội nghị đó). Hiện có:\n"
            f"  {valid_confs}"
        ),
    )
    parser.add_argument(
        "--filter", type=str, default=None, metavar="FILE",
        help=(
            "Đường dẫn tới file .txt chứa danh sách paper_id (1 ID/dòng).\n"
            "Nếu cung cấp, chỉ tính metric cho subset paper này.\n"
            "Ví dụ: --filter paper_ids_200.txt"
        ),
    )
    args = parser.parse_args()

    # -- Load filter IDs nếu có ----------------------------------
    filter_ids = None
    if args.filter:
        if not os.path.isfile(args.filter):
            print(f"[X] File filter không tồn tại: {args.filter}")
            sys.exit(1)
        with open(args.filter, "r", encoding="utf-8") as fh:
            filter_ids = {line.strip() for line in fh if line.strip()}
        print(f"\n[~] Filter: {len(filter_ids)} paper IDs từ '{args.filter}'")

    human_rows = []
    llm_rows   = []

    # ── Trường hợp: --source (chỉ 1 LLM source) ──────────────────
    if args.source:
        if args.source not in valid_sources:
            print(f"[X] Source '{args.source}' không hợp lệ.")
            print(f"   Các source hiện có: {valid_sources}")
            sys.exit(1)
        folder = config.get_llm_output_dir(args.source)
        print(f"\n{'=' * 60}")
        print(f"[LLM / {args.source}]")
        print(f"{'=' * 60}")
        df = compute_llm_metrics(args.source, folder, filter_ids=filter_ids)
        if not df.empty:
            parts = args.source.split("_", 1)
            overall = save_metrics(df, args.source, f"LLM / {args.source}")
            llm_rows.append({
                "source":     args.source,
                "type":       "LLM",
                "model":      parts[0],
                "conference": parts[1] if len(parts) > 1 else args.source,
                "n_papers":   len(df),
                **{col: round(overall[col], 4)
                   for col in ["R_premise","S_depth","DoA_score","DoA_score_HM"]
                   if col in overall.index}
            })

    # ── Trường hợp: --conference (human + llm của 1 hội nghị) ────
    elif args.conference:
        conf = args.conference.lower()

        # Human
        if conf in HUMAN_CONFERENCES:
            folder = HUMAN_CONFERENCES[conf]
            print(f"\n{'=' * 60}")
            print(f"[Human / {conf}]")
            print(f"{'=' * 60}")
            df = compute_human_metrics(conf, folder, filter_ids=filter_ids)
            if not df.empty:
                overall = save_metrics(df, f"human_{conf}", f"Human / {conf}")
                human_rows.append({
                    "source": f"human_{conf}", "type": "Human", "conference": conf,
                    "n_papers": len(df),
                    **{c: round(overall[c], 4)
                       for c in ["R_premise","S_depth","DoA_score","DoA_score_HM"]
                       if c in overall.index}
                })
        else:
            print(f"  [!] Không có cấu hình Human cho hội nghị '{conf}'")

        # LLM (tất cả sources có tên chứa conf)
        matching = [s for s in valid_sources if conf in s]
        if matching:
            llm_rows += run_all_llm(filter_ids=filter_ids, target_sources=matching)
        else:
            print(f"  [!] Không có LLM source nào cho hội nghị '{conf}'")

    # ── Trường hợp: --mode ────────────────────────────────────────
    else:
        if args.mode in ("all", "human"):
            human_rows = run_all_human(filter_ids=filter_ids)

        if args.mode in ("all", "llm"):
            llm_rows = run_all_llm(filter_ids=filter_ids)

    # ── In bảng tổng hợp ─────────────────────────────────────────
    if human_rows or llm_rows:
        print(f"\n{'=' * 60}")
        print(f"[DONE] Lưu bảng tổng hợp...")
        print(f"{'=' * 60}")
        save_summary(human_rows, llm_rows)
    else:
        print("\n[!] Không có kết quả nào được tính.")

    print(f"\n✅ Hoàn tất!")

