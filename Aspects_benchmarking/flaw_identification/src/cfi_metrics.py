from typing import Dict, List


SEVERITY_POINTS = {
    "Critical": 2.0,
    "Minor": 1.0,
    "None": 0.0,
}


class DecoupledMetricsCalculator:
    """
    Decoupled CFI metrics with 4 recall-style outputs in [0, 1]:

      - Critical_Recall
      - Minor_Recall
      - Critical_Recall_ConsensusWeighted
      - Minor_Recall_ConsensusWeighted

    Consensus weight for flaw i:
      W_i = C_i / N
      C_i = number of reviewers who mentioned flaw i
      N   = total reviewers for the paper
    """

    def __init__(self, micro_flaws_json: dict, evaluations_json: dict, total_reviewers_count: int):
        self.flaws = micro_flaws_json.get("micro_flaws", [])
        self.evals = evaluations_json.get("evaluations", {})
        self.N = total_reviewers_count
        self.flaw_weights: Dict[str, Dict[str, float | str | int]] = {}
        self.valid_flaw_ids = set()
        self.critical_flaw_ids = set()
        self.minor_flaw_ids = set()
        self.total_critical_weight = 0.0
        self.total_minor_weight = 0.0

        for flaw in self.flaws:
            flaw_id = flaw.get("flaw_id")
            if not flaw_id:
                continue

            eval_data = self.evals.get(flaw_id, {})
            if not eval_data.get("is_valid"):
                continue

            severity = eval_data.get("severity")
            if severity not in {"Critical", "Minor"}:
                continue

            raw_args = flaw.get("raw_arguments", {})
            consensus_count = sum(1 for quote in raw_args.values() if quote and str(quote).strip())
            weight = consensus_count / self.N if self.N > 0 else 0.0
            self.flaw_weights[flaw_id] = {
                "weight": weight,
                "severity": severity,
                "consensus_count": consensus_count,
            }
            self.valid_flaw_ids.add(flaw_id)

            if severity == "Critical":
                self.critical_flaw_ids.add(flaw_id)
                self.total_critical_weight += weight
            elif severity == "Minor":
                self.minor_flaw_ids.add(flaw_id)
                self.total_minor_weight += weight

    def get_reviewer_flaws(self, reviewer_id: str) -> set:
        detected_flaws = set()
        for flaw in self.flaws:
            flaw_id = flaw.get("flaw_id")
            if not flaw_id:
                continue
            raw_args = flaw.get("raw_arguments", {})
            for key in raw_args.keys():
                lowered = key.lower()
                if reviewer_id == "LLM_Reviewer" and ("llm" in lowered or "sea" in lowered):
                    detected_flaws.add(flaw_id)
                    break
                if reviewer_id.startswith("Human_"):
                    human_num = reviewer_id.split("_")[1]
                    if human_num in key:
                        detected_flaws.add(flaw_id)
                        break
        return detected_flaws

    @staticmethod
    def _safe_ratio(numerator: float, denominator: float) -> float | None:
        if denominator <= 0:
            return None
        return round(numerator / denominator, 4)

    @staticmethod
    def _sort_value(value: float | None) -> float:
        return -1.0 if value is None else value

    def calculate_reviewer_scores(self, reviewer_id: str) -> dict:
        detected_flaws = self.get_reviewer_flaws(reviewer_id).intersection(self.valid_flaw_ids)
        detected_critical = detected_flaws.intersection(self.critical_flaw_ids)
        detected_minor = detected_flaws.intersection(self.minor_flaw_ids)

        detected_critical_weight = sum(float(self.flaw_weights[flaw_id]["weight"]) for flaw_id in detected_critical)
        detected_minor_weight = sum(float(self.flaw_weights[flaw_id]["weight"]) for flaw_id in detected_minor)

        critical_recall = self._safe_ratio(len(detected_critical), len(self.critical_flaw_ids))
        minor_recall = self._safe_ratio(len(detected_minor), len(self.minor_flaw_ids))
        critical_recall_weighted = self._safe_ratio(detected_critical_weight, self.total_critical_weight)
        minor_recall_weighted = self._safe_ratio(detected_minor_weight, self.total_minor_weight)

        return {
            "Reviewer_ID": reviewer_id,
            "Critical_Recall": critical_recall,
            "Minor_Recall": minor_recall,
            "Critical_Recall_ConsensusWeighted": critical_recall_weighted,
            "Minor_Recall_ConsensusWeighted": minor_recall_weighted,
            "Detected_Critical_Flaws": len(detected_critical),
            "Detected_Minor_Flaws": len(detected_minor),
            "Detected_Critical_ConsensusWeight": round(detected_critical_weight, 4),
            "Detected_Minor_ConsensusWeight": round(detected_minor_weight, 4),
            "Total_Valid_Flaws_Found": len(detected_flaws),
        }

    def generate_final_report(self, human_ids: List[str]) -> dict:
        report = {
            "Severity_Points": SEVERITY_POINTS,
            "Flaw_Weights_Summary": self.flaw_weights,
            "Ground_Truth_Summary": {
                "Total_Valid_Flaws": len(self.valid_flaw_ids),
                "Total_Critical_Flaws": len(self.critical_flaw_ids),
                "Total_Minor_Flaws": len(self.minor_flaw_ids),
                "Total_Critical_ConsensusWeight": round(self.total_critical_weight, 4),
                "Total_Minor_ConsensusWeight": round(self.total_minor_weight, 4),
            },
            "Reviewer_Rankings": [],
        }
        report["Reviewer_Rankings"].append(self.calculate_reviewer_scores("LLM_Reviewer"))
        for h_id in human_ids:
            report["Reviewer_Rankings"].append(self.calculate_reviewer_scores(h_id))
        report["Reviewer_Rankings"].sort(
            key=lambda row: (
                self._sort_value(row["Critical_Recall_ConsensusWeighted"]),
                self._sort_value(row["Minor_Recall_ConsensusWeighted"]),
                self._sort_value(row["Critical_Recall"]),
                self._sort_value(row["Minor_Recall"]),
                row["Detected_Critical_Flaws"],
                row["Detected_Minor_Flaws"],
            ),
            reverse=True,
        )
        return report
