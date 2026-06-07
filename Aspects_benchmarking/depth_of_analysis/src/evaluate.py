import statistics
from typing import List, Dict, Any

def calculate_review_metrics(arguments: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculate metrics for a single review's arguments.
    
    Arguments format:
    [
        {
            "argument": "The paper...",
            "role": "Claim",
            "aspect": "Contribution",
            "grounding_score": None
        },
        ...
    ]
    """
    res = {
        "r_premise": 0.0,
        "s_depth": 0.0,
        "doa_score": 0.0,
        "doa_score_hm": 0.0,
        "total_claims": 0.0,
        "total_premises": 0.0
    }
    
    if not arguments:
        return res
        
    total_args = len(arguments)
    total_premises = sum(1 for a in arguments if a.get("role") == "Premise")
    
    r_premise = total_premises / total_args if total_args > 0 else 0.0
    
    premises_scores = [
        a.get("grounding_score")
        for a in arguments
        if a.get("role") == "Premise" and a.get("grounding_score") is not None
    ]
    
    s_depth = statistics.mean(premises_scores) if premises_scores else 0.0
    
    # DoA score (product) and DoA score HM (harmonic mean)
    # Grounding score average (s_depth) is in range [0, 2], so we normalize it by / 2.0 to [0, 1]
    normalized_depth = s_depth / 2.0
    doa_score = r_premise * normalized_depth
    
    if r_premise + normalized_depth > 0:
        doa_score_hm = 2 * r_premise * normalized_depth / (r_premise + normalized_depth)
    else:
        doa_score_hm = 0.0
        
    res.update({
        "r_premise": r_premise,
        "s_depth": s_depth,
        "doa_score": doa_score,
        "doa_score_hm": doa_score_hm,
        "total_premises": float(total_premises),
        "total_claims": float(total_args)
    })
    
    # Calculate aspect ratios
    aspects = [a.get("aspect") for a in arguments if a.get("aspect")]
    all_aspects = set(aspects)
    
    for asp in all_aspects:
        if not asp:
            continue
        # Ratio of all arguments
        ratio_all = sum(1 for a in arguments if a.get("aspect") == asp) / total_args
        res[f"Ratio_All_{asp}"] = ratio_all
        
        # Ratio of premise arguments
        premises_in_asp = sum(1 for a in arguments if a.get("role") == "Premise" and a.get("aspect") == asp)
        ratio_prem = premises_in_asp / total_premises if total_premises > 0 else 0.0
        res[f"Ratio_Prem_{asp}"] = ratio_prem
        res[f"Ratio_Prem__{asp}"] = ratio_prem
        
    return res
