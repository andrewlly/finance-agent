import csv
import json
import math
import os
import re
from typing import Dict, Any, Optional

def normalize_q(q: str) -> str:
    """Normalizes text by lowercasing and removing extra spaces."""
    return re.sub(r"\s+", " ", q.strip().lower())

def extract_number(text: str) -> Optional[float]:
    """Simple number extractor for heuristic matching."""
    match = re.search(r"([\d,]+\.?\d*)", text)
    if match:
        try:
            return float(match.group(1).replace(",", ""))
        except:
            return None
    return None

def heuristic_match(gt: str, pred: str) -> bool:
    """Basic check: do the numbers match? (5% tolerance)"""
    gt_num = extract_number(gt)
    pred_num = extract_number(pred)
    
    if gt_num is not None and pred_num is not None:
        if math.isclose(gt_num, pred_num, rel_tol=0.05): 
            return True

    return normalize_q(gt) in normalize_q(pred)


def load_refs(path: str) -> Dict[str, Dict[str, Any]]:
    """Load ground truth CSV."""
    refs = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            qid = (row.get("question_id") or "").strip()
            q = (row.get("question") or row.get("Question") or "").strip()
            ans = (row.get("answer") or row.get("Answer") or "").strip()
            tier_req = (row.get("min_tier") or "3").strip()
            
            try:
                et_str = (row.get("Expert time (mins)") or "0").strip()
                expert_time = float(et_str)
            except ValueError:
                expert_time = 10.0 

            if not ans or not (qid or q):
                continue
            
            key = qid if qid else normalize_q(q)
            refs[key] = {
                "question": q, 
                "answer": ans, 
                "question_id": qid,
                "min_tier": tier_req,
                "expert_time": expert_time
            }
    return refs

def load_preds(path: str) -> Dict[str, Dict[str, Any]]:
    """
    Load predictions JSON. 
    Crucial: Preserves the 'logs' (metadata) for the Source Judge.
    """
    preds = {}
    if not os.path.exists(path):
        raise FileNotFoundError(f"Prediction file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict): data = [data]
    
    for obj in data:
        q = obj.get("question", "")
        qid = obj.get("question_id", "")

        res_list = obj.get("result", [])
        ans_text = ""
        meta = {}
        
        if isinstance(res_list, list) and len(res_list) > 0:
            ans_text = res_list[0]
            if len(res_list) > 1:
                meta = res_list[1]
        elif isinstance(res_list, str):
            ans_text = res_list
        
        clean_ans = re.sub(r"FINAL ANSWER:\s*", "", ans_text, flags=re.IGNORECASE)
        clean_ans = re.sub(r"\{.*\"sources\".*\}", "", clean_ans, flags=re.DOTALL).strip()
        
        tokens = 0
        if "total_tokens" in meta:
            tt = meta["total_tokens"]
            if isinstance(tt, dict):
                tokens = tt.get("total_input_tokens", 0) + tt.get("total_output_tokens", 0)
            else:
                tokens = int(tt)
                
        duration = meta.get("total_duration_seconds", 0.0)

        key = qid if qid else normalize_q(q)
        
        preds[key] = {
            "question": q,
            "answer": clean_ans,
            "final_answer": ans_text, 
            "tokens": tokens,
            "duration": duration,
            "logs": meta
        }
        
    return preds