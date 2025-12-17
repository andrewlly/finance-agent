import argparse
import csv
import json
import math
import os
import re
import sys
from collections import defaultdict
from typing import Dict, Tuple, Any, Optional
from dotenv import load_dotenv
load_dotenv()

try:
    from openai import OpenAI
except ImportError:
    print("Error: OpenAI SDK not installed. Run `pip install openai`")
    sys.exit(1)

# Import your custom judge
# Ensure you have the file: judges/source_judge.py
from judges.source_judge import SourceJudge

# --- HELPER FUNCTIONS ---

def normalize_q(q: str) -> str:
    return re.sub(r"\s+", " ", q.strip().lower())

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
            
            # Parse expert time
            try:
                et_str = (row.get("Expert time (mins)") or "0").strip()
                expert_time = float(et_str)
            except ValueError:
                expert_time = 10.0 # Default

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
    """Load predictions JSON."""
    preds = {}
    if not os.path.exists(path):
        raise FileNotFoundError(f"Prediction file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Flatten list if needed
    if isinstance(data, dict): data = [data]
    
    for obj in data:
        # 1. Basic Extraction
        q = obj.get("question", "")
        qid = obj.get("question_id", "")
        
        # 2. Extract Result & Metadata
        # Structure is often ["Answer", {metadata}]
        res_list = obj.get("result", [])
        ans_text = ""
        meta = {}
        
        if isinstance(res_list, list) and len(res_list) > 0:
            ans_text = res_list[0]
            if len(res_list) > 1:
                meta = res_list[1]
        elif isinstance(res_list, str):
            ans_text = res_list
        
        # 3. Clean Answer
        # Remove the JSON source block from the text for comparison
        clean_ans = re.sub(r"FINAL ANSWER:\s*", "", ans_text, flags=re.IGNORECASE)
        clean_ans = re.sub(r"\{.*\"sources\".*\}", "", clean_ans, flags=re.DOTALL).strip()

        # 4. Extract Metrics
        tokens = 0
        duration = 0.0
        
        if "total_tokens" in meta:
            tt = meta["total_tokens"]
            if isinstance(tt, dict):
                tokens = tt.get("total_input_tokens", 0) + tt.get("total_output_tokens", 0)
            else:
                tokens = int(tt)
                
        duration = meta.get("total_duration_seconds", 0.0)

        # 5. Store Data (Critical: Store 'logs'/'turns' for the Judge)
        key = qid if qid else normalize_q(q)
        
        # We construct the object expected by SourceJudge
        preds[key] = {
            "question": q,
            "answer": clean_ans,
            "final_answer": ans_text, # Contains the sources JSON
            "tokens": tokens,
            "duration": duration,
            "logs": meta # This contains 'turns' and 'tool_calls'
        }
        
    return preds

def extract_number(text: str) -> Optional[float]:
    """Simple number extractor for heuristic matching."""
    # Matches 10.5, 1,000, 70 (bps)
    match = re.search(r"([\d,]+\.?\d*)", text)
    if match:
        try:
            return float(match.group(1).replace(",", ""))
        except:
            return None
    return None

def heuristic_match(gt: str, pred: str) -> bool:
    """Basic check: do the numbers match?"""
    gt_num = extract_number(gt)
    pred_num = extract_number(pred)
    
    # 1. Exact Number Match (within margin)
    if gt_num is not None and pred_num is not None:
        if math.isclose(gt_num, pred_num, rel_tol=0.05): # 5% tolerance
            return True
            
    # 2. Text Match
    return normalize_q(gt) in normalize_q(pred)

def calculate_efficiency(actual_time, actual_tokens, expert_time_min):
    """
    Score from 0.0 to 1.0 based on expert baseline.
    """
    expert_time_sec = expert_time_min * 60
    if expert_time_sec == 0: expert_time_sec = 300 # 5 min default

    # Time Score (Faster is better)
    # If Agent takes double expert time, score drops significantly
    time_score = min(1.0, expert_time_sec / max(1.0, actual_time))
    
    return time_score

# --- MAIN EXECUTION ---

def main():
    ap = argparse.ArgumentParser(description="Green Finance Agent Judge")
    ap.add_argument("--refs", required=True, help="Path to reference CSV")
    ap.add_argument("--preds", required=True, help="Path to agent result JSON")
    ap.add_argument("--out", required=True, help="Output CSV path")
    args = ap.parse_args()

    # 1. Setup OpenAI for the Source Judge
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("WARNING: OPENAI_API_KEY not found. Validity checks will default to 0.0.")
    client = OpenAI(api_key=api_key) if api_key else None
    
    # 2. Initialize Judges
    source_evaluator = SourceJudge(llm_client=client)

    # 3. Load Data
    print(f"Loading references from {args.refs}...")
    refs = load_refs(args.refs)
    print(f"Loading predictions from {args.preds}...")
    preds = load_preds(args.preds)

    scored_rows = []

    print("\nStarting Evaluation...")
    print("-" * 50)

    for pkey, prow in preds.items():
        # Match Prediction to Reference
        ref = refs.get(pkey)
        
        # Fallback: try matching normalized question text
        if not ref:
            q_norm = normalize_q(prow["question"])
            for rk, rv in refs.items():
                if normalize_q(rv["question"]) == q_norm:
                    ref = rv
                    break
        
        if not ref:
            print(f"[SKIP] No reference found for question: {prow['question'][:30]}...")
            continue

        print(f"Evaluating Q: {ref['question'][:40]}...")

        # --- A. FACTUAL EVALUATION (Heuristic) ---
        is_correct = heuristic_match(ref["answer"], prow["answer"])
        fact_score = 100.0 if is_correct else 0.0

        # --- B. SOURCE EVALUATION (The Green Audit) ---
        # This calls your custom judges/source_judge.py logic
        source_result = source_evaluator.evaluate(ref, prow)
        source_score = source_result['score']
        audit_report = source_result['report_text']

        # --- C. EFFICIENCY EVALUATION ---
        eff_score = calculate_efficiency(prow["duration"], prow["tokens"], ref["expert_time"]) * 100

        # --- D. FINAL COMPOSITE SCORE ---
        # Weights: 40% Fact, 40% Source, 20% Efficiency
        final_score = (0.4 * fact_score) + (0.4 * source_score) + (0.2 * eff_score)

        # Store Result
        scored_rows.append({
            "question_id": ref["question_id"],
            "question": ref["question"],
            "pred_answer": prow["answer"],
            "gt_answer": ref["answer"],
            
            # Detailed Scores
            "fact_score": fact_score,
            "source_score": round(source_score, 1),
            "efficiency_score": round(eff_score, 1),
            "final_score": round(final_score, 1),
            
            # The Full Audit Text
            "audit_report": audit_report
        })

    # 4. Save Output
    headers = [
        "question_id", "question", "pred_answer", "gt_answer",
        "fact_score", "source_score", "efficiency_score", "final_score",
        "audit_report"
    ]
    
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        w.writerows(scored_rows)

    print("-" * 50)
    print(f"Evaluation Complete. Processed {len(scored_rows)} rows.")
    print(f"Results saved to: {args.out}")

if __name__ == "__main__":
    main()