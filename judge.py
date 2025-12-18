"""
Multi-Judge Evaluation System for Finance Agent

This module provides a comprehensive evaluation pipeline with specialized judges:
- FactJudge: Evaluates factual correctness using rubrics or semantic comparison
- SourceJudge: Validates source citations and their quality tiers
- EfficiencyJudge: Measures speed and cost efficiency vs human baseline
- RefusalJudge: Checks adversarial questions for proper refusal behavior
- CalibrationJudge: Measures confidence calibration vs actual performance

Usage:
    python judge.py --refs data/public.csv --preds results/predictions.json --out_json results/audit.json

For simple judging without the full audit pipeline, use the legacy functions:
    from judge import heuristic_match, call_openai_judge
"""

import argparse
import json
import os
import csv
import math
import re
from datetime import datetime
from typing import Dict, Tuple, Any, Optional
from dotenv import load_dotenv

from logger import get_logger

log = get_logger("judge")

try:
    from openai import OpenAI
    _OPENAI_OK = True
except Exception:
    _OPENAI_OK = False

load_dotenv()


# =============================================================================
# LEGACY FUNCTIONS (for backwards compatibility with green_agent integration)
# =============================================================================

def normalize_q(q: str) -> str:
    """Normalizes text by lowercasing and removing extra spaces."""
    return re.sub(r"\s+", " ", q.strip().lower())


def extract_final_answer(text: str) -> str:
    """Extract the final answer from agent response text."""
    if not isinstance(text, str): 
        return ""
    # Extract after FINAL ANSWER:
    m = re.search(r"FINAL ANSWER\s*:\s*(.*)", text, re.I | re.S)
    extracted = m.group(1).strip() if m else text.strip()
    
    # Remove sources JSON block if it exists at the end
    m2 = re.search(r"(.*)\s*(\{[\"']sources[\"']:.*)", extracted, re.S)
    if m2:
        return m2.group(1).strip()
    
    return extracted


# Heuristic matching
_NUM_RE = re.compile(
    r"(?P<sign>[-+])?\s*(?P<num>\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?)\s*(?P<Unit>billion|bn|million|m|thousand|k|trillion|tn|t)?",
    re.I,
)

UNIT_SCALE = {
    None: 1.0,
    "k": 1e3, "thousand": 1e3,
    "m": 1e6, "million": 1e6,
    "bn": 1e9, "billion": 1e9,
    "tn": 1e12, "t": 1e12, "trillion": 1e12,
}


def normalize_text(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[\$,]", "", s)
    s = re.sub(r"\s+", " ", s)
    return s


def extract_number_and_scale(s: str) -> Optional[float]:
    """Extract the first numeric figure and convert by unit word."""
    s_norm = s.lower()
    m = _NUM_RE.search(s_norm)
    if not m:
        return None
    raw = m.group("num")
    sign = -1 if (m.group("sign") == "-") else 1
    unit = (m.group("Unit") or "").lower() or None
    try:
        val = float(raw.replace(",", ""))
        scale = UNIT_SCALE.get(unit, 1.0)
        return sign * val * scale
    except Exception:
        return None


def heuristic_match(gt: str, pred: str, rel_tol=0.01, abs_tol=1e-4) -> Tuple[bool, str]:
    """
    Heuristic matching:
    1) Exact normalized string match
    2) Numeric match with unit scaling within tolerance
    3) Percentage-style loose check
    """
    gt_n = normalize_text(gt)
    pd_n = normalize_text(pred)

    if gt_n == pd_n:
        return True, "exact_normalized_match"
    
    gt_num = extract_number_and_scale(gt)
    pd_num = extract_number_and_scale(pred)
    if gt_num is not None and pd_num is not None:
        if math.isclose(gt_num, pd_num, rel_tol=rel_tol, abs_tol=abs_tol):
            return True, "numeric_within_tol"

    if "%" in gt_n and "%" in pd_n:
        gt_num2 = extract_number_and_scale(gt_n.replace("%",""))
        pd_num2 = extract_number_and_scale(pd_n.replace("%",""))
        if gt_num2 is not None and pd_num2 is not None:
            if math.isclose(gt_num2, pd_num2, rel_tol=rel_tol, abs_tol=abs_tol):
                return True, "percent_within_tol"

    return False, "no_heuristic_match"


def calculate_efficiency(actual_time: float, actual_tokens: int, expert_time_min: float) -> float:
    """
    Calculate efficiency score.
    efficiency = 0.5 * time_efficiency + 0.5 * token_efficiency
    """
    expert_time_sec = expert_time_min * 60
    if expert_time_sec <= 0: 
        expert_time_sec = 600  # Default 10 mins
    
    if actual_time <= 0: 
        actual_time = 1.0
    time_score = expert_time_sec / (actual_time + expert_time_sec)
    
    expert_tokens = expert_time_min * 500
    if expert_tokens <= 0: 
        expert_tokens = 5000
    
    if actual_tokens <= 0: 
        actual_tokens = 1
    token_score = expert_tokens / (actual_tokens + expert_tokens)
    
    return 0.5 * time_score + 0.5 * token_score


def check_attribution(text: str, raw_text: str) -> float:
    """Check if the answer contains sources or citations."""
    if "http://" in raw_text or "https://" in raw_text:
        return 1.0
    
    if "sources" in raw_text.lower() and ("{" in raw_text or "[" in raw_text):
        return 1.0
        
    return 0.0


JUDGE_PROMPT = """You are an impartial financial QA judge.

Decide if the model's FINAL ANSWER is factually equivalent to the ground truth FINAL ANSWER for the question below.

Rules:
- Judge only factual equivalence (ignore style and wording).
- Allow rounding/formatting differences up to ~1% for numeric values.
- If the prediction contains extra, non-conflicting context but is otherwise correct, mark it correct.
- Use the provided RUBRIC (if any) as a strict checklist.
- Output strict JSON: {{"correct": true|false, "explanation": "<short reason>"}}

Question:
{question}

Rubric:
{rubric}

Ground truth:
{ground}

Model answer:
{pred}
"""


def call_openai_judge(model: str, question: str, ground: str, pred: str, rubric: str = "") -> Tuple[bool, str]:
    """Call OpenAI to judge a response (legacy function for backwards compatibility)."""
    if not _OPENAI_OK:
        raise RuntimeError("OpenAI SDK not available. `pip install openai`")

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    prompt = JUDGE_PROMPT.format(question=question, ground=ground, pred=pred, rubric=rubric)

    resp = client.responses.create(
        model=model,
        input=[{"role":"user","content":prompt}],
        temperature=0,
        max_output_tokens=200,
    )
    text = (getattr(resp, "output_text", None) or "").strip()
    if not text and getattr(resp, "output", None):
        parts = []
        for item in resp.output:
            if getattr(item, "type", "") == "message":
                for ct in getattr(item, "content", []):
                    if getattr(ct, "type","") == "output_text":
                        parts.append(ct.text)
        text = "\n".join(parts).strip()

    try:
        obj = json.loads(text)
        correct = bool(obj.get("correct"))
        expl = str(obj.get("explanation") or "")
        return correct, expl
    except Exception:
        return False, f"Judge JSON parse failed: {text[:200]}"


# Legacy loaders (kept for backwards compatibility)
def load_refs(path: str) -> Dict[str, Dict[str, Any]]:
    """Load ground truth CSV."""
    refs = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            qid = (row.get("question_id") or "").strip()
            q = (row.get("question") or row.get("Question") or "").strip()
            ans = (row.get("answer") or row.get("Answer") or "").strip()
            rubric = (row.get("Rubric") or row.get("rubric") or "").strip()
            
            try:
                et_str = (row.get("Expert time (mins)") or "0").strip()
                expert_time = float(et_str) if et_str else 0.0
            except ValueError:
                expert_time = 0.0

            if not ans or not (qid or q):
                continue
            
            key = qid if qid else normalize_q(q)
            refs[key] = {
                "question": q, 
                "answer": ans, 
                "question_id": qid or "",
                "rubric": rubric,
                "expert_time": expert_time
            }
    return refs


# =============================================================================
# MULTI-JUDGE AUDIT PIPELINE (from andrew_test branch)
# =============================================================================

def run_full_audit(refs_path: str, preds_path: str, out_json: str, out_txt: str = None):
    """
    Run the full multi-judge audit pipeline.
    
    Args:
        refs_path: Path to ground truth CSV
        preds_path: Path to predictions JSON
        out_json: Path to save structured JSON results
        out_txt: Path to save human-readable report (optional)
    """
    from judges import SourceJudge, FactJudge, EfficiencyJudge, RefusalJudge, CalibrationJudge
    from judge_utils import load_refs as load_refs_util, load_preds
    
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    judges = {
        "source": SourceJudge(llm_client=client),
        "fact": FactJudge(llm_client=client),
        "efficiency": EfficiencyJudge(),
        "refusal": RefusalJudge(llm_client=client),
        "calibration": CalibrationJudge()
    }

    refs = load_refs_util(refs_path)
    preds = load_preds(preds_path)

    full_audit_log = []
    human_readable_buffer = []

    ADVERSARIAL_TRIGGERS = ["NOT_FOUND", "N/A", "UNKNOWN", "NO_DATA", "NOT DETERMINABLE"]

    print(f"Auditing {len(preds)} predictions...")

    for pkey, prow in preds.items():
        ref = refs.get(pkey)
        if not ref: 
            continue

        gt_answer = str(ref['answer']).upper().strip()
        is_adversarial = any(trigger in gt_answer for trigger in ADVERSARIAL_TRIGGERS)
        
        q_result = {
            "question_id": ref.get("question_id"),
            "question": ref["question"],
            "prediction": prow["answer"],
            "ground_truth": ref["answer"],
            "pipeline": "Adversarial" if is_adversarial else "Standard",
            "judges": {},
            "composite_score": 0.0
        }
        
        current_fact_score = 0.0

        if is_adversarial:
            print(f"  > [Adversarial] Auditing {ref.get('question_id')}...")
            
            r_res = judges["refusal"].evaluate(ref, prow)
            q_result["judges"]["refusal"] = r_res
            
            current_fact_score = r_res['score']
            
            s_res = judges["source"].evaluate(ref, prow, is_adversarial=True)
            q_result["judges"]["source"] = s_res

        else:
            print(f"  > [Standard] Auditing {ref.get('question_id')}...")

            f_res = judges["fact"].evaluate(ref, prow)
            q_result["judges"]["fact"] = f_res
            
            current_fact_score = f_res['score']
            
            s_res = judges["source"].evaluate(ref, prow, is_adversarial=False)
            q_result["judges"]["source"] = s_res

        e_res = judges["efficiency"].evaluate(ref, prow)
        q_result["judges"]["efficiency"] = e_res
        
        c_res = judges["calibration"].evaluate(ref, prow, current_fact_score)
        q_result["judges"]["calibration"] = c_res

        # Composite score: 50% fact, 30% source, 10% efficiency, 10% calibration
        total_score = (
            0.5 * current_fact_score +
            0.3 * s_res['score'] +
            0.1 * e_res['score'] +
            0.1 * c_res['score']
        )
        
        q_result["composite_score"] = round(total_score, 2)
        full_audit_log.append(q_result)

        # Build human-readable report segment
        primary_render = (
            judges["refusal"].render(r_res) if is_adversarial 
            else judges["fact"].render(f_res)
        )

        report_segment = f"""
##################################################
QUESTION ID: {ref.get('question_id')}
PIPELINE:    {q_result['pipeline'].upper()}
SCORE:       {q_result['composite_score']}/100
##################################################
Q: {ref['question']}
A: {prow['answer']}

{primary_render}
{judges['source'].render(s_res)}
{judges['calibration'].render(c_res)}
{judges['efficiency'].render(e_res)}
--------------------------------------------------
"""
        human_readable_buffer.append(report_segment)

    # Save results
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(full_audit_log, f, indent=2)
    
    if out_txt:
        with open(out_txt, "w", encoding="utf-8") as f:
            f.write("\n".join(human_readable_buffer))

    # Calculate summary stats
    if full_audit_log:
        avg_score = sum(r["composite_score"] for r in full_audit_log) / len(full_audit_log)
        correct_count = sum(1 for r in full_audit_log if r["composite_score"] >= 70)
    else:
        avg_score = 0
        correct_count = 0

    print(f"\nAudit Complete.")
    print(f"Total: {len(full_audit_log)}  Passing (>=70): {correct_count}  Avg Score: {avg_score:.1f}")
    print(f"Structured Data: {out_json}")
    if out_txt:
        print(f"Human Report:    {out_txt}")
    
    return full_audit_log


def main():
    ap = argparse.ArgumentParser(description="Multi-Judge Evaluation for Finance Agent")
    ap.add_argument("--refs", required=True, help="Path to ground truth CSV")
    ap.add_argument("--preds", required=True, help="Path to agent predictions JSON")
    ap.add_argument("--out_json", default="results/audit.json", help="Path to save structured JSON")
    ap.add_argument("--out_txt", default="results/audit_report.txt", help="Path to save human readable report")
    ap.add_argument("--legacy", action="store_true", help="Use legacy simple judge instead of multi-judge pipeline")
    args = ap.parse_args()

    if args.legacy:
        # Legacy simple judging mode
        from judge_utils import load_preds
        
        refs = load_refs(args.refs)
        preds = load_preds(args.preds)
        
        qtext_to_key = {}
        for key, row in refs.items():
            if row["question"]:
                qtext_to_key[normalize_q(row["question"])] = key

        scored_rows = []
        correct_count = 0
        total = 0

        for pkey, prow in preds.items():
            total += 1
            ref = refs.get(pkey)
            if not ref and prow["question"]:
                ref = refs.get(qtext_to_key.get(normalize_q(prow["question"])))

            if not ref:
                scored_rows.append({
                    "question_id": prow.get("question_id",""),
                    "question": prow.get("question",""),
                    "pred_answer": prow["answer"],
                    "gt_answer": "",
                    "correct": 0,
                    "method": "not_found",
                    "explanation": "No matching reference question_id or question text.",
                })
                continue

            gt = ref["answer"]
            pred = prow["answer"]
            rubric = ref.get("rubric", "")

            is_ok, reason = heuristic_match(gt, pred)
            method = "heuristic"
            explanation = reason
            correct = is_ok

            if not is_ok:
                try:
                    j_ok, j_expl = call_openai_judge("gpt-4o", ref["question"], gt, pred, rubric)
                    correct = j_ok
                    method = "llm_judge"
                    explanation = j_expl or "LLM judge decision"
                except Exception as e:
                    method = "judge_error"
                    explanation = f"Judge error: {e}"

            correct_count += int(bool(correct))
            scored_rows.append({
                "question_id": ref.get("question_id",""),
                "question": ref["question"],
                "pred_answer": pred,
                "gt_answer": gt,
                "correct": 1 if correct else 0,
                "method": method,
                "explanation": explanation,
            })

        acc = (correct_count / total) * 100 if total else 0.0
        print(f"Total: {total}  Correct: {correct_count}  Accuracy: {acc:.2f}%")

        os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
        with open(args.out_json.replace('.json', '_legacy.csv'), "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=[
                "question_id","question","pred_answer","gt_answer","correct","method","explanation"
            ])
            w.writeheader()
            w.writerows(scored_rows)
    else:
        # Full multi-judge audit pipeline
        run_full_audit(args.refs, args.preds, args.out_json, args.out_txt)


if __name__ == "__main__":
    main()
