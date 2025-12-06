import argparse, csv, json, math, os, re, sys
from collections import defaultdict
from typing import Dict, Tuple, Any, Optional
from logger import get_logger

log = get_logger("judge")

try:
    from openai import OpenAI
    _OPENAI_OK = True
except Exception:
    _OPENAI_OK = False

def load_refs(path: str) -> Dict[str, Dict[str, Any]]:
    """
    Load the ground truth CSV. Expected columns:
      - question_id (recommended)
      - question       (required if no question_id)
      - answer        (required)
      - Rubric        (optional)
      - Expert time (mins) (optional)
    Returns dict keyed by question_id if present, else by normalized question text.
    """
    refs = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            qid = (row.get("question_id") or "").strip()
            q   = (row.get("question") or row.get("Question") or "").strip()
            ans = (row.get("answer")   or row.get("Answer") or "").strip()
            rubric = (row.get("Rubric") or "").strip()
            expert_time_str = (row.get("Expert time (mins)") or "").strip()
            try:
                expert_time = float(expert_time_str) if expert_time_str else 0.0
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

def load_preds(path: str) -> Dict[str, Dict[str, Any]]:
    """
    Load predictions from JSON/JSONL/CSV.
    Returns dict keyed by question_id/normalized_q.
    Values include: answer, token_usage, duration, raw_content
    """
    preds = {}
    ext = os.path.splitext(path)[1].lower()

    def _append(obj: Dict[str, Any]):
        if not isinstance(obj, dict):
            return
        qid = (obj.get("question_id") or "").strip()
        q   = (obj.get("question") or "").strip()
        
        # Extract metadata if available
        # Structure in results: "result": ["answer", metadata_dict] or "result": "answer" or dict
        ans = None
        metadata = {}
        
        res = obj.get("result")
        # Check if result is [answer, metadata] tuple/list
        if isinstance(res, list) and len(res) > 0:
            ans = res[0]
            if len(res) > 1 and isinstance(res[1], dict):
                metadata = res[1]
        elif isinstance(res, str):
            ans = res
        elif isinstance(res, dict):
            # Sometimes result is a dict with answer?
            if "final_answer" in res:
                ans = res["final_answer"]
                metadata = res # Assuming metadata is merged or in the same dict
            else:
                # Fallback
                ans = str(res)
        else:
            ans = first_not_none(
                obj.get("final_answer"),
                obj.get("answer"),
                obj.get("model_answer"),
            )
            # Try to find metadata in the object itself
            if "total_tokens" in obj:
                 metadata["total_tokens"] = obj["total_tokens"]
            if "total_duration_seconds" in obj:
                 metadata["total_duration_seconds"] = obj["total_duration_seconds"]

        # Parse output if raw text
        raw_ans = ""
        if isinstance(ans, str):
             raw_ans = ans
             ans_text = extract_final_answer(ans)
        else:
             ans_text = ""

        # Extract stats
        tokens = 0
        duration = 0.0
        
        # Metadata format from agent.py: "total_tokens": { "prompt_tokens": ..., "total_tokens": ... }
        if "total_tokens" in metadata:
            tt = metadata["total_tokens"]
            if isinstance(tt, dict):
                tokens = tt.get("total_tokens", 0)
            elif isinstance(tt, (int, float)):
                tokens = tt
        
        if "total_duration_seconds" in metadata:
            duration = metadata["total_duration_seconds"] or 0.0
        elif "duration_seconds" in metadata:
            duration = metadata["duration_seconds"] or 0.0

        key = qid if qid else normalize_q(q)
        preds[key] = {
            "question": q, 
            "answer": ans_text, 
            "question_id": qid or "",
            "tokens": tokens,
            "duration": duration,
            "raw_answer": raw_ans
        }

    if ext in [".json", ".ndjson"]:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                for obj in data:
                    _append(obj)
            elif isinstance(data, dict):
                _append(data)
    elif ext in [".jsonl", ".ndjsonl"]:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        _append(json.loads(line))
                    except: pass
    elif ext == ".csv":
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                _append(row)
    
    return preds

def first_not_none(*vals):
    for v in vals:
        if v is not None and v != "":
            return v
    return None

def normalize_q(q: str) -> str:
    return re.sub(r"\s+", " ", q.strip().lower())

def extract_final_answer(text: str) -> str:
    if not isinstance(text, str): return ""
    # Extract after FINAL ANSWER:
    m = re.search(r"FINAL ANSWER\s*:\s*(.*)", text, re.I | re.S)
    extracted = m.group(1).strip() if m else text.strip()
    
    # Remove sources JSON block if it exists at the end
    # Often formatted as ```json ... ``` or just { "sources": ... }
    # Simple heuristic: cut off at last occurrences of {"sources" or similar
    # But let's be careful not to cut real content.
    # The agent output usually appends sources at the end.
    
    # Look for { "sources": ... } or { "sources": ... } at the end
    m2 = re.search(r"(.*)\s*(\{[\"']sources[\"']:.*)", extracted, re.S)
    if m2:
        return m2.group(1).strip()
    
    return extracted

# heuristic judge
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
    time_efficiency = expert_time / (actual_time + expert_time)
    token_efficiency = expected_tokens / (actual_tokens + expected_tokens)
    """
    expert_time_sec = expert_time_min * 60
    if expert_time_sec <= 0: expert_time_sec = 600 # Default 10 mins
    
    # Time score (using soft ratio)
    if actual_time <= 0: actual_time = 1.0 # Avoid div by zero
    time_score = expert_time_sec / (actual_time + expert_time_sec)
    
    # Token score
    # Estimate expert tokens based on time (e.g. 500 tokens/min)
    expert_tokens = expert_time_min * 500
    if expert_tokens <= 0: expert_tokens = 5000
    
    if actual_tokens <= 0: actual_tokens = 1 # Avoid div by zero
    token_score = expert_tokens / (actual_tokens + expert_tokens)
    
    return 0.5 * time_score + 0.5 * token_score

def check_attribution(text: str, raw_text: str) -> float:
    """
    Check if the answer contains sources or citations.
    Returns 1.0 if sources present, 0.0 otherwise.
    """
    # Check for "http://" or "https://"
    if "http://" in raw_text or "https://" in raw_text:
        return 1.0
    
    # Check for sources block in raw text
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

def call_openai_judge(model: str, question: str, ground: str, pred: str, rubric: str) -> Tuple[bool, str]:
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
        expl    = str(obj.get("explanation") or "")
        return correct, expl
    except Exception:
        return False, f"Judge JSON parse failed: {text[:200]}"

def main():
    ap = argparse.ArgumentParser(description="LLM judge for Finance Agent outputs")
    ap.add_argument("--refs", required=True, help="Path to ground-truth CSV (e.g., data/public.csv)")
    ap.add_argument("--preds", required=True, help="Path to predictions file (json/jsonl/csv)")
    ap.add_argument("--out",   required=True, help="Where to write scored CSV")
    ap.add_argument("--no_llm", action="store_true", help="Disable LLM judge; heuristics only")
    ap.add_argument("--judge_model", default="openai/gpt-5",
                    help="OpenAI model id for judging (Responses API). Works with GPT-4o, GPT-5, etc.")
    args = ap.parse_args()

    refs  = load_refs(args.refs)
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
                "efficiency": 0.0,
                "attribution": 0.0,
                "composite_score": 0.0
            })
            continue

        gt = ref["answer"]
        pred = prow["answer"]
        rubric = ref.get("rubric", "")

        is_ok, reason = heuristic_match(gt, pred)
        method = "heuristic"
        explanation = reason
        correct = is_ok

        if not is_ok and not args.no_llm:
            try:
                j_ok, j_expl = call_openai_judge(args.judge_model, ref["question"], gt, pred, rubric)
                correct = j_ok
                method = "llm_judge"
                explanation = j_expl or "LLM judge decision"
            except Exception as e:
                method = "judge_error"
                explanation = f"Judge error: {e}"

        # Calculate metrics
        eff_score = calculate_efficiency(prow["duration"], prow["tokens"], ref["expert_time"])
        attr_score = check_attribution(pred, prow["raw_answer"])
        
        # Composite score: 0.7 * correct + 0.2 * efficiency + 0.1 * attribution
        comp_score = 0.7 * float(correct) + 0.2 * eff_score + 0.1 * attr_score

        correct_count += int(bool(correct))
        scored_rows.append({
            "question_id": ref.get("question_id",""),
            "question": ref["question"],
            "pred_answer": pred,
            "gt_answer": gt,
            "correct": 1 if correct else 0,
            "method": method,
            "explanation": explanation,
            "efficiency": round(eff_score, 3),
            "attribution": attr_score,
            "composite_score": round(comp_score, 3)
        })

    acc = (correct_count / total) * 100 if total else 0.0
    print(f"Total: {total}  Correct: {correct_count}  Accuracy: {acc:.2f}%")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "question_id","question","pred_answer","gt_answer","correct","method","explanation","efficiency","attribution","composite_score"
        ])
        w.writeheader()
        w.writerows(scored_rows)
    
    log.info(f"Total={total} Correct={correct_count} Accuracy={acc:.2f}%")

if __name__ == "__main__":
    main()
