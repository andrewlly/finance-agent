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

def load_refs(path: str) -> Dict[str, Dict[str, str]]:
    """
    Load the ground truth CSV. Expected columns:
      - question_id (recommended)
      - question       (required if no question_id)
      - answer        (required)
    Returns dict keyed by question_id if present, else by normalized question text.
    """
    refs = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            qid = (row.get("question_id") or "").strip()
            q   = (row.get("question") or "").strip()
            ans = (row.get("answer")   or "").strip()
            if not ans or not (qid or q):
                continue
            key = qid if qid else normalize_q(q)
            refs[key] = {"question": q, "answer": ans, "question_id": qid or ""}
    return refs

def load_preds(path: str) -> Dict[str, Dict[str, str]]:
    """
    Load predictions from JSON/JSONL/CSV.
    Accepts flexible schemas, tries to extract:
      - question_id (preferred)
      - question (fallback key)
      - final_answer OR answer OR content with 'FINAL ANSWER:' prefix
      - results like your example where `result` is [string, { ... final_answer: "..."}]
    Returns dict keyed by question_id if present, else by normalized question text.
    """
    preds = {}
    ext = os.path.splitext(path)[1].lower()
    rows = []

    def _append(obj: Dict[str, Any]):
        if not isinstance(obj, dict):
            return
        qid = (obj.get("question_id") or "").strip()
        q   = (obj.get("question") or "").strip()

        ans = first_not_none(
            obj.get("final_answer"),
            obj.get("answer"),
            obj.get("model_answer"),
        )

        if ans is None and isinstance(obj.get("result"), list):
            rlist = obj["result"]
            if len(rlist) > 0 and isinstance(rlist[0], str) and "FINAL ANSWER" in rlist[0]:
                ans = rlist[0]
            if ans is None and len(rlist) > 1 and isinstance(rlist[1], dict):
                ans = first_not_none(
                    rlist[1].get("final_answer"),
                    rlist[1].get("answer"),
                )

        if ans is None:
            content = obj.get("content")
            if isinstance(content, str) and "FINAL ANSWER" in content:
                ans = content

        if ans is None and isinstance(obj.get("data"), dict):
            ans = obj["data"].get("final_answer") or obj["data"].get("answer")

        if ans is None:
            return

        ans_text = extract_final_answer(ans)

        key = qid if qid else normalize_q(q)
        preds[key] = {"question": q, "answer": ans_text, "question_id": qid or ""}

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
                line=line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    _append(obj)
                except Exception:
                    continue
    elif ext == ".csv":
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                _append(row)
    else:
        raise ValueError(f"Unsupported preds file extension: {ext}")

    return preds

def first_not_none(*vals):
    for v in vals:
        if v is not None and v != "":
            return v
    return None

def normalize_q(q: str) -> str:
    return re.sub(r"\s+", " ", q.strip().lower())

def extract_final_answer(text: str) -> str:
    m = re.search(r"FINAL ANSWER\s*:\s*(.*)", text, re.I | re.S)
    return m.group(1).strip() if m else text.strip()

# heuristic judge (note this logic is fragile at this stage, llm judge output will override on failure) 
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

STRIP_TOKENS = r"[\s,\$%()]+"

def normalize_text(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[\$,]", "", s)
    s = re.sub(r"\s+", " ", s)
    return s

def extract_number_and_scale(s: str) -> Optional[float]:
    """
    Extract the first numeric figure and convert by unit word (billion, million, etc.).
    Returns value in absolute dollars (float), or None.
    """
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
    1) Exact normalized string match (loose formatting)
    2) Numeric match with unit scaling within rel_tol
    3) Percentage-style loose check (if both contain '%')
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

# llm judge (only support openai now)
JUDGE_PROMPT = """You are an impartial financial QA judge.

Decide if the model's FINAL ANSWER is factually equivalent to the ground truth FINAL ANSWER for the question below.

Rules:
- Judge only factual equivalence (ignore style and wording).
- Allow rounding/formatting differences up to ~1% for numeric values.
- If the prediction contains extra, non-conflicting context but is otherwise correct, mark it correct.
- Output strict JSON: {{"correct": true|false, "explanation": "<short reason>"}}

Question:
{question}

Ground truth:
{ground}

Model answer:
{pred}
"""

def call_openai_judge(model: str, question: str, ground: str, pred: str) -> Tuple[bool, str]:
    if not _OPENAI_OK:
        raise RuntimeError("OpenAI SDK not available. `pip install openai`")

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    prompt = JUDGE_PROMPT.format(question=question, ground=ground, pred=pred)

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
                "explanation": "No matching reference question_id or question text."
            })
            continue

        gt = ref["answer"]
        pred = prow["answer"]

        is_ok, reason = heuristic_match(gt, pred)
        method = "heuristic"
        explanation = reason
        correct = is_ok

        if not is_ok and not args.no_llm:
            try:
                j_ok, j_expl = call_openai_judge(args.judge_model, ref["question"], gt, pred)
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

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "question_id","question","pred_answer","gt_answer","correct","method","explanation"
        ])
        w.writeheader()
        w.writerows(scored_rows)
    acc = (correct_count / total) * 100 if total else 0.0
    log.info(f"Total={total} Correct={correct_count} Accuracy={acc:.2f}%")

if __name__ == "__main__":
    main()
