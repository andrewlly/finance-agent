import argparse
import json
import os
import csv
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI

from judges.source_judge import SourceJudge
from judges.fact_judge import FactJudge
from judges.efficiency_judge import EfficiencyJudge
from judges.refusal_judge import RefusalJudge
from judges.calibration_judge import CalibrationJudge


from judge_utils import load_refs, load_preds

load_dotenv()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refs", required=True, help="Path to ground truth CSV")
    ap.add_argument("--preds", required=True, help="Path to agent predictions JSON")
    ap.add_argument("--out_json", default="results/audit.json", help="Path to save structured JSON")
    ap.add_argument("--out_txt", default="results/audit_report.txt", help="Path to save human readable report")
    args = ap.parse_args()

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    judges = {
        "source": SourceJudge(llm_client=client),
        "fact": FactJudge(llm_client=client),
        "efficiency": EfficiencyJudge(),
        "refusal": RefusalJudge(llm_client=client),
        "calibration": CalibrationJudge()
    }

    refs = load_refs(args.refs)
    preds = load_preds(args.preds)

    full_audit_log = []
    human_readable_buffer = []

    ADVERSARIAL_TRIGGERS = ["NOT_FOUND", "N/A", "UNKNOWN", "NO_DATA", "NOT DETERMINABLE"]

    print(f"Auditing {len(preds)} predictions...")

    for pkey, prow in preds.items():
        ref = refs.get(pkey)
        if not ref: continue

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

        total_score = (
            0.5 * current_fact_score +
            0.3 * s_res['score'] +
            0.1 * e_res['score'] +
            0.1 * c_res['score']
        )
        
        q_result["composite_score"] = round(total_score, 2)
        full_audit_log.append(q_result)

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

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(full_audit_log, f, indent=2)
    
    with open(args.out_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(human_readable_buffer))

    print(f"\nAudit Complete.")
    print(f"Structured Data: {args.out_json}")
    print(f"Human Report:    {args.out_txt}")

if __name__ == "__main__":
    main()