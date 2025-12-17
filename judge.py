import argparse
import json
import os
import csv
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI

# Judges
from judges.source_judge import SourceJudge
from judges.fact_judge import FactJudge
from judges.efficiency_judge import EfficiencyJudge

# Helpers
from judge_utils import load_refs, load_preds 

load_dotenv()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refs", required=True)
    ap.add_argument("--preds", required=True)
    ap.add_argument("--out_json", default="results/audit.json", help="Path to save structured JSON")
    ap.add_argument("--out_txt", default="results/audit_report.txt", help="Path to save human readable report")
    args = ap.parse_args()

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    judges = {
        "source": SourceJudge(llm_client=client),
        "fact": FactJudge(llm_client=client),
        "efficiency": EfficiencyJudge()
    }

    refs = load_refs(args.refs)
    preds = load_preds(args.preds)

    full_audit_log = []
    human_readable_buffer = []

    print(f"Auditing {len(preds)} predictions...")

    for pkey, prow in preds.items():
        ref = refs.get(pkey)
        if not ref: continue

        q_result = {
            "question_id": ref.get("question_id"),
            "question": ref["question"],
            "prediction": prow["answer"],
            "ground_truth": ref["answer"],
            "judges": {},
            "composite_score": 0.0
        }
        
        total_weighted_score = 0.0
        
        weights = {"source": 0.3, "fact": 0.6, "efficiency": 0.1} 
        
        s_res = judges["source"].evaluate(ref, prow)
        q_result["judges"]["source"] = s_res
        total_weighted_score += s_res["score"] * weights["source"]

        f_res = judges["fact"].evaluate(ref, prow)
        q_result["judges"]["fact"] = f_res
        total_weighted_score += f_res["score"] * weights["fact"]

        e_res = judges["efficiency"].evaluate(ref, prow)
        q_result["judges"]["efficiency"] = e_res
        total_weighted_score += e_res["score"] * weights["efficiency"]

        q_result["composite_score"] = round(total_weighted_score, 2)
        full_audit_log.append(q_result)
        
        report_segment = f"""
##################################################
QUESTION: {ref['question']}
ID:       {ref.get('question_id')}
SCORE:    {q_result['composite_score']}/100
##################################################
ANSWER:
{prow['answer']}

{judges['fact'].render(f_res)}
{judges['source'].render(s_res)}
{judges['efficiency'].render(e_res)}

--------------------------------------------------
"""
        human_readable_buffer.append(report_segment)
        print(f"Processed {ref.get('question_id')}: Score {q_result['composite_score']}")

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