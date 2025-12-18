import json
import ast
from typing import List, Dict, Any
from .base import BaseJudge

class FactJudge(BaseJudge):
    def evaluate(self, question: dict, prediction: dict) -> dict:
        """
        Evaluates factual correctness using either the Rubric (if available) 
        or Semantic Comparison against the Ground Truth.
        """
        pred_text = prediction.get('final_answer', '')
        gt_text = question.get('answer', '')
        rubric_raw = question.get('rubric', '')
        
        # 1. Try to parse the Rubric
        criteria = self._parse_rubric(rubric_raw)
        
        # 2. Decide Evaluation Mode
        if criteria:
            return self._evaluate_with_rubric(pred_text, criteria)
        else:
            return self._evaluate_semantic(pred_text, gt_text)

    def render(self, result: dict) -> str:
        """
        Human-readable report generator.
        """
        score = result['score']
        mode = result['metadata'].get('mode', 'Unknown')
        details = result['metadata'].get('details', [])
        
        if score == 100: status = "PERFECT"
        elif score >= 80: status = "PASS"
        elif score >= 50: status = "WEAK"
        else: status = "FAIL"

        # Format the checklist items
        report_lines = []
        if mode == "Rubric":
            for item in details:
                icon = "✅" if item['met'] else "❌"
                report_lines.append(f"{icon} {item['criteria']}")
                if not item['met']:
                    report_lines.append(f"    Missed: {item.get('reason', 'Not found in answer')}")
        else:
            report_lines.append(f"Reasoning: {result['reason']}")

        return f"""
FACT COMPLIANCE REPORT
======================
STATUS:   {status} ({score:.1f}/100)
MODE:     {mode} Analysis

DETAILS:
{chr(10).join(report_lines)}
======================
""".strip()

    def _parse_rubric(self, rubric_str: str) -> List[str]:
        """
        Parses the JSON-like rubric string from CSV.
        Handles both valid JSON and Python-style dict strings (single quotes).
        """
        if not rubric_str or str(rubric_str).lower() == 'nan':
            return []

        try:
            data = json.loads(rubric_str)
        except:
            try:
                data = ast.literal_eval(rubric_str)
            except:
                return []

        criteria_list = []
        if isinstance(data, list):
            for item in data:
                if item.get('operator') == 'correctness':
                    criteria_list.append(item.get('criteria'))
        
        return criteria_list

    def _evaluate_with_rubric(self, pred_text: str, criteria: List[str]) -> dict:
        """
        Ask LLM to check off items from the criteria list.
        """
        if not self.client:
            return {"score": 0.0, "reason": "No LLM Client", "metadata": {}}

        criteria_block = "\n".join([f"{i+1}. {c}" for i, c in enumerate(criteria)])
        
        prompt = f"""
        You are a Financial Auditor. Compare the Candidate Answer against the Checklist.
        
        CANDIDATE ANSWER:
        "{pred_text}"
        
        CHECKLIST:
        {criteria_block}
        
        Instructions:
        - For each item, determine if the Candidate satisfies the criteria.
        - Allow for minor rounding differences (e.g. 10.5 vs 10.48 is OK).
        - Allow for semantic equivalents (e.g. "Revenue grew" vs "Sales increased").
        - If a specific number or fact is missing, mark it as False.
        
        Output STRICT JSON:
        {{
            "results": [
                {{"id": 1, "met": true, "reason": "Found match..."}},
                {{"id": 2, "met": false, "reason": "Missing value..."}}
            ]
        }}
        """

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            
            result_json = json.loads(response.choices[0].message.content)
            results = result_json.get("results", [])
            
            met_count = sum(1 for r in results if r['met'])
            total_count = len(criteria)
            score = (met_count / total_count) * 100.0 if total_count > 0 else 0.0

            details = []
            for i, res in enumerate(results):
                crit_text = criteria[i] if i < len(criteria) else "Unknown Criterion"
                details.append({
                    "criteria": crit_text,
                    "met": res['met'],
                    "reason": res.get('reason', '')
                })

            return {
                "score": round(score, 2),
                "reason": f"Matches {met_count}/{total_count} criteria.",
                "metadata": {
                    "mode": "Rubric",
                    "details": details,
                    "met_count": met_count,
                    "total_count": total_count
                }
            }
            
        except Exception as e:
            return {"score": 0.0, "reason": f"Judge Error: {e}", "metadata": {"mode": "Rubric Error"}}

    def _evaluate_semantic(self, pred_text: str, gt_text: str) -> dict:
        """
        Fallback: Standard Semantic Comparison when no rubric exists.
        """
        if not self.client:
             return {"score": 0.0, "reason": "No LLM Client", "metadata": {}}

        prompt = f"""
        You are a Financial Fact Checker.
        
        Ground Truth: "{gt_text}"
        Candidate: "{pred_text}"
        
        Compare the factual accuracy.
        - Score 100: Fully accurate, numbers match (within 5%), key facts present.
        - Score 50: Partially accurate, some facts missing or imprecise.
        - Score 0: Wrong numbers, wrong direction, or irrelevant.
        
        Output STRICT JSON:
        {{
            "score": 0.0 to 100.0,
            "reason": "Explanation..."
        }}
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            data = json.loads(response.choices[0].message.content)
            
            return {
                "score": float(data.get('score', 0)),
                "reason": data.get('reason', 'N/A'),
                "metadata": {
                    "mode": "Semantic",
                    "details": [] 
                }
            }
        except Exception as e:
            return {"score": 0.0, "reason": f"Judge Error: {e}", "metadata": {"mode": "Semantic Error"}}

