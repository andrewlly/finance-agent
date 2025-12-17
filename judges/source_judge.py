import re
import json
from urllib.parse import urlparse
from .base import BaseJudge

class SourceJudge(BaseJudge):
    # Tier Definitions: Higher score = More trusted
    TIER_SCORES = {
        # Tier 1: Primary / Gold Standard (100 pts)
        "sec.gov": 100,
        "investor.": 100,
        "ir.": 100,
        
        # Tier 2: Reputable Financial News (95 pts)
        "bloomberg.com": 95,
        "reuters.com": 95,
        "finance.yahoo.com": 95,
        "wsj.com": 95,
        "cnbc.com": 95,
        "marketwatch.com": 95,
        
        # Tier 3: Aggregators / General (60 pts) - Default for others
    }

    def evaluate(self, question: dict, prediction: dict, is_adversarial: bool = False) -> dict:
        """
        Evaluate sources. 
        If is_adversarial=True, checks for RELEVANCE instead of SUPPORT.
        """
        pred_text = prediction.get('final_answer', '')
        logs = prediction.get('logs', {}) 
        cited_urls = self._extract_urls(pred_text)
        
        result_data = {
            "citation_count": len(cited_urls),
            "hallucinations": [],
            "contradictions": [],
            "verified_sources": [],
            "failure_reasons": []
        }

        # RULE: Even in adversarial cases, you must cite where you looked.
        if not cited_urls:
            return self._build_output(0.0, ["FAIL: No evidence of search provided (Lazy Refusal)."], result_data)

        source_scores = []

        for url in cited_urls:
            raw_tier_score = self._get_tier_score(url)
            retrieved_text = self._find_text_in_logs(url, logs)
            
            if not retrieved_text:
                result_data["hallucinations"].append(url)
                continue 

            # --- BRANCHING LOGIC ---
            if is_adversarial:
                # ADVERSARIAL MODE: Check Relevance ("Is this a good place to have looked?")
                validity_score, reason = self._verify_relevance_llm(question['question'], retrieved_text)
            else:
                # STANDARD MODE: Check Support ("Does this support the claim?")
                validity_score, reason = self._verify_validity_llm(pred_text, retrieved_text)
            
            # ... (Rest of scoring logic is the same) ...
            
            if validity_score < 0.2:
                result_data["contradictions"].append({"url": url, "reason": reason})
            else:
                result_data["verified_sources"].append({
                    "url": url, 
                    "tier_score": raw_tier_score, 
                    "validity": validity_score,
                    "reason": reason
                })
            
            # Normalize and append score...
            min_tier_req = int(question.get('min_tier', 3)) 
            required_score = 95 if min_tier_req == 2 else (100 if min_tier_req == 1 else 60)
            tier_ratio = min(1.0, raw_tier_score / required_score)
            source_scores.append((tier_ratio * validity_score) * 100)

        # ... (Calculate final score max/avg) ...
        current_score = max(source_scores) if source_scores else 0.0
        
        # ... (Penalties) ...
        if result_data["hallucinations"]:
            current_score = max(0.0, current_score - 30.0)

        return self._build_output(current_score, result_data["failure_reasons"], result_data)

    def _verify_relevance_llm(self, user_question, source_text):
        """
        Adversarial Check: Did the agent look in a RELEVANT document?
        (It doesn't matter if the answer isn't there, as long as the document is related).
        """
        prompt = f"""
        You are a Research Auditor.
        The user asked: "{user_question}"
        The agent checked this document: "{source_text[:3000]}"
        
        Is this document RELEVANT to the question topic?
        (e.g. If asking about Tesla Revenue, a Tesla 10-K is RELEVANT, even if it doesn't have the specific year).
        
        Output JSON: {{ "score": 0.0 to 1.0, "reason": "Explanation" }}
        """
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            data = json.loads(response.choices[0].message.content)
            return float(data.get("score", 0.0)), data.get("reason", "N/A")
        except:
            return 0.0, "Judge Error"

    def render(self, result: dict) -> str:
        """
        Takes the evaluation result and returns a Human-Readable String Report.
        """
        score = result['score']
        meta = result['metadata']
        
        if score == 100: status = "PERFECT"
        elif score >= 80: status = "PASS"
        elif score >= 50: status = "WEAK"
        else: status = "FAIL"

        details_txt = []
        for item in meta.get('contradictions', []):
            details_txt.append(f"[CONTRADICTION] {item['url']}\n    Reason: {item['reason']}")
        
        for item in meta.get('verified_sources', []):
            tier = item['tier_score']
            validity = item['validity']
            details_txt.append(f"[VERIFIED] {item['url']} (Tier {tier}, Validity {validity})\n    Context: {item['reason']}")
            
        for url in meta.get('hallucinations', []):
            details_txt.append(f"[MISSING] {url} (Not found in logs)")

        return f"""
SOURCE COMPLIANCE REPORT
========================
STATUS:      {status} ({score:.1f}/100)
CITATIONS:   {meta.get('citation_count', 0)}
BEST SOURCE: {meta.get('best_source', 'None')}

RISK LOG:
• Hallucinations: {len(meta.get('hallucinations', []))}
• Contradictions: {len(meta.get('contradictions', []))}

DETAILS:
{chr(10).join(details_txt) if details_txt else "No details."}

JUDGE SUMMARY:
{result['reason']}
========================
""".strip()


    def _build_output(self, score, reasons, data):
        return {
            "score": round(score, 2),
            "reason": " ".join(reasons) if reasons else "Compliant",
            "metadata": data
        }

    def _extract_urls(self, text):
        return re.findall(r'https?://[^\s"\'\)\]\}]+', text)

    def _get_tier_score(self, url):
        try:
            domain = urlparse(url).netloc.lower()
            for key, score in self.TIER_SCORES.items():
                if key in domain:
                    return score
            return 60.0 
        except:
            return 0.0

    def _find_text_in_logs(self, url, logs):
        """
        Locates the BEST content for a URL. 
        Prioritizes: 
        1. Retrieval/Analysis (High context) 
        2. Parse HTML Output (High context)
        3. Search Snippet (Low context - Fallback)
        """
        turns = logs.get('turns', [])
        
        best_content = None
        data_key = None
        
        for turn in turns:
            for tool in turn.get('tool_calls', []):
                args = tool.get('arguments', {})
                if isinstance(args, str): 
                    try: args = json.loads(args)
                    except: continue
                
                if tool['tool_name'] == 'parse_html_page':
                    if url in args.get('url', '') or args.get('url', '') in url:
                        data_key = args.get('key')
                        if tool.get('tool_output'):
                            best_content = tool.get('tool_output')

        if data_key:
            for turn in turns:
                for tool in turn.get('tool_calls', []):
                    if tool['tool_name'] == 'retrieve_information':
                        args = tool.get('arguments', {})
                        if isinstance(args, str): 
                            try: args = json.loads(args)
                            except: continue
                        
                        if data_key in args.get('prompt', ''):
                            return tool.get('tool_output') or tool.get('result', '')

        if best_content:
            return best_content


        for turn in turns:
            for tool in turn.get('tool_calls', []):
                if tool['tool_name'] == 'google_web_search':
                    output = tool.get('tool_output') or tool.get('result', '')
                    if url in str(output):
                        return str(output)[:5000]

        return None

    def _verify_validity_llm(self, claim, source_text):
        """
        Ask LLM for a JSON response containing Score AND Reasoning.
        """
        prompt = f"""
        You are a strict Financial Auditor. 
        Verify if the Source Text supports the Claim.

        Claim: "{claim[:1000]}"
        Source Text: "{source_text[:3000]}"

        Output STRICT JSON only:
        {{
            "reason": "Brief explanation of why it supports or contradicts...",
            "score": 0.0 to 1.0
        }}
        
        Scoring Guide:
        1.0 = Fully Supported (Numbers and context match)
        0.5 = Partially Supported (Context matches, numbers slightly off or ambiguous)
        0.0 = Contradicted / Not Found / Irrelevant
        """
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            content = response.choices[0].message.content.strip()
            result = json.loads(content)
            
            score = float(result.get("score", 0.0))
            reason = result.get("reason", "No reason provided by judge.")
            
            return max(0.0, min(1.0, score)), reason
        except Exception as e:
            return 0.0, f"Judge Error: {str(e)}"