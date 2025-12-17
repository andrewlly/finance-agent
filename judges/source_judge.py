# judges/source_judge.py
import re
import json
from urllib.parse import urlparse
from .base import BaseJudge


class SourceJudge(BaseJudge):
    # Tier Definitions
    TIER_SCORES = {
        "sec.gov": 100,
        "investor.": 100,
        "ir.": 100,
        "bloomberg.com": 95,
        "reuters.com": 95,
        "finance.yahoo.com": 95,
        "wsj.com": 95,
        "cnbc.com": 95,
        "marketwatch.com": 95,
    }

    def evaluate(self, question: dict, prediction: dict) -> dict:
        pred_text = prediction.get('final_answer', '')
        logs = prediction.get('logs', {}) 
        
        # 1. Extract Citations
        cited_urls = self._extract_urls(pred_text)
        
        # Initialize Report Data
        audit_data = {
            "citation_count": len(cited_urls),
            "hallucinations": [],
            "contradictions": [],
            "best_source": "N/A",
            "details": []
        }

        if not cited_urls:
            return self._finalize_result(0.0, "HARD FAIL: No sources cited in final answer.", audit_data)

        # 2. Determine Required Tier
        min_tier_req = int(question.get('min_tier', 3)) 
        required_score = 95 if min_tier_req == 2 else (100 if min_tier_req == 1 else 60)

        source_scores = []
        
        # 3. Evaluate Each Source
        for url in cited_urls:
            # A. Tier Score
            raw_tier_score = self._get_tier_score(url)
            
            # B. Grounding Check (Trace URL -> Key -> Content)
            retrieved_text = self._find_text_in_logs(url, logs)
            
            if not retrieved_text:
                audit_data["hallucinations"].append(url)
                audit_data["details"].append(f"[MISSING] {url}: URL cited but not found in execution logs.")
                continue 

            # C. Validity Check (LLM Verification with Reasoning)
            validity_score = 1.0
            validity_reason = "Implicitly trusted (Judge LLM not active)"
            
            if self.client:
                # Returns tuple: (score, reason)
                validity_score, validity_reason = self._verify_validity_llm(pred_text, retrieved_text)
            
            # CHECK: Contradiction Rule
            if validity_score < 0.2:
                audit_data["contradictions"].append(url)
                audit_data["details"].append(f"[CONTRADICTION] {url}: {validity_reason}")
            else:
                audit_data["details"].append(f"[VERIFIED] {url} (Tier: {raw_tier_score}): {validity_reason}")
            
            # D. Normalize Score
            tier_ratio = min(1.0, raw_tier_score / required_score)
            final_source_score = (tier_ratio * validity_score) * 100
            source_scores.append(final_source_score)

        # 4. Scoring Logic
        current_score = max(source_scores) if source_scores else 0.0
        
        if source_scores:
             best_idx = source_scores.index(max(source_scores))
             audit_data["best_source"] = cited_urls[best_idx]
        elif audit_data["hallucinations"]:
             current_score = 0.0

        # Apply Penalties
        failure_reasons = []
        if audit_data["contradictions"]:
            current_score = 0.0
            failure_reasons.append("CRITICAL: Source contradiction detected.")

        if audit_data["hallucinations"]:
            current_score = max(0.0, current_score - 30.0)
            failure_reasons.append(f"PENALTY: -30 points for {len(audit_data['hallucinations'])} hallucinated URL(s).")

        final_reason = " ".join(failure_reasons) if failure_reasons else "All sources verified compliant."
        
        return self._finalize_result(current_score, final_reason, audit_data)

    def _finalize_result(self, score, reason, data):
        """Helper to package the score + the text report (Natural/Professional Style)"""
        
        # Status Logic
        if score == 100: status = "PERFECT"
        elif score >= 80: status = "PASS"
        elif score >= 50: status = "WEAK"
        else: status = "FAIL"

        # Professional Text Block (No Emojis)
        report_text = f"""
SOURCE COMPLIANCE REPORT
==================================================
STATUS:        {status} ({score:.1f}/100)
CITATIONS:     {data['citation_count']} present
BEST SOURCE:   {data['best_source']}

RISK ASSESSMENT
---------------
Hallucinations: {len(data['hallucinations'])} detected
Contradictions: {len(data['contradictions'])} detected

AUDIT LOGS
----------
{chr(10).join(data['details']) if data['details'] else "No audit details available."}

JUDGE SUMMARY
-------------
{reason}
==================================================
"""
        return {
            "score": round(score, 2),
            "reason": reason,
            "report_text": report_text.strip(),
            "metadata": data
        }

    # --- Internal Helpers ---

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
        
        # Pass 1: Look for the Data Key (Parse HTML)
        for turn in turns:
            for tool in turn.get('tool_calls', []):
                args = tool.get('arguments', {})
                if isinstance(args, str): 
                    try: args = json.loads(args)
                    except: continue
                
                # Check parse_html_page
                if tool['tool_name'] == 'parse_html_page':
                    # Fuzzy match URL
                    if url in args.get('url', '') or args.get('url', '') in url:
                        data_key = args.get('key')
                        # If we logged the full HTML content here, grab it!
                        if tool.get('tool_output'):
                            best_content = tool.get('tool_output')

        # Pass 2: Look for Retrieval Analysis using that Key (Highest Priority)
        if data_key:
            for turn in turns:
                for tool in turn.get('tool_calls', []):
                    if tool['tool_name'] == 'retrieve_information':
                        args = tool.get('arguments', {})
                        if isinstance(args, str): 
                            try: args = json.loads(args)
                            except: continue
                        
                        # Did this analysis use our document?
                        if data_key in args.get('prompt', ''):
                            # This is the "Gold" content - the agent's actual analysis
                            return tool.get('tool_output') or tool.get('result', '')

        # Pass 3: If we have the direct HTML content from Pass 1, return it
        if best_content:
            return best_content

        # Pass 4: Fallback to Google Search Snippet (Lowest Priority)
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
                response_format={"type": "json_object"} # Force JSON mode
            )
            content = response.choices[0].message.content.strip()
            result = json.loads(content)
            
            score = float(result.get("score", 0.0))
            reason = result.get("reason", "No reason provided by judge.")
            
            return max(0.0, min(1.0, score)), reason
        except Exception as e:
            return 0.0, f"Judge Error: {str(e)}"