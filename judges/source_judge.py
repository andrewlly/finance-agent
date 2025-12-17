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

    def evaluate(self, question: dict, prediction: dict) -> dict:
        """
        Core evaluation logic. Returns a dictionary of PURE DATA (no formatting).
        """
        pred_text = prediction.get('final_answer', '')
        logs = prediction.get('logs', {}) 
        
        # 1. Extract Citations
        cited_urls = self._extract_urls(pred_text)
        
        # Data container for JSON output
        result_data = {
            "citation_count": len(cited_urls),
            "hallucinations": [],
            "contradictions": [],
            "verified_sources": [],
            "best_source": "None",
            "failure_reasons": []
        }

        if not cited_urls:
            return self._build_output(0.0, ["HARD FAIL: No sources cited in final answer."], result_data)

        # 2. Determine Required Tier (Adaptive Scoring)
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
                result_data["hallucinations"].append(url)
                continue 

            # C. Validity Check (LLM Verification)
            validity_score = 1.0
            validity_reason = "Implicitly trusted (Judge LLM not active)"
            
            if self.client:
                validity_score, validity_reason = self._verify_validity_llm(pred_text, retrieved_text)
            
            # CHECK: Contradiction Rule
            if validity_score < 0.2:
                result_data["contradictions"].append({"url": url, "reason": validity_reason})
            else:
                result_data["verified_sources"].append({
                    "url": url, 
                    "tier_score": raw_tier_score, 
                    "validity": validity_score,
                    "reason": validity_reason
                })
            
            # D. Normalize Score
            # If we need Tier 2 (95) and get Tier 1 (100), ratio is capped at 1.0
            tier_ratio = min(1.0, raw_tier_score / required_score)
            
            # Final Source Score = (Tier Ratio * Validity) * 100
            final_source_score = (tier_ratio * validity_score) * 100
            source_scores.append(final_source_score)

        # 4. Calculate Final Score
        current_score = max(source_scores) if source_scores else 0.0
        
        if source_scores:
             best_idx = source_scores.index(max(source_scores))
             # Identify which source gave the best score
             # (Mapping back to citation list index, assuming order preserved)
             # To be safe, we check the verified_sources list order logic
             result_data["best_source"] = cited_urls[best_idx]
        elif result_data["hallucinations"]:
             # If only hallucinations exist, score is 0
             current_score = 0.0

        # 5. Apply Hard Penalties
        if result_data["contradictions"]:
            current_score = 0.0
            result_data["failure_reasons"].append("CRITICAL: Source contradiction detected.")

        if result_data["hallucinations"]:
            current_score = max(0.0, current_score - 30.0)
            result_data["failure_reasons"].append(f"PENALTY: -30 points for {len(result_data['hallucinations'])} hallucinated URL(s).")

        return self._build_output(current_score, result_data["failure_reasons"], result_data)

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

        # Build Details Block
        details_txt = []
        for item in meta.get('contradictions', []):
            details_txt.append(f"⛔ [CONTRADICTION] {item['url']}\n    Reason: {item['reason']}")
        
        for item in meta.get('verified_sources', []):
            tier = item['tier_score']
            validity = item['validity']
            details_txt.append(f"✅ [VERIFIED] {item['url']} (Tier {tier}, Validity {validity})\n    Context: {item['reason']}")
            
        for url in meta.get('hallucinations', []):
            details_txt.append(f"❌ [MISSING] {url} (Not found in logs)")

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

    # --- INTERNAL HELPERS ---

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
        
        # Pass 1: Look for the Data Key (Parse HTML)
        for turn in turns:
            for tool in turn.get('tool_calls', []):
                args = tool.get('arguments', {})
                if isinstance(args, str): 
                    try: args = json.loads(args)
                    except: continue
                
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
                        
                        # Did this analysis use our document key?
                        if data_key in args.get('prompt', ''):
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