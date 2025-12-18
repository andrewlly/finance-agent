import re
import json
from urllib.parse import urlparse
from .base import BaseJudge

class SourceJudge(BaseJudge):
    TIER_SCORES = {
        1: 100,  # Primary (SEC, Gov, Company IR)
        2: 95,   # Major financial news (Bloomberg, Reuters, CNBC)
        3: 50   # General news, blogs, social media, forums, unknown
    }

    TIER_1_DOMAINS = [
        "sec.gov", "investor.", "investors.", "ir.", "about.", "finance.yahoo.com/quote", 
        "europa.eu", "gov.uk", "statista.com", "macrotrends.net"
    ]
    TIER_2_DOMAINS = [
        "bloomberg.com", "reuters.com", "cnbc.com", "wsj.com", "ft.com", 
        "forbes.com", "marketwatch.com", "seekingalpha.com", "morningstar.com"
    ]

    def _get_tier_score(self, url: str) -> int:
        try:
            domain = urlparse(url).netloc.lower()
            if any(d in domain for d in self.TIER_1_DOMAINS): return 100
            if any(d in domain for d in self.TIER_2_DOMAINS): return 75
            return 50 
        except:
            return 0

    def _extract_urls(self, text: str) -> list[str]:
        urls = set()
        try:
            json_match = re.search(r'("sources"\s*:\s*\[.*?\])', text, re.DOTALL)
            if json_match:
                json_str = "{" + json_match.group(1) + "}"
                try:
                    data = json.loads(json_str)
                    for src in data.get("sources", []):
                        if "url" in src:
                            urls.add(src["url"])
                except:
                    block = json_match.group(1)
                    raw_links = re.findall(r'(https?://[^\s"\'<>]+)', block)
                    for link in raw_links:
                        urls.add(link.strip('",\')]}'))
        except Exception:
            pass
        
        if not urls:
            raw_links = re.findall(r'(https?://[^\s"\'<>]+)', text)
            for link in raw_links:
                clean_link = link.strip('",\')]}')
                urls.add(clean_link)

        return list(urls)

    def _find_text_in_logs(self, url: str, logs: dict) -> str:
        """
        Scans the agent logs to find the text content associated with a URL.
        """
        if not logs or "turns" not in logs:
            return ""

        for turn in logs["turns"]:
            for tool_call in turn.get("tool_calls", []):
                args = tool_call.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except:
                        pass
                
                if isinstance(args, dict) and args.get("url") == url:
                    return tool_call.get("tool_output", "")
                
                if url in str(tool_call.get("tool_output", "")):
                    return tool_call.get("tool_output", "")

        return ""

    def _verify_validity_llm(self, claim, source_text):
        """
        Standard Pipeline: Does the source support the claim?
        """
        if not source_text or len(source_text) < 50:
            return 0.0, "Source content missing or empty in logs."

        prompt = f"""
        Verify if the Source Text supports the Agent's Claim.
        
        Agent Claim: "{claim}"
        
        Source Text (Excerpt):
        "{source_text[:3000]}..."
        
        Task:
        1. Does the source explicitly contain the numbers/facts in the claim?
        2. Is the context correct (same company, same year, same metric)?
        
        Output JSON: {{ "score": float (0.0 to 1.0), "reason": "str" }}
        """
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            data = json.loads(response.choices[0].message.content)
            return float(data['score']), data['reason']
        except:
            return 0.0, "Validation Error"

    def _verify_relevance_llm(self, user_question, source_text):
        """
        Adversarial Pipeline: Is the source RELEVANT? 
        (Used when the answer is NOT_FOUND to ensure agent looked in the right place).
        """
        if not source_text or len(source_text) < 50:
            return 0.0, "Source content missing in logs."

        prompt = f"""
        You are a Research Auditor.
        The user asked: "{user_question}"
        The agent checked this document: "{source_text[:3000]}..."
        
        Task:
        Is this document RELEVANT to the question topic?
        (e.g., If asking about Tesla Revenue, a Tesla 10-K is RELEVANT, even if it doesn't have the specific year).
        
        Output JSON: {{ "score": float (0.0 to 1.0), "reason": "str" }}
        """
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            data = json.loads(response.choices[0].message.content)
            return float(data['score']), data['reason']
        except:
            return 0.0, "Validation Error"

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

        if not cited_urls:
            # If adversarial, refusing without sources is "Lazy Refusal"
            if is_adversarial and ("not found" in pred_text.lower() or "no info" in pred_text.lower()):
                msg = "FAIL: Lazy Refusal. Agent claimed data not found but did not cite where it looked."
            else:
                msg = "FAIL: Unsourced Claim. Answer provided but no sources cited."
            
            return self._build_output(0.0, [msg], result_data)

        source_scores = []

        for url in cited_urls:
            raw_tier_score = self._get_tier_score(url)
            retrieved_text = self._find_text_in_logs(url, logs)
            
            if not retrieved_text:
                result_data["hallucinations"].append(url)
                continue 

            if is_adversarial:
                validity_score, reason = self._verify_relevance_llm(question['question'], retrieved_text)
            else:
                validity_score, reason = self._verify_validity_llm(pred_text, retrieved_text)
            
            if validity_score < 0.2:
                result_data["contradictions"].append({"url": url, "reason": reason})
                final_source_score = 0.0
            else:
                min_tier_req = int(question.get('min_tier', 3)) 
                required_score = 100 if min_tier_req == 1 else (75 if min_tier_req == 2 else 50)
                
                tier_ratio = min(1.0, raw_tier_score / required_score)
                final_source_score = (tier_ratio * validity_score) * 100.0
                
                result_data["verified_sources"].append({
                    "url": url, 
                    "tier_score": raw_tier_score, 
                    "validity": validity_score,
                    "reason": reason
                })
            
            source_scores.append(final_source_score)

        if not source_scores:
            current_score = 0.0
        else:
            current_score = max(source_scores)
        
        if result_data["hallucinations"]:
            current_score = max(0.0, current_score - 25.0)

        return self._build_output(current_score, result_data["failure_reasons"], result_data)

    def render(self, result: dict) -> str:
        meta = result.get('metadata', {})
        
        output = [
            "SOURCE COMPLIANCE REPORT",
            "========================",
            f"STATUS:      {'PASS' if result['score'] >= 70 else 'FAIL'} ({result['score']}/100)",
            f"CITATIONS:   {meta.get('citation_count', 0)}",
        ]
        
        if meta.get('hallucinations'):
             output.append(f"FAKE URLS:   {len(meta['hallucinations'])} (Penalty Applied)")

        if meta.get('verified_sources'):
            best = max(meta['verified_sources'], key=lambda x: x['validity'])
            output.append(f"BEST SOURCE: {best['url']} (Validity: {best['validity']:.0%})")
        elif not meta.get('verified_sources') and meta.get('citation_count', 0) > 0:
            output.append("BEST SOURCE: None verified (All sources failed validation)")
        else:
            output.append("BEST SOURCE: None provided")

        if result.get('reason'):
             output.append(f"JUDGE NOTE:  {result['reason']}")

        return "\n".join(output)

