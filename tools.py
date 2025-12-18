import json
import os
import re
import traceback
import io
import heapq
from abc import ABC, abstractmethod

import aiohttp
import backoff
from bs4 import BeautifulSoup
from pypdf import PdfReader

from model_library.base import LLM, ToolBody, ToolDefinition

from logger import get_logger

tool_logger = get_logger(__name__)

MAX_END_DATE = "2025-04-07"

IMAGE_TRIGGER_PROMPT = """
Assess if the users would be able to understand response better with the use of diagrams and trigger them. 
You can insert a diagram by adding the 

[Image of X]
 tag where X is a contextually relevant and domain-specific query to fetch the diagram. 
Examples: 

[Image of the human digestive system]
, 

[Image of hydrogen fuel cell]
.
- Avoid triggering images just for visual appeal.
- Be economical but strategic.
- Place the image tag immediately before or after the relevant text without disrupting the flow.
"""

def is_429(exception):
    is429 = (
        isinstance(exception, aiohttp.ClientResponseError)
        and exception.status == 429
        or "429" in str(exception)
    )
    if is429:
        tool_logger.error(f"429 error: {exception}")
    return is429


def retry_on_429(func):
    @backoff.on_exception(
        backoff.expo,
        aiohttp.ClientResponseError,
        max_tries=8,
        base=2,
        factor=3,
        jitter=backoff.full_jitter,
        giveup=lambda e: not is_429(e),
    )
    async def wrapper(*args, **kwargs):
        return await func(*args, **kwargs)

    return wrapper


class Tool(ABC):
    """
    Abstract base class for tools.
    """

    name: str
    description: str
    input_arguments: dict
    required_arguments: list[str]

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__()

    def get_tool_definition(self) -> ToolDefinition:
        body = ToolBody(
            name=self.name,
            description=self.description,
            properties=self.input_arguments,
            required=self.required_arguments,
        )

        definition = ToolDefinition(name=self.name, body=body)

        return definition

    @abstractmethod
    def call_tool(self, arguments: dict, *args, **kwargs) -> list[str]:
        pass

    async def __call__(self, arguments: dict = None, *args, **kwargs) -> list[str]:
        tool_logger.info(
            f"\033[1;33m[TOOL: {self.name.upper()}]\033[0m Calling with arguments: {arguments}"
        )

        try:
            tool_result = await self.call_tool(arguments, *args, **kwargs)
            tool_logger.info(
                f"\033[1;32m[TOOL: {self.name.upper()}]\033[0m Returned: {tool_result}"
            )
            if self.name == "retrieve_information":
                return {
                    "success": True,
                    "result": tool_result["retrieval"],
                    "usage": tool_result["usage"],
                }
            else:
                return {"success": True, "result": json.dumps(tool_result)}
        except Exception as e:
            is_verbose = os.environ.get("EDGAR_AGENT_VERBOSE", "0") == "1"
            error_msg = str(e)
            if is_verbose:
                error_msg += f"\nTraceback: {traceback.format_exc()}"
                tool_logger.warning(
                    f"\033[1;31m[TOOL: {self.name.upper()}]\033[0m Error: {e}\nTraceback: {traceback.format_exc()}"
                )
            else:
                tool_logger.warning(
                    f"\033[1;31m[TOOL: {self.name.upper()}]\033[0m Error: {e}"
                )
            return {"success": False, "result": error_msg}


class GoogleWebSearch(Tool):
    name: str = "google_web_search"
    description: str = "Search the web for information"
    input_arguments: dict = {
        "search_query": {
            "type": "string",
            "description": "The query to search for",
        }
    }
    required_arguments: list[str] = ["search_query"]

    def __init__(
        self,
        top_n_results: int = 10,
        serpapi_api_key: str | None = None,
        *args,
        **kwargs,
    ):
        super().__init__(
            self.name,
            self.description,
            self.input_arguments,
            self.required_arguments,
            *args,
            **kwargs,
        )
        self.top_n_results = top_n_results
        if serpapi_api_key is None:
            serpapi_api_key = os.getenv("SERPAPI_API_KEY")
        self.serpapi_api_key = serpapi_api_key

    @retry_on_429
    async def _execute_search(self, search_query: str) -> list[str]:
        if not self.serpapi_api_key:
            raise ValueError("SERPAPI_API_KEY is not set")

        max_date_parts = MAX_END_DATE.split("-")
        google_date_format = (
            f"{max_date_parts[1]}/{max_date_parts[2]}/{max_date_parts[0]}"
        )

        params = {
            "api_key": self.serpapi_api_key,
            "engine": "google",
            "q": search_query,
            "num": self.top_n_results,
            "tbs": f"cdr:1,cd_max:{google_date_format}",
        }

        async with aiohttp.ClientSession() as session:
            async with session.get(
                "https://serpapi.com/search.json", params=params
            ) as response:
                response.raise_for_status()
                results = await response.json()

        return results.get("organic_results", [])

    async def call_tool(self, arguments: dict) -> list[str]:
        results = await self._execute_search(**arguments)
        return results


class EDGARSearch(Tool):
    name: str = "edgar_search"
    description: str = (
        """
    Search the EDGAR Database through the SEC API.
    You should provide a query, a list of form types, a list of CIKs, a start date, an end date, a page number, and a top N results.
    The results are returned as a list of dictionaries, each containing the metadata for a filing. It does not contain the full text of the filing.
    """.strip()
    )
    input_arguments: dict = {
        "query": {
            "type": "string",
            "description": """The keyword or phrase to search, such as 'substantial doubt' OR 'material weakness'""",
        },
        "form_types": {
            "type": "array",
            "description": """Limits search to specific SEC form types (e.g., ['8-K', '10-Q']) list of strings. Default is None (all form types)""",
            "items": {"type": "string"},
        },
        "ciks": {
            "type": "array",
            "description": "Filters results to filings by specified CIKs, type list of strings. Default is None (all filers).",
            "items": {"type": "string"},
        },
        "start_date": {
            "type": "string",
            "description": "Start date for the search range in yyyy-mm-dd format. Used with endDate to define the date range. Example: '2024-01-01'. Default is 30 days ago",
        },
        "end_date": {
            "type": "string",
            "description": "End date for the search range, in the same format as startDate. Default is today",
        },
        "page": {
            "type": "string",
            "description": "Pagination for results. Default is '1'",
        },
        "top_n_results": {
            "type": "integer",
            "description": "The top N results to return after the query. Useful if you are not sure the result you are loooking for is ranked first after your query.",
        },
    }
    required_arguments: list[str] = [
        "query",
        "form_types",
        "ciks",
        "start_date",
        "end_date",
        "page",
        "top_n_results",
    ]

    def __init__(
        self,
        sec_api_key: str | None = None,
        *args,
        **kwargs,
    ):
        super().__init__(
            *args,
            **kwargs,
        )
        if sec_api_key is None:
            sec_api_key = os.getenv("SEC_EDGAR_API_KEY")
        self.sec_api_key = sec_api_key
        self.sec_api_url = "https://api.sec-api.io/full-text-search"

    @retry_on_429
    async def _execute_search(
        self,
        query: str,
        form_types: list[str],
        ciks: list[str],
        start_date: str,
        end_date: str,
        page: int,
        top_n_results: int,
    ) -> list[str]:
        if not self.sec_api_key:
            raise ValueError("SEC_EDGAR_API_KEY is not set")

        if (
            isinstance(form_types, str)
            and form_types.startswith("[")
            and form_types.endswith("]")
        ):
            try:
                form_types = json.loads(form_types.replace("'", '"'))
            except json.JSONDecodeError:
                form_types = [
                    item.strip(" \"'") for item in form_types[1:-1].split(",")
                ]

        if isinstance(ciks, str) and ciks.startswith("[") and ciks.endswith("]"):
            try:
                ciks = json.loads(ciks.replace("'", '"'))
            except json.JSONDecodeError:
                ciks = [item.strip(" \"'") for item in ciks[1:-1].split(",")]

        if end_date > MAX_END_DATE:
            end_date = MAX_END_DATE

        payload = {
            "query": query,
            "formTypes": form_types,
            "ciks": ciks,
            "startDate": start_date,
            "endDate": end_date,
            "page": page,
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": self.sec_api_key,
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.sec_api_url, json=payload, headers=headers
            ) as response:
                response.raise_for_status()
                result = await response.json()

        return result.get("filings", [])[: int(top_n_results)]

    async def call_tool(self, arguments: dict) -> list[str]:
        try:
            return await self._execute_search(**arguments)
        except Exception as e:
            is_verbose = os.environ.get("EDGAR_AGENT_VERBOSE", "0") == "1"
            if is_verbose:
                tool_logger.error(
                    f"SEC API error: {e}\nTraceback: {traceback.format_exc()}"
                )
            else:
                tool_logger.error(f"SEC API error: {e}")
            raise


class ParseHtmlPage(Tool):
    name: str = "parse_html_page"
    description: str = (
        """
        Parse an HTML page. This tool is used to parse the HTML content of a page and saves the content outside of the conversation to avoid context window issues.
        You should provide both the URL of the page to parse, as well as the key you want to use to save the result in the agent's data structure.
        The data structure is a dictionary.
    """.strip()
    )

    input_arguments: dict = {
        "url": {"type": "string", "description": "The URL of the HTML page to parse"},
        "key": {
            "type": "string",
            "description": """The key to use when saving the result in the conversation's data structure (dict).""",
        },
    }
    required_arguments: list[str] = ["url", "key"]

    def __init__(
        self, headers: dict = {"User-Agent": "ValsAI/finance-agent"}, *args, **kwargs
    ):
        super().__init__(
            *args,
            **kwargs,
        )
        self.headers = headers

    @retry_on_429
    async def _parse_html_page(self, url: str) -> str:
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(
                    url, headers=self.headers, timeout=60
                ) as response:
                    try:
                        html_content = await response.text()
                    except UnicodeDecodeError:
                        raw_content = await response.read()
                        html_content = raw_content.decode('utf-8', errors='ignore')
                    
            except Exception as e:
                if len(str(e)) == 0:
                    raise TimeoutError(
                        "Timeout error when parsing HTML page after 60 seconds."
                    )
                else:
                    raise Exception(str(e))

        soup = BeautifulSoup(html_content, "html.parser")

        for script_or_style in soup(["script", "style"]):
            script_or_style.extract()

        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = "\n".join(chunk for chunk in chunks if chunk)

        return text

    async def _save_tool_output(
        self, output: list[str], key: str, data_storage: dict
    ) -> None:
        if not output:
            return

        tool_result = ""
        if key in data_storage:
            tool_result = "WARNING: Key exists. Overwriting.\n"
        tool_result += (
            f"SUCCESS: Saved to data storage under key: {key}.\n"
        )

        data_storage[key] = output

        keys_list = "\n".join(data_storage.keys())
        tool_result += (
            f"""
        Data storage keys:
        {keys_list}
        """.strip()
            + "\n"
        )

        return tool_result

    async def call_tool(self, arguments: dict, data_storage: dict) -> list[str]:
        url = arguments.get("url")
        key = arguments.get("key")
        text_output = await self._parse_html_page(url)
        tool_result = await self._save_tool_output(text_output, key, data_storage)

        return tool_result

class ParsePDF(Tool):
    name: str = "parse_pdf"
    description: str = (
        """
        Parse a PDF file from a URL. This tool downloads the PDF and extracts its text content.
        Use this tool when the URL ends in .pdf or when 'parse_html_page' fails due to binary encoding errors.
        You should provide both the URL of the PDF and the key to save the result.
        """
    )

    input_arguments: dict = {
        "url": {"type": "string", "description": "The URL of the PDF file to parse"},
        "key": {
            "type": "string",
            "description": """The key to use when saving the result in the conversation's data storage.""",
        },
    }
    required_arguments: list[str] = ["url", "key"]

    def __init__(
        self, headers: dict = {"User-Agent": "ValsAI/finance-agent"}, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.headers = headers

    @retry_on_429
    async def _parse_pdf(self, url: str) -> str:
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(
                    url, headers=self.headers, timeout=60
                ) as response:
                    response.raise_for_status()
                    pdf_bytes = await response.read()
            except Exception as e:
                raise Exception(f"Failed to download PDF: {e}")

        try:
            reader = PdfReader(io.BytesIO(pdf_bytes))
            text_content = []
            
            for i, page in enumerate(reader.pages):
                page_text = page.extract_text()
                if page_text:
                    text_content.append(f"--- Page {i+1} ---\n{page_text}")
            
            full_text = "\n".join(text_content)
            
            return full_text if full_text.strip() else "PDF downloaded but contains no extractable text."
            
        except Exception as e:
            raise Exception(f"Failed to parse PDF content: {e}")

    async def _save_tool_output(self, output: str, key: str, data_storage: dict) -> str:
        if not output: return "No output to save."
        
        msg = f"SUCCESS: PDF content saved under key: {key}."
        if key in data_storage:
            msg = "WARNING: Key exists. Overwriting.\n" + msg
            
        data_storage[key] = output
        return msg

    async def call_tool(self, arguments: dict, data_storage: dict) -> list[str]:
        url = arguments.get("url")
        key = arguments.get("key")
        
        if not url.lower().endswith(".pdf") and "pdf" not in url.lower():
            tool_logger.warning(f"URL {url} does not look like a PDF, but attempting parse anyway.")

        text_output = await self._parse_pdf(url)
        result_msg = await self._save_tool_output(text_output, key, data_storage)
        return result_msg


class RetrieveInformation(Tool):
    name: str = "retrieve_information"
    description: str = (
        """
    Retrieve information from the conversation's data structure (dict) and analyze it.
    
    IMPORTANT: 
    1. Your prompt MUST include at least one key from data storage: {{key_name}}.
    2. You can specify character ranges to read only specific parts.
    3. You can request DIAGRAMS. If you see complex concepts (systems, flows, structures) in the text, 
       insert 

[Image of X]
 tags in your output where helpful.
    
    Output is the result from the LLM analysis.
    """.strip()
    )
    input_arguments: dict = {
        "prompt": {
            "type": "string",
            "description": """The prompt passed to the LLM. Must include {{key_name}}. Ask the LLM to explain concepts and optionally include  tags if useful.""",
        },
        "input_character_ranges": {
            "type": "object",
            "description": "Map keys to [start, end] ranges. Default is full text.",
            "additionalProperties": {
                "type": "array",
                "items": {
                    "type": "integer",
                },
            },
        },
    }
    required_arguments: list[str] = ["prompt"]

    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            **kwargs,
        )

    def _filter_content(self, content: str, query: str, max_chunks: int = 5) -> str:
        """
        Local Relevance Search:
        Splits a large document into chunks and returns only the chunks 
        most relevant to the query prompt.
        """
        if "--- Page" in content:
            chunks = content.split("--- Page")
            chunks = [f"--- Page{c}" for c in chunks if c.strip()]
        else:
            chunks = content.split("\n\n")

        stop_words = {"the", "a", "an", "in", "on", "at", "for", "to", "of", "and", "is", "are", "extract", "find", "summarize", "from", "document"}
        query_words = set(word.lower() for word in query.split() if word.lower() not in stop_words and len(word) > 2)

        if not query_words:
            return content[:20000] + "\n...[TRUNCATED]..."

        scored_chunks = []
        for i, chunk in enumerate(chunks):
            score = sum(1 for word in query_words if word in chunk.lower())
            
            if "revenue" in query.lower() and "revenue" in chunk.lower(): score += 2
            if "table" in chunk.lower(): score += 0.5

            scored_chunks.append((score, i, chunk))

        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        top_chunks = scored_chunks[:max_chunks]
        
        top_chunks.sort(key=lambda x: x[1])
        
        filtered_text = "\n\n... [Skipped Irrelevant Sections] ...\n\n".join([c[2] for c in top_chunks])
        
        tool_logger.info(f"Smart Filter: Reduced document from {len(content)} chars to {len(filtered_text)} chars using keywords: {query_words}")
        
        return filtered_text

    async def call_tool(
        self, arguments: dict, data_storage: dict, model: LLM, *args, **kwargs
    ) -> list[str]:
        prompt: str = arguments.get("prompt")
        input_character_ranges = arguments.get("input_character_ranges", {})
        if input_character_ranges is None:
            input_character_ranges = {}

        if not re.search(r"{{[^{}]+}}", prompt):
            raise ValueError(
                "ERROR: Prompt must include {{key_name}} from data storage."
            )

        keys = re.findall(r"{{([^{}]+)}}", prompt)
        formatted_data = {}

        for key in keys:
            if key not in data_storage:
                raise KeyError(
                    f"ERROR: Key '{key}' not found. Available: {', '.join(data_storage.keys())}"
                )

            doc_content = str(data_storage[key])

            if len(doc_content) > 20000 and key not in input_character_ranges:
                doc_content = self._filter_content(doc_content, prompt)

            if key in input_character_ranges:
                char_range = input_character_ranges[key]
                if len(char_range) == 2:
                    start_idx = int(char_range[0])
                    end_idx = int(char_range[1])
                    formatted_data[key] = doc_content[start_idx:end_idx]
                else:
                    formatted_data[key] = doc_content
            else:
                formatted_data[key] = doc_content

        formatted_prompt = re.sub(r"{{([^{}]+)}}", r"{\1}", prompt)
        
        try:
            final_prompt = formatted_prompt.format(**formatted_data)
        except KeyError as e:
            raise KeyError(f"Key error during formatting: {str(e)}")

        final_prompt_with_images = f"{IMAGE_TRIGGER_PROMPT}\n\n{final_prompt}"

        response = await model.query(final_prompt_with_images)

        return {
            "retrieval": response.output_text_str,
            "usage": {**response.metadata.model_dump()},
        }
        
def get_tool_by_name(tool_name: str):
    """
    Factory function to initialize tools by name.
    Used by the Green Agent to load tools from configuration.
    """
    # Normalize string
    name = tool_name.lower().strip()
    
    # Map strings to your actual Tool classes
    # NOTE: Ensure these class names match what is defined above in this file!
    if name == "google_web_search":
        return GoogleWebSearch() 
    elif name == "retrieve_information":
        return RetrieveInformation()
    elif name == "parse_html_page":
        return ParseHTMLPage()
    elif name == "parse_pdf":
        return ParsePDF()
    elif name == "edgar_search":
        return EdgarSearch()
    else:
        # Fallback or error
        print(f"Warning: Tool '{name}' not found in get_tool_by_name registry.")
        raise ValueError(f"Unknown tool: {name}")