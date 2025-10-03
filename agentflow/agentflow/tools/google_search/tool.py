import os
import json
import requests
from dotenv import load_dotenv
load_dotenv()

from google import genai
from google.genai import types

from agentflow.tools.base import BaseTool

# For formatting the response
import requests
from typing import List
import re

# Tool name mapping - this defines the external name for this tool
TOOL_NAME = "Ground_Google_Search_Tool"

LIMITATIONS = """
1. This tool is only suitable for general information search.
2. This tool contains less domain specific information.
3. This tools is not suitable for searching and analyzing videos at YouTube or other video platforms.
"""

BEST_PRACTICES = """
1. Choose this tool when you want to search general information about a topic.
2. Choose this tool for question type of query, such as "What is the capital of France?" or "What is the capital of France?"
3. The tool will return a summarized information.
4. This tool is more suiable for defination, world knowledge, and general information search.
"""

class Google_Search_Tool(BaseTool):
    def __init__(self, model_string="gemini-2.5-flash"):
        super().__init__(
            tool_name=TOOL_NAME,
            tool_description="A web search tool powered by Google's Gemini AI that provides real-time information from the internet with citation support.",
            tool_version="1.0.0",
            input_types={
                "query": "str - The search query to find information on the web.",
                "add_citations": "bool - Whether to add citations to the results. If True, the results will be formatted with citations. By default, it is True.",
            },
            output_type="str - The search results of the query.",
            demo_commands=[
                {
                    "command": 'execution = tool.execute(query="What is the capital of France?")',
                    "description": "Search for general information about the capital of France with default citations enabled."
                },
                {
                    "command": 'execution = tool.execute(query="Who won the euro 2024?", add_citations=False)',
                    "description": "Search for information about Euro 2024 winner without citations."
                },
                {
                    "command": 'execution = tool.execute(query="Physics and Society article arXiv August 11, 2016", add_citations=True)',
                    "description": "Search for specific academic articles with citations enabled."
                }
            ],
            user_metadata={
                "limitations": LIMITATIONS,
                "best_practices": BEST_PRACTICES,
            }
        )
        self.max_retries = 5
        self.search_model = model_string

        try:
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise Exception("Google API key not found. Please set the GOOGLE_API_KEY environment variable.")
        except Exception as e:
            raise Exception(f"Google API key not found. Please set the GOOGLE_API_KEY environment variable.")

        self.client = genai.Client(api_key=api_key)


    @staticmethod
    def get_real_url(url):
        """
        Convert a redirect URL to the final real URL in a stable manner.

        This function handles redirects by:
        1.  Setting a browser-like User-Agent to avoid being blocked or throttled.
        2.  Using a reasonable timeout to prevent getting stuck indefinitely.
        3.  Following HTTP redirects automatically (default requests behavior).
        4.  Catching specific request-related exceptions for cleaner error handling.
        """
        try:
            # Headers to mimic a real browser visit
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            # allow_redirects=True is the default, but we state it for clarity.
            # The request will automatically follow the 3xx redirect chain.
            response = requests.get(
                url, 
                headers=headers, 
                timeout=8, # Increased timeout for more reliability
                allow_redirects=True 
            )
            
            # After all redirects, response.url contains the final URL.
            return response.url
            
        except Exception as e:
            # Catching specific exceptions from the requests library is better practice.
            # print(f"An error occurred: {e}")
            return url

    @staticmethod
    def extract_urls(text: str) -> List[str]:
        """
        Extract all URLs from Markdown‑style citations [number](url) in the given text.

        Args:
            text: A string containing Markdown citations.

        Returns:
            A list of URL strings.
        """
        pattern = re.compile(r'\[\d+\]\((https?://[^\s)]+)\)')
        urls = pattern.findall(text)
        return urls

    def reformat_response(self, response: str) -> str:
        """
        Reformat the response to a readable format.
        """
        urls = self.extract_urls(response)
        for url in urls:
            direct_url = self.get_real_url(url)
            response = response.replace(url, direct_url)
        return response

    @staticmethod
    def add_citations(response):
        text = response.text
        supports = response.candidates[0].grounding_metadata.grounding_supports
        chunks = response.candidates[0].grounding_metadata.grounding_chunks

        # Sort supports by end_index in descending order to avoid shifting issues when inserting.
        sorted_supports = sorted(supports, key=lambda s: s.segment.end_index, reverse=True)

        for support in sorted_supports:
            end_index = support.segment.end_index
            if support.grounding_chunk_indices:
                # Create citation string like [1](link1)[2](link2)
                citation_links = []
                for i in support.grounding_chunk_indices:
                    if i < len(chunks):
                        uri = chunks[i].web.uri
                        citation_links.append(f"[{i + 1}]({uri})")

                citation_string = ", ".join(citation_links)
                text = text[:end_index] + citation_string + text[end_index:]

        return text

    def _execute_search(self, query: str, add_citations_flag: bool):
        """
        https://ai.google.dev/gemini-api/docs/google-search
        """
        # Define the grounding tool
        grounding_tool = types.Tool(
            google_search=types.GoogleSearch()
        )

        # Configure generation settings
        config = types.GenerateContentConfig(
            tools=[grounding_tool]
        )
        

        response = None
        response_text = None
        
        for attempt in range(self.max_retries):
            try:
                response = self.client.models.generate_content(
                    model=self.search_model,
                    contents=query,
                    config=config,
                )
                response_text = response.text
                # If we get here, the API call was successful, so break out of the retry loop
                break
            except Exception as e:
                print(f"Google Search attempt {attempt + 1} failed: {str(e)}. Retrying...")
                if attempt == self.max_retries - 1:  # Last attempt
                    print(f"Google Search failed after {self.max_retries} attempts. Last error: {str(e)}")
                    return f"Google Search tried {self.max_retries} times but failed. Last error: {str(e)}"
                # Continue to next attempt

        # Check if we have a valid response before proceeding
        if response is None or response_text is None:
            return "Google Search failed to get a valid response"

        # Add citations if needed
        try:
            response_text = self.add_citations(response) if add_citations_flag else response_text
        except Exception as e:
            pass
            # print(f"Error adding citations: {str(e)}")
            # Continue with the original response_text if citations fail

        # Format the response
        try:
            response_text = self.reformat_response(response_text)
        except Exception as e:
            pass
            # print(f"Error reformatting response: {str(e)}")
            # Continue with the current response_text if reformatting fails

        return response_text

    def execute(self, query: str, add_citations: bool = True):
        """
        Execute the Google search tool.

        Parameters:
            query (str): The search query to find information on the web.
            add_citations (bool): Whether to add citations to the results. Default is True.

        Returns:
            str: The search results of the query.
        """
        # Perform the search
        response = self._execute_search(query, add_citations)
        
        return response

    def get_metadata(self):
        """
        Returns the metadata for the Google_Search tool.

        Returns:
            dict: A dictionary containing the tool's metadata.
        """
        metadata = super().get_metadata()
        return metadata


if __name__ == "__main__":
    """
    Test:
    cd agentflow/tools/google_search
    python tool.py
    """
    def print_json(result):
        import json
        print(json.dumps(result, indent=4))

    google_search = Google_Search_Tool()

    # Get tool metadata
    metadata = google_search.get_metadata()
    print("Tool Metadata:")
    print_json(metadata)

    examples = [
        {'query': 'What is the capital of France?', 'add_citations': True},
        {'query': 'Who won the euro 2024?', 'add_citations': False},
        {'query': 'Physics and Society article arXiv August 11, 2016', 'add_citations': True},
    ]
    
    for example in examples:
        print(f"\nExecuting search: {example['query']}")
        try:
            result = google_search.execute(**example)
            print("Search Result:")
            print(result)
        except Exception as e:
            print(f"Error: {str(e)}")
        print("-" * 50)

    print("Done!")