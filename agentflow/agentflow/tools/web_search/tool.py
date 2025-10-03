import os
import numpy as np
import openai
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

from agentflow.tools.base import BaseTool
from agentflow.engine.factory import create_llm_engine

load_dotenv()

# Tool name mapping - this defines the external name for this tool
TOOL_NAME = "Web_RAG_Search_Tool"

LIMITATION = f"""
The {TOOL_NAME} has several limitations: 
1) Requires valid URLs that are accessible and contain text content. 
2) May not work with JavaScript-heavy websites or those requiring authentication. 
3) Performance depends on the quality and relevance of the website content. 
4) May return incomplete or inaccurate information if the website content is not comprehensive. 
5) Limited by the chunking and embedding process which may miss context. 
6) Requires OpenAI API access for embeddings and LLM generation.
"""

BEST_PRACTICE = f"""
For optimal results with the {TOOL_NAME}:
1) Use specific, targeted queries rather than broad questions.
2) Ensure the URL is accessible and contains relevant information.
3) Prefer websites with well-structured, text-rich content.
4) For complex queries, break them down into smaller, specific questions.
5) Verify important information from multiple sources when possible.
6) Use it as part of a multi-step research process rather than a single source of truth.
7) It is highly recommended to use this tool after calling other web-based tools (e.g., Google_Search_Tool, Wiki_Search_Tool, etc.) to get the real, accessible URLs.
"""


SUMMARIZE_PROMPT_TEMPLATE = """
You are an expert AI assistant. Your task is to provide a clear, concise, and accurate answer to the user's query based **exclusively** on the provided reference information.

## Step-by-Step Instructions
1.  **Analyze the Query:** First, fully understand the user's query and identify the specific information being asked for.
2.  **Scan for Relevance:** Read through each numbered chunk in the reference information. Identify all chunks that contain information directly relevant to answering the query. A simple keyword match is not sufficient; the chunk must contain a substantive fact that helps answer the question.
3.  **Extract Key Facts & Synthesize:** From the relevant chunks, extract only the key facts and figures needed. Synthesize these extracted facts into a comprehensive, single-paragraph answer. Write the answer in your own words. **Do not** copy entire chunks.

## Output Format and Example

**IMPORTANT:** You must follow this format exactly.

### Example Input
- **User Query:** What were the key financial results for Q4 2023?
- **Reference Information:**
[1] The company's new "Project Starlight" initiative launched in January 2024.
[2] In Q4 2023, the company reported a total revenue of $5.2 million and a net profit of $800,000. This was a 15% increase in revenue compared to Q3 2023.
[3] Marketing spend in Q4 2023 was focused on digital channels, totaling $450,000.
[4] The CEO stated that the strong Q4 performance was driven by robust sales in the North American market.

### Example Output
Answer:
In the fourth quarter of 2023, the company achieved a total revenue of $5.2 million, which represented a 15% increase from the previous quarter, and a net profit of $800,000. The strong performance was attributed to robust sales in the North American market. The marketing expenditure for this period was $450,000.

---
## Your Turn

### User Query
{query}

### Reference Information
{reference_information}

### Output
"""

class Web_Search_Tool(BaseTool):
    require_llm_engine = True

    def __init__(self, model_string="gpt-4o-mini"):
        super().__init__(
            tool_name=TOOL_NAME,
            tool_description="A specialized tool for answering questions by retrieving relevant information from a given website using RAG (Retrieval-Augmented Generation).",
            tool_version="1.0.0",
            input_types={
                "query": "str - The search query for the website.",
                "url": "str - The URL of the website to retrieve information from.",
            },
            output_type="str - The answer to the user's query based on the information gathered from the website.",
            demo_commands=[
                {
                    "command": 'execution = tool.execute(query="What is the exact mass in kg of the moon?", url="https://en.wikipedia.org/wiki/Moon")',
                    "description": "Retrieve information about the moon's mass from Wikipedia."
                },
                {
                    "command": 'execution = tool.execute(query="What are the main features of Python programming language?", url="https://www.python.org/about/apps/")',
                    "description": "Get information about Python features from the official website."
                }
            ],
            user_metadata = {
                "limitation": LIMITATION,
                "best_practice": BEST_PRACTICE
            }
        )

        # self.model_string = "gpt-4o-mini" # NOTE: strong LLM for tool
        # self.model_string = "gemini-1.5-flash" # NOTE: weak 8B model for tool
        # self.model_string = "dashscope" # NOTE: weak Qwen2.5-7B model for tool

        self.model_string = model_string
        print(f"Initializing Website RAG Tool with model: {self.model_string}")
        self.chunk_size = 200
        self.chunk_overlap = 20
        self.top_k = 10
        # self.embeddings_model = "text-embedding-3-large" # or "text-embedding-3-small" for efficiency
        self.embeddings_model = "text-embedding-3-small"
        self.max_window_size = 1000000

        # NOTE: deterministic mode
        self.llm_engine = create_llm_engine(
            model_string=self.model_string, 
            temperature=0.0, 
            top_p=1.0, 
            frequency_penalty=0.0, 
            presence_penalty=0.0
            )

    def _get_website_content(self, url):
        """ 
        Extracts all text from the given URL.

        Parameters:
            url (str): The URL from which to extract text.

        Returns:
            str: The extracted text.
        """
        url = url.replace("arxiv.org/pdf", "arxiv.org/abs")

        # Add headers to mimic a real browser request
        # NOTE: this is a workaround to avoid being blocked by the website
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }

        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            text = soup.get_text(separator='\n', strip=True)
            text = text[:self.max_window_size] # Limit the text to max_window_size characters
            return text
        except requests.RequestException as e:
            return f"Error fetching URL: {str(e)}"
        except Exception as e:
            return f"Error extracting text: {str(e)}"

    def _chunk_website_content(self, content):
        """
        Chunks the website content into smaller chunks based on the chunk size and overlap.
        Parameters:
            content (str): The website content to chunk.
        Returns:
            list: A list of chunks.
        """
        # Split the content string by whitespace characters
        words = content.split()
        ptr = 0
        chunks = []
        while True:
            start, end = ptr, min(ptr + self.chunk_size, len(words))
            chunk = " ".join(words[start:end])
            chunks.append(chunk)
            if end >= len(words):
                break
            ptr = end - self.chunk_overlap
        return chunks

    def _embed_strings(self, strings):
        """
        Embed the strings using OpenAI's embedding model.
        Parameters:
            strings (list): A list of strings to embed.
        Returns:
            list: A list of embeddings.
        """
        try:
            client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            embeddings = client.embeddings.create(
                input=strings,
                model=self.embeddings_model
            )
            res = [embedding.embedding for embedding in embeddings.data]
            return res
        except Exception as e:
            raise Exception(f"Error embedding strings: {str(e)}")

    def _cosine_similarity(self, a, b):
        """
        Calculate the cosine similarity between two vectors.
        """
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def _rank_chunks(self, query_embedding, chunk_embeddings):
        """
        Rank the chunks based on the query embedding.
        Parameters:
            query_embedding (list): The embedding of the query.
            chunk_embeddings (list): The embeddings of the chunks.
        Returns:
            list: The indices of the ranked chunks in descending order of similarity.
        """
        similarities = [self._cosine_similarity(query_embedding, chunk_embedding) for chunk_embedding in chunk_embeddings]
        return list(np.argsort(similarities)[::-1])

    def _concatenate_chunks(self, chunks):
        """
        Concatenate the chunks into a single string.
        """
        for i, chunk in enumerate(chunks):
            chunks[i] = f"Chunk [{i+1}]\n{chunk}"
        return "\n".join(chunks)

    def _construct_final_output(self, query, reference_information):
        """
        Construct the final output from the top chunks.
        """
        summary_prompt = SUMMARIZE_PROMPT_TEMPLATE.format(
            query=query,
            reference_information=reference_information
        )
        
        summary = self.llm_engine(summary_prompt)
        return summary

    def execute(self, query, url):
        try:
            # step 1: get content from the website
            website_content = self._get_website_content(url)
            
            if website_content.startswith("Error"):
                return website_content

            # step 2: chunk the content
            chunks = self._chunk_website_content(website_content)
            
            if not chunks:
                return "Error: No content could be extracted from the website."

            # step 3: embed the chunks
            embeddings = self._embed_strings([query] + chunks)
            query_embedding = embeddings[0]
            chunk_embeddings = embeddings[1:]

            # step 4: rank the chunks
            ranked_chunks = self._rank_chunks(query_embedding, chunk_embeddings)
            top_chunks = [chunks[i] for i in ranked_chunks[:self.top_k]]

            # step 5: summarize the top chunks
            reference_string = self._concatenate_chunks(top_chunks)
            summary = self._construct_final_output(query, reference_string)

            return summary
        except Exception as e:
            return f"Error processing request: {str(e)}"

    def get_metadata(self):
        metadata = super().get_metadata()
        # metadata['require_llm_engine'] = self.require_llm_engine
        return metadata

if __name__ == "__main__":
    # Test command:
    """
    Run the following commands in the terminal to test the script:
    
    cd agentflow/tools/web_search
    python tool.py
    """

    import json

    # Example usage of the Web_Search_Tool
    tool = Web_Search_Tool(model_string="gpt-4o-mini") # NOTE: strong LLM for tool
    # tool = Web_Search_Tool(model_string="gemini-1.5-flash") # NOTE: weak 8B model for tool
    # tool = Web_Search_Tool(model_string="dashscope") # NOTE: weak Qwen2.5-7B model for tool

    # Get tool metadata
    metadata = tool.get_metadata()
    # print("Tool Metadata:")
    # print(json.dumps(metadata, indent=4))

    examples = [
        {
            "query": "What is the exact mass in kg of the moon?", 
            "url": "https://en.wikipedia.org/wiki/Moon"
        },
        {
            "query": "What is the capital of France?", 
            "url": "https://en.wikipedia.org/wiki/France"
        },
        {
            "query": "What are the main features of Python programming language?", 
            "url": "https://www.python.org/about/apps/"
        }
    ]

    for example in examples:
        try:
            # Execute the tool with example query
            execution = tool.execute(**example)
            print("\nGenerated Response:")
            print(execution)
            print("\n")
        except Exception as e:
            print(f"Execution failed: {e}")


    print("\nDone!")