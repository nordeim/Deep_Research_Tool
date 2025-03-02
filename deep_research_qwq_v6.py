import os
import requests
from bs4 import BeautifulSoup
import gradio as gr
from openai import OpenAI
import google.generativeai as genai
from dotenv import load_dotenv
import concurrent.futures
import time
import urllib.parse
import json
import re
import logging
import tldextract
from datetime import datetime
from crawl4ai import AsyncWebCrawler
from scholarly import scholarly  # For academic source integration

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# --- Enhanced Configuration ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
ACADEMIC_SOURCES_ENABLED = os.getenv("ACADEMIC_SOURCES_ENABLED", "True").lower() == "true"

# Domain credibility ratings with expanded academic sources
CREDIBILITY_RATINGS = {
    "edu": 0.95,
    "gov": 0.9,
    "org": 0.75,
    "wikipedia.org": 0.8,
    "springer.com": 0.9,
    "jstor.org": 0.85,
    "sciencedirect.com": 0.85,
    "ieee.org": 0.85,
    "medium.com": 0.5,
    "blogspot.com": 0.3,
}

# --- Enhanced Helper Functions ---
async def fetch_web_content_crawl4ai(url):
    """Enhanced content fetching with academic source handling"""
    try:
        async with AsyncWebCrawler() as crawler:
            result = await crawler.arun(url=url)
            if not result.markdown and not result.text:
                return {"content": f"No content extracted from {url}", "error": True}
            
            domain = tldextract.extract(url).registered_domain
            metadata = {
                "url": url,
                "domain": domain,
                "title": result.title,
                "timestamp": datetime.now().isoformat(),
                "credibility_score": get_domain_credibility_score(domain),
                "content_type": "academic" if is_academic_domain(domain) else "general"
            }
            
            return {"content": result.markdown or result.text, "metadata": metadata, "error": False}
    except Exception as e:
        logger.error(f"Error fetching URL: {url} - {str(e)}")
        return {"content": f"Error: {str(e)}", "error": True, "metadata": {"url": url}}

def is_academic_domain(domain):
    """Check if domain is academic"""
    return any(academic in domain for academic in ["edu", "springer", "jstor", "sciencedirect", "ieee"])

def get_domain_credibility_score(domain):
    """Enhanced credibility scoring with academic checks"""
    if ACADEMIC_SOURCES_ENABLED and is_academic_domain(domain):
        return 0.9  # Automatic high score for verified academic domains
        
    if domain in CREDIBILITY_RATINGS:
        return CREDIBILITY_RATINGS[domain]
        
    tld = domain.split('.')[-1] if '.' in domain else ''
    return CREDIBILITY_RATINGS.get(tld, 0.5)

def query_openai(prompt, model="gpt-4o-mini", temperature=0.7, system_message=None):
    """Enhanced OpenAI query with academic source integration"""
    if ACADEMIC_SOURCES_ENABLED:
        system_message = f"{system_message or ''} When appropriate, incorporate findings from academic databases."
    
    # ... (rest of existing implementation)

def analyze_research_query(query, llm_choice):
    """Enhanced query analysis with academic source suggestions"""
    prompt = f"""
    Analyze the research query: "{query}"
    
    1. Identify main concepts and relationships
    2. Suggest 3-5 academic databases relevant to the topic
    3. Identify potential interdisciplinary connections
    4. Create a research matrix with key variables
    5. Suggest validation methods for findings
    """
    # ... (rest of existing implementation)

def generate_search_queries(base_query, num_queries, llm_choice, research_analysis=""):
    """Enhanced query generation with academic integration"""
    prompt = f"""
    Generate {num_queries} diverse search queries for "{base_query}"
    
    Include:
    - 2 academic database specific queries (e.g., "site:springer.com")
    - 2 technical/industry focused queries
    - 1 query specifically for recent developments
    - 1 query for contrasting viewpoints
    """
    # ... (rest of existing implementation)

def perform_google_search(query, num_results=10):
    """Enhanced search with academic source prioritization"""
    try:
        # ... (existing implementation)
        
        # Prioritize academic results
        prioritized_urls = []
        for url in urls:
            if any(academic in url for academic in ["springer", "jstor", "sciencedirect"]):
                prioritized_urls.insert(0, url)
            else:
                prioritized_urls.append(url)
        
        return prioritized_urls[:num_results]
    except Exception as e:
        logger.error(f"Search error: {str(e)}")
        return []

def extract_relevant_content(scraped_data, query):
    """Enhanced content extraction with academic prioritization"""
    processed = []
    for url, data in scraped_data.items():
        if data.get("error"):
            continue
            
        content = data.get("content", "")
        if len(content) < 200:
            continue
            
        metadata = data.get("metadata", {})
        relevance = calculate_relevance(content, query)
        credibility = metadata.get("credibility_score", 0.5)
        
        # Boost academic sources
        if metadata.get("content_type") == "academic":
            credibility += 0.2  # Max 1.0
            
        score = (relevance * 0.6) + (credibility * 0.4)
        processed.append({
            "url": url,
            "content": content[:10000],
            "score": min(score, 1.0),
            "metadata": metadata
        })
    
    processed.sort(key=lambda x: x["score"], reverse=True)
    return processed

def generate_content_summary(content_items, query, llm_choice):
    """Enhanced summary with academic integration"""
    academic_content = []
    general_content = []
    for item in content_items:
        if item["metadata"].get("content_type") == "academic":
            academic_content.append(item)
        else:
            general_content.append(item)
    
    prompt = f"""
    Synthesize research on "{query}" using:
    
    Academic sources:
    {format_content_list(academic_content[:3])}
    
    General sources:
    {format_content_list(general_content[:5])}
    
    Create a structured summary with:
    1. Key theoretical frameworks
    2. Empirical findings
    3. Methodological approaches
    4. Current debates
    5. Practical applications
    """
    # ... (rest of existing implementation)

def generate_citations(sources):
    """Generate formatted citations in APA style"""
    citations = []
    for source in sources:
        metadata = source.get("metadata", {})
        domain = metadata.get("domain", "")
        title = metadata.get("title", "No Title")
        url = metadata.get("url", "")
        date = metadata.get("timestamp", "")[:10]
        
        if "doi.org" in url:
            citation = f"{title}. ({date}). Retrieved from {url}"
        else:
            citation = f"{title}. ({date). {domain}. Retrieved from {url}"
        
        citations.append(citation)
    return "\n\n".join(citations)

# --- Enhanced Gradio Interface ---
def gradio_research_handler(query, llm_choice, depth, num_queries, urls, include_academic):
    """Handler with academic toggle and enhanced output"""
    global ACADEMIC_SOURCES_ENABLED
    ACADEMIC_SOURCES_ENABLED = include_academic
    
    # ... (rest of existing implementation)
    
    # Add citation section
    report += "\n\n## References\n" + generate_citations(all_sources)
    
    return report

if __name__ == '__main__':
    with gr.Blocks(title="AI Deep Research Tool") as interface:
        # ... (existing interface elements)
        
        # Add academic toggle
        with gr.Row():
            include_academic = gr.Checkbox(
                label="Include Academic Sources", 
                value=True
            )
        
        # Modify submit button click handler
        submit_button.click(
            fn=gradio_research_handler, 
            inputs=[query_input, llm_choice, depth_input, num_queries_input, urls_input, include_academic], 
            outputs=output_text
        )
        
        # Add progress bar and status indicator
        progress_bar = gr.Progress(track_tqdm=True)
        status_text = gr.Textbox(label="Status", interactive=False)
        
        # ... (rest of interface)
