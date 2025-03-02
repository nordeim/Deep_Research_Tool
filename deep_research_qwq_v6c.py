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
import asyncio
import json
import re
import logging
import tldextract
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# --- Configuration ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

# Default models
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
DEFAULT_GEMINI_MODEL = "gemini-2.0-flash"

# Domain credibility ratings
CREDIBILITY_RATINGS = {
    "edu": 0.9,
    "gov": 0.9,
    "org": 0.7,
    "wikipedia.org": 0.75,
    "medium.com": 0.5,
    "blogspot.com": 0.3,
    "wordpress.com": 0.3,
}

# --- Helper Functions ---
async def fetch_web_content(url):
    """Fetches web content using requests and BeautifulSoup with improved error handling"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        text_content = soup.get_text(separator='\n', strip=True)
        
        domain = tldextract.extract(url).registered_domain
        metadata = {
            "url": url,
            "domain": domain,
            "title": soup.title.string if soup.title else "No Title",
            "timestamp": datetime.now().isoformat(),
            "credibility_score": get_domain_credibility_score(domain)
        }
        
        return {
            "content": text_content,
            "metadata": metadata,
            "error": False,
            "status": "Success"
        }
    except Exception as e:
        logger.error(f"Error fetching URL: {url} - {str(e)}")
        return {
            "content": f"Error: {str(e)}",
            "error": True,
            "status": str(e),
            "metadata": {"url": url}
        }

def get_domain_credibility_score(domain):
    """Estimates the credibility of a domain based on TLD and known sites"""
    if domain in CREDIBILITY_RATINGS:
        return CREDIBILITY_RATINGS[domain]
    
    tld = domain.split('.')[-1] if '.' in domain else ''
    return CREDIBILITY_RATINGS.get(tld, 0.5)

def query_openai(prompt, model=DEFAULT_OPENAI_MODEL, temperature=0.7):
    """Queries OpenAI with enhanced error handling and retry logic"""
    if not OPENAI_API_KEY:
        return "OpenAI API key not configured"
        
    client = OpenAI(api_key=OPENAI_API_KEY)
    retries = 3
    
    for i in range(retries):
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a research assistant"},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
            )
            return completion.choices[0].message.content.strip()
        except Exception as e:
            if i == retries - 1:
                return f"OpenAI Error: {str(e)}"
            time.sleep(2 ** (i+1))

def query_gemini(prompt, model=DEFAULT_GEMINI_MODEL, temperature=0.7):
    """Queries Google Gemini with enhanced error handling"""
    if not GOOGLE_GEMINI_API_KEY:
        return "Gemini API key not configured"
    
    genai.configure(api_key=GOOGLE_GEMINI_API_KEY)
    retries = 3
    
    for i in range(retries):
        try:
            model_instance = genai.GenerativeModel(model)
            response = model_instance.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            if i == retries - 1:
                return f"Gemini Error: {str(e)}"
            time.sleep(2 ** (i+1))

def analyze_research_query(query, llm_choice):
    """Generates structured research analysis using LLM"""
    prompt = f"""
    Analyze this research query: "{query}"
    
    1. Identify main concepts and relationships
    2. Suggest 3-5 academic databases
    3. Identify interdisciplinary connections
    4. Create research matrix with variables
    5. Suggest validation methods
    
    Format as markdown with clear sections
    """
    
    if llm_choice == "openai":
        return query_openai(prompt, temperature=0.5)
    return query_gemini(prompt, temperature=0.5)

def generate_search_queries(base_query, num_queries, llm_choice):
    """Generates diverse search queries with academic integration"""
    prompt = f"""
    Generate {num_queries} diverse search queries for "{base_query}"
    
    Include:
    - 2 academic database queries (e.g., site:springer.com)
    - 2 technical/industry queries
    - 1 recent developments query
    - 1 contrasting viewpoints query
    
    Return numbered list without formatting
    """
    
    if llm_choice == "openai":
        response = query_openai(prompt, temperature=0.8)
    else:
        response = query_gemini(prompt, temperature=0.8)
    
    return [q.strip() for q in response.split('\n') if q.strip()]

def perform_google_search(query, num_results=8):
    """Performs Google search with improved result parsing"""
    try:
        search_url = f"https://www.google.com/search?q={urllib.parse.quote_plus(query)}&num={num_results}"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }
        response = requests.get(search_url, headers=headers, timeout=15)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        urls = []
        for a in soup.select('a[href^="/url?q="]'):
            url = urllib.parse.unquote(a['href'].split('/url?q=')[1].split('&')[0])
            if should_include_url(url):
                urls.append(url)
        
        return list(dict.fromkeys(urls))[:num_results]
    except Exception as e:
        logger.error(f"Search error: {str(e)}")
        return []

def should_include_url(url):
    """Filters out low-quality URLs"""
    excluded = [
        'google.com/aclk', 'amazon.com/s?', 'youtube.com/shorts',
        '.pdf', '.jpg', '.png', 'facebook.com', 'linkedin.com'
    ]
    return not any(ex in url for ex in excluded)

async def scrape_urls_concurrently(urls):
    """Scrapes multiple URLs concurrently with rate limiting"""
    results = {}
    semaphore = asyncio.Semaphore(5)  # Limit concurrent requests
    
    async def safe_fetch(url):
        async with semaphore:
            return await fetch_web_content(url)
    
    tasks = [asyncio.create_task(safe_fetch(url)) for url in urls]
    responses = await asyncio.gather(*tasks)
    
    for url, response in zip(urls, responses):
        results[url] = response
    return results

def extract_relevant_content(scraped_data, query):
    """Extracts and scores content relevance"""
    processed = []
    for url, data in scraped_data.items():
        if data.get('error'):
            continue
            
        content = data.get('content', '')
        if len(content) < 200:
            continue
            
        relevance = sum(content.lower().count(q) for q in query.lower().split())
        credibility = data['metadata']['credibility_score']
        score = (relevance * 0.6) + (credibility * 0.4)
        
        processed.append({
            "url": url,
            "content": content[:10000],
            "score": min(score, 1.0),
            "metadata": data['metadata']
        })
    
    processed.sort(key=lambda x: x['score'], reverse=True)
    return processed

def generate_content_summary(content_items, query, llm_choice):
    """Generates APA-style research summary"""
    if not content_items:
        return "No relevant content found"
        
    content = "\n\n".join(
        f"Source ({i+1}): {item['url']}\n{item['content'][:8000]}"
        for i, item in enumerate(content_items[:5])
    )
    
    prompt = f"""
    Analyze and synthesize this content about "{query}":
    
    {content}
    
    Create APA-style summary with:
    1. Key theoretical frameworks
    2. Empirical findings
    3. Methodological approaches
    4. Current debates
    5. Practical applications
    
    Include in-text citations by source number
    """
    
    if llm_choice == "openai":
        return query_openai(prompt, temperature=0.3)
    return query_gemini(prompt, temperature=0.3)

def generate_follow_up_questions(content_items, base_query, llm_choice):
    """Generates follow-up questions based on the research findings."""
    if not content_items:
        return "Unable to generate follow-up questions due to lack of content."
    
    # Prepare the content for analysis  # <-- Fixed indentation (4 spaces)
    combined_content = ""
    for i, item in enumerate(content_items[:3]):  # Focus on top 3 results
        combined_content += f"\n\n--- Content from {item['url']} ---\n{item['content'][:5000]}"
    
    prompt = f"""
    Based on the following research content about "{base_query}":
    
    {combined_content}
    
    Generate 3-5 insightful follow-up questions that would:
    1. Address gaps in the current information
    2. Explore important aspects not covered in the existing content
    3. Help deepen understanding of contradictions or complex aspects
    4. Explore practical implications or applications
    5. Consider alternative perspectives or approaches
    
    Return ONLY the numbered list of questions, one per line, without any additional text.
    """
    
    if llm_choice == "openai":
        response = query_openai(prompt, temperature=0.7)
    else:
        response = query_gemini(prompt, temperature=0.7)
    
    # Clean up the response
    if response.startswith("Error") or "Failed to get response" in response:
        return response
        
    questions = []
    lines = response.split("\n")
    for line in lines:
        line = line.strip()
        if not line:
            continue 
            
        # Remove numbering and any extra characters
        cleaned_line = re.sub(r'^\d+[\.\)]\s*', '', line).strip()
        if cleaned_line and cleaned_line not in questions:
            questions.append(cleaned_line)
    
    return "\n".join([f"{i+1}. {q}" for i, q in enumerate(questions)])
    
def generate_citations(sources):
    """Generates APA-formatted references"""
    citations = []
    for i, source in enumerate(sources):
        metadata = source.get('metadata', {})
        domain = metadata.get('domain', 'Unknown')
        title = metadata.get('title', 'No Title')
        date = metadata.get('timestamp', '')[:10]
        url = metadata.get('url', '')
        
        citation = f"{title}. ({date}). {domain}. Retrieved from {url}"
        citations.append(f"{i+1}. {citation}")
    return "\n".join(citations)

async def deep_research(query, llm_choice, depth=2, num_queries=5, urls=None):
    """Main research function with iterative refinement"""
    report = f"# Research Report: {query}\n\n"
    all_sources = []
    follow_up = ""
    
    # Analyze initial query
    analysis = analyze_research_query(query, llm_choice)
    report += f"## Research Analysis\n{analysis}\n\n"
    
    # Process provided URLs
    if urls:
        provided_urls = [u.strip() for u in urls.split(",") if u.strip()]
        scraped = await scrape_urls_concurrently(provided_urls)
        relevant = extract_relevant_content(scraped, query)
        if relevant:
            summary = generate_content_summary(relevant, query, llm_choice)
            report += f"## Provided URLs Analysis\n{summary}\n\n"
            all_sources.extend([r['url'] for r in relevant])
    
    for i in range(depth):
        report += f"## Iteration {i+1}\n"
        queries = generate_search_queries(query, num_queries, llm_choice)
        report += "### Search Queries\n" + "\n".join([f"- {q}" for q in queries]) + "\n\n"
        
        # Execute searches
        search_results = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_query = {executor.submit(perform_google_search, q): q for q in queries}
            for future in concurrent.futures.as_completed(future_to_query):
                try:
                    search_results.extend(future.result())
                except Exception as e:
                    logger.error(f"Search error: {str(e)}")
        
        # Scrape and process
        search_results = list(dict.fromkeys(search_results))[:15]
        scraped = await scrape_urls_concurrently(search_results)
        relevant = extract_relevant_content(scraped, query)
        all_sources.extend([r['url'] for r in relevant])
        
        # Generate outputs
        summary = generate_content_summary(relevant, query, llm_choice)
        follow_up = generate_follow_up_questions(relevant, query, llm_choice)
        
        report += f"### Key Findings\n{summary}\n\n"
        report += f"### Follow-Up Questions\n{follow_up}\n\n"
    
    # Final synthesis
    if depth > 1:
        synthesis_prompt = f"""
        Synthesize all research on "{query}" into a final report.
        Highlight: key insights, contradictions, methodology strengths,
        remaining gaps, and practical applications.
        """
        if llm_choice == "openai":
            final = query_openai(synthesis_prompt, temperature=0.3)
        else:
            final = query_gemini(synthesis_prompt, temperature=0.3)
        report += f"## Final Synthesis\n{final}\n\n"
    
    # Add references
    report += "## References\n" + generate_citations(all_sources)
    return report

# --- Gradio Interface ---
def research_handler(query, llm, depth, num_queries, urls):
    if not query:
        return "Please enter a research query"
    if llm == "openai" and not OPENAI_API_KEY:
        return "OpenAI API key not configured"
    if llm == "gemini" and not GOOGLE_GEMINI_API_KEY:
        return "Gemini API key not configured"
    
    try:
        depth = int(depth)
        num_queries = int(num_queries)
        return asyncio.run(deep_research(query, llm, depth, num_queries, urls))
    except Exception as e:
        logger.error(f"Research error: {str(e)}")
        return f"Error during research: {str(e)}"

# Gradio App
with gr.Blocks(title="Deep Research Tool") as app:
    gr.Markdown("# 🕵️ AI-Powered Research Assistant")
    
    with gr.Row():
        with gr.Column(scale=2):
            query = gr.Textbox(
                label="Research Question",
                placeholder="Enter your research topic...",
                lines=3
            )
            llm = gr.Radio(
                label="AI Model",
                choices=["openai", "gemini"],
                value="openai"
            )
            depth = gr.Slider(1, 5, 2, step=1, label="Research Depth")
            num_queries = gr.Slider(3, 10, 5, step=1, label="Queries per Iteration")
            urls = gr.Textbox(
                label="Optional URLs",
                placeholder="Comma-separated URLs to include",
                lines=2
            )
            btn = gr.Button("Start Research", variant="primary")
        
        with gr.Column(scale=3):
            output = gr.Markdown("Research results will appear here...")

    btn.click(
        fn=research_handler,
        inputs=[query, llm, depth, num_queries, urls],
        outputs=output
    )

if __name__ == "__main__":
    app.launch()
  
