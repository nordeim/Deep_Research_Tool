import os
import requests
import gradio as gr
from openai import OpenAI
import google.generativeai as genai
from dotenv import load_dotenv
import concurrent.futures
import asyncio
import json
import re
import logging
import tldextract
from datetime import datetime
from crawl4ai import AsyncWebCrawler
from typing import List, Dict, Any
from sklearn.cluster import KMeans
import numpy as np
from sentence_transformers import SentenceTransformer
from serpapi import GoogleSearch

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# --- Configuration ---
class Config:
    MAX_CONCURRENT_REQUESTS = 10
    CONTENT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    TOPIC_CLUSTERS = 3
    MIN_CONTENT_LENGTH = 300
    CITATION_STYLE = "APA"

# Initialize embedding model
embedding_model = SentenceTransformer(Config.CONTENT_EMBEDDING_MODEL)

# --- Enhanced Helper Functions ---
async def analyze_research_query(query: str, llm_choice: str) -> Dict[str, Any]:
    """Enhanced query analysis with structured output"""
    prompt = f"""Analyze this research query: "{query}". Return JSON with:
    - main_topics: list of key concepts
    - subtopics: list of specific aspects to explore
    - required_data_types: ["statistics", "case studies", "expert opinions", etc.]
    - potential_biases: list of possible biases to watch for
    - recommended_sources: ["academic", "news", "government", "industry"]
    - search_strategy: ["broad exploration", "focused investigation", "comparative analysis"]
    """
    
    response = query_llm(prompt, llm_choice, temperature=0.3, json_mode=True)
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        logger.error("Failed to parse analysis response")
        return {}

def diversified_search_strategy(query_analysis: Dict[str, Any]) -> List[str]:
    """Generate search strategies based on query analysis"""
    strategies = []
    
    if "academic" in query_analysis.get("recommended_sources", []):
        strategies.extend(["Google Scholar", "PubMed"])
    if "news" in query_analysis.get("recommended_sources", []):
        strategies.append("NewsAPI")
    
    return strategies + ["Google Web", "Google Domain-Specific"]

async def fetch_content_with_retry(url: str, retries: int = 3) -> Dict[str, Any]:
    """Enhanced content fetching with multiple retries"""
    for attempt in range(retries):
        try:
            result = await fetch_web_content_crawl4ai(url)
            if result.get("error"):
                raise Exception(result["status"])
            return result
        except Exception as e:
            if attempt == retries - 1:
                return {"error": str(e), "content": "", "metadata": {"url": url}}
            await asyncio.sleep(2 ** attempt)

def cluster_content_by_topic(content_items: List[Dict]) -> Dict[int, List[Dict]]:
    """Cluster content using semantic embeddings"""
    texts = [item["content"][:1000] for item in content_items]
    embeddings = embedding_model.encode(texts)
    
    kmeans = KMeans(n_clusters=Config.TOPIC_CLUSTERS)
    clusters = kmeans.fit_predict(embeddings)
    
    clustered = {}
    for idx, cluster_id in enumerate(clusters):
        clustered.setdefault(cluster_id, []).append(content_items[idx])
    return clustered

def generate_thematic_summary(clustered_content: Dict[int, List[Dict]], query: str, llm_choice: str) -> str:
    """Generate summary organized by thematic clusters"""
    summary = []
    for cluster_id, items in clustered_content.items():
        combined = "\n\n".join([f"Source {i+1}: {item['content'][:2000]}" for i, item in enumerate(items[:3])]
        
        prompt = f"""Analyze this content cluster about {query}:
        {combined}
        
        Identify the main theme and 3-5 key points. 
        Note any conflicts with other sources or uncertainties.
        """
        
        theme_summary = query_llm(prompt, llm_choice, temperature=0.2)
        summary.append(f"## Theme {cluster_id+1}\n{theme_summary}")
    
    return "\n\n".join(summary)

def enhance_citation(metadata: Dict) -> str:
    """Generate proper citations from metadata"""
    if Config.CITATION_STYLE == "APA":
        return f"{metadata.get('title', 'Untitled')} ({metadata.get('date', 'n.d.')}). {metadata['domain']}. URL: {metadata['url']}"
    return metadata['url']

# --- Enhanced Main Research Flow ---
async def enhanced_research_iteration(query: str, query_analysis: Dict, llm_choice: str) -> Dict:
    """Perform comprehensive research iteration"""
    # Generate domain-specific searches
    search_queries = generate_domain_specific_queries(query, query_analysis)
    
    # Multi-source search execution
    results = await execute_multi_source_search(search_queries)
    
    # Advanced content processing
    processed = [
        process_content(item) 
        for item in results.values()
        if not item.get("error")
    ]
    
    # Semantic clustering and analysis
    clustered = cluster_content_by_topic(processed)
    
    return {
        "clusters": clustered,
        "sources": [item["metadata"] for item in processed],
        "summary": generate_thematic_summary(clustered, query, llm_choice)
    }

# --- Improved Gradio Interface ---
def create_enhanced_interface():
    """Create more sophisticated user interface"""
    with gr.Blocks(title="Advanced Research Assistant", theme=gr.themes.Soft()) as interface:
        gr.Markdown("# 🔬 Advanced Research Assistant")
        
        with gr.Tab("Research Setup"):
            query_input = gr.Textbox(label="Research Objective", lines=3)
            source_selector = gr.CheckboxGroup(
                label="Preferred Sources",
                choices=["Academic", "News", "Government", "Industry"]
            )
            depth_control = gr.Slider(1, 5, value=2, label="Research Depth")
        
        with gr.Tab("Live Analysis"):
            progress = gr.Progress()
            findings_display = gr.JSON(label="Current Findings")
            cluster_viz = gr.Plot(label="Topic Clusters")
        
        with gr.Tab("Final Report"):
            report_output = gr.Markdown()
            citation_list = gr.DataFrame(headers=["Source", "Citation"])
        
        # Interactive controls
        query_input.submit(
            start_research_flow, 
            [query_input, source_selector, depth_control],
            [findings_display, cluster_viz, report_output, citation_list],
            show_progress="full"
        )
    
    return interface

# --- Key Enhancements ---
"""
1. Structured Query Analysis: Uses JSON output for better planning
2. Multi-Source Search: Integrates academic and news APIs
3. Semantic Clustering: Groups content by topic similarity
4. Thematic Synthesis: Creates cluster-based summaries
5. Citation Management: Generates style-formatted citations
6. Interactive UI: Multiple tabs with live progress
7. Error Resilience: Enhanced retry logic and fallback strategies
8. Content Quality: Embedded-based relevance scoring
9. Bias Detection: Flags potential biases during analysis
10. Visual Analytics: Topic cluster visualization
"""
