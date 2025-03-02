# pip install openai google-generativeai sentence-transformers scikit-learn tldextract requests gradio python-dotenv
# Create .env file with: OPENAI_API_KEY=your_key \nGEMINI_API_KEY=your_key
import os
import requests
import gradio as gr
from openai import OpenAI
import google.generativeai as genai
from dotenv import load_dotenv
import asyncio
import json
import logging
import tldextract
from datetime import datetime
from typing import List, Dict, Any
import numpy as np
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer

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

# Initialize models
embedding_model = SentenceTransformer(Config.CONTENT_EMBEDDING_MODEL)
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# --- Core Functions ---
def query_llm(prompt: str, llm_choice: str, temperature: float = 0.7, json_mode: bool = False) -> str:
    """Unified LLM query interface"""
    try:
        if llm_choice == "openai":
            response = openai_client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                response_format={"type": "json_object"} if json_mode else None
            )
            return response.choices[0].message.content
        elif llm_choice == "gemini":
            model = genai.GenerativeModel('gemini-pro')
            response = model.generate_content(prompt)
            return response.text
        return "Invalid LLM choice"
    except Exception as e:
        logger.error(f"LLM query failed: {str(e)}")
        return f"Error: {str(e)}"

async def fetch_web_content(url: str) -> Dict[str, Any]:
    """Simplified web content fetcher"""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return {
            "content": response.text[:5000],  # Limit content size
            "metadata": {
                "url": url,
                "domain": tldextract.extract(url).registered_domain,
                "timestamp": datetime.now().isoformat()
            }
        }
    except Exception as e:
        logger.error(f"Failed to fetch {url}: {str(e)}")
        return {"error": str(e), "content": "", "metadata": {"url": url}}

def cluster_content(content_items: List[Dict]) -> Dict[int, List[Dict]]:
    """Semantic content clustering"""
    texts = [item["content"][:1000] for item in content_items]
    embeddings = embedding_model.encode(texts)
    
    kmeans = KMeans(n_clusters=Config.TOPIC_CLUSTERS)
    clusters = kmeans.fit_predict(embeddings)
    
    clustered = {}
    for idx, cluster_id in enumerate(clusters):
        clustered.setdefault(cluster_id, []).append(content_items[idx])
    return clustered

def generate_thematic_summary(clustered: Dict[int, List[Dict]], query: str, llm_choice: str) -> str:
    """Generate cluster-based summary"""
    summary = []
    for cluster_id, items in clustered.items():
        combined = "\n\n".join(
            [f"Source {i+1}: {item['content'][:2000]}" 
             for i, item in enumerate(items[:3])]
        )
        
        prompt = f"""Analyze this content cluster about {query}:
        {combined}
        
        Identify the main theme and 3-5 key points. 
        Note any conflicts with other sources or uncertainties.
        """
        
        theme_summary = query_llm(prompt, llm_choice, temperature=0.2)
        summary.append(f"## Theme {cluster_id+1}\n{theme_summary}")
    
    return "\n\n".join(summary)

# --- Gradio Interface ---
def create_interface():
    with gr.Blocks(title="Research Assistant", theme=gr.themes.Soft()) as interface:
        gr.Markdown("# 🔍 Advanced Research Assistant")
        
        with gr.Row():
            query_input = gr.Textbox(label="Research Query", lines=3)
            llm_choice = gr.Radio(["openai", "gemini"], label="AI Model", value="openai")
            
        with gr.Row():
            depth = gr.Slider(1, 3, value=2, label="Research Depth")
            submit_btn = gr.Button("Start Research", variant="primary")
            
        output_report = gr.Markdown()
        
        submit_btn.click(
            fn=lambda q, l, d: f"# Research Initiated\nQuery: {q}\nModel: {l}\nDepth: {d}",
            inputs=[query_input, llm_choice, depth],
            outputs=output_report
        )
    
    return interface

# --- Main Execution ---
if __name__ == "__main__":
    interface = create_interface()
    interface.launch()
    
