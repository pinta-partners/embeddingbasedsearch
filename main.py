import os
import json
import logging
import numpy as np
import torch
import uuid
import traceback

from scipy.spatial.distance import cosine
from flask import Flask, request, jsonify, render_template, send_file
from transformers import AutoTokenizer, AutoModel

# If you store your Anthropic key in a .env file, uncomment these lines:
# from dotenv import load_dotenv
# load_dotenv()

import anthropic  # pip install anthropic

# Setup basic logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Configuration
CONFIG = {
    "BASE_OUTPUT_DIR": "torah_search_data",
    "MAX_TOKENS": 512,
    "MIN_CHARS": 50,
    "EMBEDDING_MODEL_NAME": "dicta-il/BEREL_2.0",  # Example embedding model name
}

# Folder to hold generated result files
RESULTS_FOLDER = "results_files"
os.makedirs(RESULTS_FOLDER, exist_ok=True)

app = Flask(__name__)

# -----------------------------
# Load the embedding model and tokenizer once.
# -----------------------------
tokenizer = AutoTokenizer.from_pretrained(CONFIG["EMBEDDING_MODEL_NAME"])
model = AutoModel.from_pretrained(CONFIG["EMBEDDING_MODEL_NAME"])
model.eval()

def embed_text(text: str) -> np.array:
    """Embed text using the model with mean pooling."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    embedding = outputs.last_hidden_state.mean(dim=1).detach().numpy()[0]
    return embedding

def load_embeddings_and_metadata(chumash: str, parsha: str):
    dir_path = os.path.join(CONFIG["BASE_OUTPUT_DIR"], chumash, parsha)
    embeddings_path = os.path.join(dir_path, "embeddings.npy")
    metadata_path = os.path.join(dir_path, "metadata.json")
    if not os.path.exists(embeddings_path) or not os.path.exists(metadata_path):
        raise Exception(f"Missing embeddings or metadata in {dir_path}")
    embeddings = np.load(embeddings_path)
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    # If metadata is a dict with a "paragraphs" key, extract the list.
    if isinstance(metadata, dict) and "paragraphs" in metadata:
        metadata = metadata["paragraphs"]

    n_embeddings = embeddings.shape[0]
    n_metadata = len(metadata)
    if n_embeddings != n_metadata:
        logger.warning(
            f"Mismatch between embeddings count ({n_embeddings}) and metadata count ({n_metadata}). "
            "Using the minimum of both."
        )
        n = min(n_embeddings, n_metadata)
        embeddings = embeddings[:n]
        metadata = metadata[:n]

    return embeddings, metadata

def count_tokens(text: str) -> int:
    """Estimate token count using whitespace splitting."""
    return len(text.split())

def save_results_to_file(data: dict) -> str:
    """Save results to a JSON file; return a unique file ID."""
    unique_id = str(uuid.uuid4())
    file_path = os.path.join(RESULTS_FOLDER, f"{unique_id}.json")
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return unique_id

# -----------------------------
# Claude Analysis Function (with Citations)
# -----------------------------
def get_claude_analysis(query: str, chunk: str, chunk_title: str) -> dict:
    """
    Calls Anthropic's Claude with a custom content document for the chunk,
    enabling citations so Claude can cite relevant text.
    """
    try:
        logger.debug("Initializing Anthropic client...")
        client = anthropic.Anthropic()
        if not os.getenv('ANTHROPIC_API_KEY'):
            logger.error("ANTHROPIC_API_KEY environment variable is not set")
            return {"error": "API key not configured"}

        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "document", 
                            "source": {
                                "type": "content",
                                "content": [
                                    {"type": "text", "text": chunk}
                                ]
                            },
                            "title": chunk_title,
                            "context": "Torah chunk for analysis",
                            "citations": {"enabled": True}
                        },
                        {
                            "type": "text",
                            "text": f"Please analyze this Torah chunk in relation to the query: \"{query}\".\nProvide insights about the connections and relevance of these passages.\nUse citations to support your analysis."
                        }
                    ]
                }
            ],
            stream=False
        )
        logger.debug("Claude raw response:")
        logger.debug(response)

        if response and hasattr(response, 'content'):
            formatted_blocks = []
            for block in response.content:
                if hasattr(block, 'text'):
                    block_data = {
                        "text": block.text,
                        "citations": []
                    }
                    
                    citations = getattr(block, 'citations', [])
                    if citations:
                        for citation in citations:
                            if hasattr(citation, 'start_char_index') and hasattr(citation, 'end_char_index'):
                                block_data["citations"].append({
                                    "cited_text": block.text[citation.start_char_index:citation.end_char_index],
                                    "start_index": citation.start_char_index,
                                    "end_index": citation.end_char_index
                                })
                    
                    formatted_blocks.append(block_data)

            return {
                "analysis_blocks": formatted_blocks,
                "raw_text": chunk,
                "success": True
            }
        else:
            logger.warning("Could not extract content from Claude API response")
            return {"error": "No analysis available", "success": False}
    except Exception as e:
        logger.error(f"Error calling Claude API: {e}")
        logger.debug(traceback.format_exc())
        return {"error": str(e), "success": False}

# -----------------------------
# Main Search & Analyze Function
# -----------------------------
def search_and_analyze(query: str, chumash: str, parsha: str, top_k: int):
    embeddings, metadata = load_embeddings_and_metadata(chumash, parsha)
    logger.debug(f"Loaded embeddings with shape {embeddings.shape} and {len(metadata)} metadata entries.")

    # 1. Embed the query.
    query_emb = embed_text(query)

    # 2. Compute cosine similarity between the query and each passage.
    similarities = []
    for emb in embeddings:
        if np.linalg.norm(query_emb) == 0 or np.linalg.norm(emb) == 0:
            sim = 0.0
        else:
            sim = 1 - cosine(query_emb, emb)
        similarities.append(sim)

    # 3. Get indices of top-K passages.
    sorted_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)
    top_indices = sorted_indices[:top_k]
    results = []
    for i in top_indices:
        if i >= len(metadata):
            continue
        res = metadata[i].copy()
        # Convert NumPy float to native Python float
        res["score"] = float(similarities[i])
        results.append(res)

    # 4. Combine the selected passages into one document.
    passages = [res["text"] for res in results if "text" in res]
    combined_text = "\n\n".join(passages)
    total_tokens = count_tokens(combined_text)
    logger.debug(f"Total tokens in combined passages: {total_tokens}")

    # 5. Split the document if it exceeds 150,000 tokens
    max_tokens_per_chunk = 150000
    chunks = []
    if total_tokens > max_tokens_per_chunk:
        current_chunk = ""
        current_tokens = 0
        for para in passages:
            para_tokens = count_tokens(para)
            if current_tokens + para_tokens > max_tokens_per_chunk:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = para
                current_tokens = para_tokens
            else:
                current_chunk = f"{current_chunk}\n\n{para}" if current_chunk else para
                current_tokens += para_tokens
        if current_chunk:
            chunks.append(current_chunk)
    else:
        chunks.append(combined_text)

    # 6. For each chunk, call Claude (with citations enabled) and collect analysis.
    analysis_results = []
    chunk_details = []
    for idx, chunk in enumerate(chunks):
        chunk_id = str(uuid.uuid4())
        chunk_file_name = f"chunk_{chunk_id}.txt"
        chunk_file_path = os.path.join(RESULTS_FOLDER, chunk_file_name)
        with open(chunk_file_path, "w", encoding="utf-8") as f:
            f.write(chunk)

        chunk_details.append({"chunk_file": chunk_file_name, "tokens": count_tokens(chunk)})

        # Actually call Claude with the chunk as a custom content doc
        chunk_title = f"Passages Chunk {idx+1}"
        chunk_analysis = get_claude_analysis(query, chunk, chunk_title)
        analysis_results.append({
            "chunk_index": idx,
            "analysis": chunk_analysis
        })

    # 7. Combine analysis from all chunks
    combined_analysis = "\n\n".join([item["analysis"] for item in analysis_results])
    final_analysis = {
        "chunk_details": chunk_details,
        "analysis_per_chunk": analysis_results,
        "combined_analysis": combined_analysis
    }

    combined = {
        "results": results,
        "analysis": final_analysis
    }

    file_id = save_results_to_file(combined)
    return results, final_analysis, file_id

# -----------------------------
# Flask Routes
# -----------------------------
@app.route("/", methods=["GET"])
def home():
    return render_template("home.html")

@app.route("/api/search", methods=["POST"])
def handle_search():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400

        query = data.get("query")
        chumash = data.get("chumash")
        parsha = data.get("parsha")
        top_k = int(data.get("top_k", 10))

        results, analysis, file_id = search_and_analyze(query, chumash, parsha, top_k)
        return jsonify({
            "results": results,
            "analysis": analysis,
            "file_id": file_id
        })
    except Exception as e:
        logger.error(f"Search error: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@app.route("/download/<file_id>", methods=["GET"])
def download_file(file_id):
    file_path = os.path.join(RESULTS_FOLDER, f"{file_id}.json")
    if not os.path.exists(file_path):
        return jsonify({"error": "File not found"}), 404
    return send_file(file_path, as_attachment=True, download_name=f"results_{file_id}.json")

@app.route("/download/chunk/<filename>", methods=["GET"])
def download_chunk_file(filename):
    file_path = os.path.join(RESULTS_FOLDER, filename)
    if not os.path.exists(file_path):
        return jsonify({"error": "Chunk file not found"}), 404
    return send_file(file_path, as_attachment=True, download_name=filename)

if __name__ == "__main__":
    # If using .env, ensure load_dotenv() is called
    app.run(host="0.0.0.0", port=3000)