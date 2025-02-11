from flask import Flask, render_template, request, jsonify
import json
import os
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
from scipy.spatial.distance import cosine
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModel
import torch
import anthropic

# Configuration remains the same
CONFIG = {
    "base_output_dir": "torah_search_data",
    "max_tokens": 512,
    "min_chars": 50,
    "model_name": "dicta-il/BEREL_2.0",
}

BOOK_DIR_MAPPING = {
    "bamidbar": "Bamidbar",
    "bereishis": "Bereshit",
    "devarim": "Devarim",
    "shemos": "Shemot",
    "vayikra": "Vayikra"
}

app = Flask(__name__)

# Helper functions remain the same
def get_output_dir(chumash: str, parsha: str) -> Path:
    mapped_chumash = BOOK_DIR_MAPPING.get(chumash.lower(), chumash)
    return Path(CONFIG["base_output_dir"]) / mapped_chumash / parsha

def load_search_data(chumash: Optional[str] = None, parsha: Optional[str] = None) -> List[Tuple[List[Dict[str, Any]], np.ndarray, str, str]]:
    print("Loading search data...")
    base_path = Path(CONFIG["base_output_dir"])
    if not base_path.exists():
        raise FileNotFoundError(f"Base directory not found: {base_path}")

    if chumash and parsha:
        index_dir = get_output_dir(chumash, parsha)
        return [load_single_index(index_dir)]
    elif chumash:
        mapped_chumash = BOOK_DIR_MAPPING.get(chumash.lower(), chumash)
        chumash_dir = base_path / mapped_chumash
        if not chumash_dir.exists():
            raise FileNotFoundError(f"Chumash directory not found: {mapped_chumash}")
        return [load_single_index(d) for d in chumash_dir.iterdir() if d.is_dir()]
    else:
        all_data = []
        for chumash_dir in base_path.iterdir():
            if chumash_dir.is_dir():
                for parsha_dir in chumash_dir.iterdir():
                    if parsha_dir.is_dir():
                        try:
                            all_data.append(load_single_index(parsha_dir))
                        except Exception as e:
                            print(f"Error loading {parsha_dir}: {e}")
        return all_data

def find_similar_paragraphs(query_emb: np.ndarray, paragraphs: List[Dict[str, Any]], stored_embeds: np.ndarray, chumash: str, parsha: str, top_k: int) -> List[Tuple[Dict[str, Any], float, str, str]]:
    similarities = [(1.0 - cosine(query_emb, emb), i) for i, emb in enumerate(stored_embeds)]
    similarities.sort(reverse=True)
    top_k = min(top_k, len(similarities))
    return [(paragraphs[idx], sim, chumash, parsha) for sim, idx in similarities[:top_k]]

def prepare_search_results(matches: List[Tuple[Dict[str, Any], float, str, str]], query: str) -> Tuple[str, Dict[str, Any]]:
    sorted_matches = sorted(matches, key=lambda x: x[1], reverse=True)
    text_content = [f"Query: {query}\n"]
    claude_blocks = []
    current_chumash = None
    current_parsha = None

    for para, similarity, chumash, parsha in sorted_matches:
        if chumash != current_chumash:
            text_content.append(f"\n{'='*20} Results from {chumash} {'='*20}\n")
            claude_blocks.append({"type": "text", "text": f"\n=== Results from {chumash} ===\n"})
            current_chumash = chumash
            current_parsha = None

        if parsha != current_parsha:
            text_content.append(f"\n{'-'*10} {parsha} {'-'*10}\n")
            claude_blocks.append({"type": "text", "text": f"\n--- {parsha} ---\n"})
            current_parsha = parsha

        match_text = (
            f"\nSimilarity: {similarity:.4f}\n"
            f"Source: {para['source_file']}\n"
            f"Text: {para['text']}\n"
            f"{'-'*80}\n"
        )
        text_content.append(match_text)
        claude_blocks.append({
            "type": "text",
            "text": f"{para['text']} [Source: {chumash}/{para['source_file']}, Similarity: {similarity:.4f}]"
        })

    claude_doc = {
        "type": "document",
        "source": {"type": "content", "content": claude_blocks},
        "title": "Search results",
        "context": f"These are the most relevant paragraphs for the query: {query}",
        "citations": {"enabled": True}
    }
    return "\n".join(text_content), claude_doc

def load_single_index(index_dir: Path) -> Tuple[List[Dict[str, Any]], np.ndarray, str, str]:
    metadata_file = index_dir / "metadata.json"
    embeddings_file = index_dir / "embeddings.npy"

    if not metadata_file.exists() or not embeddings_file.exists():
        raise FileNotFoundError(f"Index files not found in {index_dir}")

    with open(metadata_file, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    embeddings = np.load(embeddings_file)
    chumash = metadata["paragraphs"][0]["chumash"]
    parsha = metadata["paragraphs"][0]["parsha"]
    return metadata["paragraphs"], embeddings, chumash, parsha

# Modified to be synchronous
def get_claude_analysis(query: str, claude_doc: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get analysis from Claude API using the search results.
    """
    try:
        client = anthropic.Anthropic()
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": [
                        claude_doc,
                        {
                            "type": "text",
                            "text": f"Please analyze these Torah passages in relation to the query: {query}. "
                                   "Provide insights about the connections between the passages and explain their relevance "
                                   "to the query. Use citations to support your analysis."
                        }
                    ]
                }
            ]
        )
        # Extract the text content from the response
        if hasattr(response.content[0], 'text'):
            return response.content[0].text
        return str(response.content)
    except Exception as e:
        print(f"Error calling Claude API: {e}")
        return {"error": str(e)}

# Modified to be synchronous
def search_and_analyze(query: str, chumash: Optional[str] = None, parsha: Optional[str] = None, top_k: int = 20, skip_claude: bool = False) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
    try:
        sources_data = load_search_data(chumash, parsha)
        if not sources_data:
            raise ValueError("No search data found matching the specified criteria")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_name"])
        model = AutoModel.from_pretrained(CONFIG["model_name"]).to(device)
        model.eval()

        def get_query_embedding(text: str) -> np.ndarray:
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=CONFIG["max_tokens"])
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs)
            return outputs.last_hidden_state.mean(dim=1)[0].cpu().numpy()

        query_emb = get_query_embedding(query)
        all_matches = []

        for paragraphs, stored_embeds, src_chumash, src_parsha in sources_data:
            matches = find_similar_paragraphs(query_emb, paragraphs, stored_embeds, src_chumash, src_parsha, top_k)
            all_matches.extend(matches)

        if not all_matches:
            print("\nNo matches found!")
            return "", {}, {}

        text_content, claude_doc = prepare_search_results(all_matches, query)

        # Get Claude's analysis if not skipped
        claude_analysis = {}
        if not skip_claude:
            claude_analysis = get_claude_analysis(query, claude_doc)

        return text_content, claude_doc, claude_analysis

    except Exception as e:
        print(f"\nError during search: {e}")
        raise

@app.route("/", methods=["GET"])
def home():
    return render_template("home.html")

# Modified to be synchronous
@app.route("/api/search", methods=["POST"])
def handle_search():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data received"}), 400

        query = data.get("query")
        if not query:
            return jsonify({"error": "No query provided"}), 400

        chumash = data.get("chumash") or data.get("book")
        parsha = data.get("parsha")
        top_k = int(data.get("topK", 20))
        skip_claude = data.get("skipClaude", False)

        text_content, claude_doc, claude_analysis = search_and_analyze(
            query, chumash, parsha, top_k, skip_claude
        )

        return jsonify({
            "results": text_content,
            "document": claude_doc,
            "analysis": claude_analysis
        })
    except Exception as e:
        print(f"Error in handle_search: {str(e)}")
        return jsonify({"error": f"Search request failed: {str(e)}"}), 500

if __name__ == "__main__":
    load_dotenv()
    app.run(host='0.0.0.0', port=3000, debug=True)