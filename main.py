from flask import Flask, render_template, request, jsonify
import asyncio
import json
import os
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
from scipy.spatial.distance import cosine
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModel
import torch

# --- configuration (Keep this consistent with indexer.py) ---
CONFIG = {
    "base_output_dir": "torah_search_data",
    "max_tokens": 512,
    "min_chars": 50,
    "model_name": "dicta-il/BEREL_2.0",
}

# Full mapping for the Torah books based on your folder hierarchy.
# The keys are the values coming from the HTML form (all lowercase),
# and the values are the actual directory names.
BOOK_DIR_MAPPING = {
    "bamidbar": "Bamidbar",
    "bereishis": "Bereshit",
    "devarim": "Devarim",
    "shemos": "Shemot",
    "vayikra": "Vayikra"
}

app = Flask(__name__)

def get_output_dir(chumash: str, parsha: str) -> Path:
    # Convert the provided book (chumash) to the actual directory name using the mapping.
    mapped_chumash = BOOK_DIR_MAPPING.get(chumash.lower(), chumash)
    return Path(CONFIG["base_output_dir"]) / mapped_chumash / parsha

# --- Search Functions ---

def load_single_index(index_dir: Path) -> Tuple[List[Dict[str, Any]], np.ndarray, str, str]:
    metadata_file = index_dir / "metadata.json"
    embeddings_file = index_dir / "embeddings.npy"

    if not metadata_file.exists() or not embeddings_file.exists():
        raise FileNotFoundError(f"Index files not found in {index_dir}")

    with open(metadata_file, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    embeddings = np.load(embeddings_file)
    # Expect the metadata to store the actual book and parsha names.
    chumash = metadata["paragraphs"][0]["chumash"]
    parsha = metadata["paragraphs"][0]["parsha"]
    return metadata["paragraphs"], embeddings, chumash, parsha

def load_search_data(chumash: Optional[str] = None, parsha: Optional[str] = None) -> List[Tuple[List[Dict[str, Any]], np.ndarray, str, str]]:
    print("Loading search data...")
    base_path = Path(CONFIG["base_output_dir"])
    if not base_path.exists():
        raise FileNotFoundError(f"Base directory not found: {base_path}")

    if chumash and parsha:
        index_dir = get_output_dir(chumash, parsha)
        return [load_single_index(index_dir)]
    elif chumash:
        # Map the provided book to the correct directory.
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

async def search_and_analyze(query: str, chumash: Optional[str] = None, parsha: Optional[str] = None, top_k: int = 20, skip_claude: bool = False) -> Tuple[str, Dict[str, Any]]:
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
            return "", {}

        text_content, claude_doc = prepare_search_results(all_matches, query)
        return text_content, claude_doc

    except Exception as e:
        print(f"\nError during search: {e}")
        raise

@app.route("/", methods=["GET"])
def home():
    return render_template("home.html")

@app.route("/api/search", methods=["POST"])
async def handle_search():
    data = request.get_json()
    query = data.get("query")
    # Use 'book' (or 'chumash') from the form submission.
    chumash = data.get("chumash") or data.get("book")
    parsha = data.get("parsha")
    top_k = int(data.get("topK", 20))
    
    try:
        result = await search_and_analyze(query, chumash, parsha, top_k, skip_claude=True)
        return jsonify({
            "results": result[0],
            "analysis": result[1]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    load_dotenv()
    app.run(debug=True)
