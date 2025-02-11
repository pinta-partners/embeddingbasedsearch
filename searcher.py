```python
# app.py
from flask import Flask, render_template, request, jsonify
import asyncio
import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
from scipy.spatial.distance import cosine
from tqdm import tqdm
from dotenv import load_dotenv
import anthropic
from transformers import AutoTokenizer, AutoModel  # Import here
import torch

# --- Configuration (Keep this consistent with indexer.py) ---
CONFIG = {
    "BASE_OUTPUT_DIR": "torah_search_data",
    "MAX_TOKENS": 512,  #  Keep for consistency
    "MIN_CHARS": 50,   #  Keep for consistency
    "MODEL_NAME": "dicta-il/BEREL_2.0", # Keep for consistency
}

app = Flask(__name__)

def get_output_dir(chumash: str, parsha: str) -> Path:
    """Gets the output directory for a specific parsha (copied from indexer.py)."""
    return Path(CONFIG["BASE_OUTPUT_DIR"]) / chumash / parsha

# --- Search Functions ---

def load_single_index(index_dir: Path) -> Tuple[List[Dict[str, Any]], np.ndarray, str, str]:
    """Loads metadata and embeddings for a single parsha."""
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

def load_search_data(chumash: Optional[str] = None, parsha: Optional[str] = None) -> List[Tuple[List[Dict[str, Any]], np.ndarray, str, str]]:
    """Loads saved paragraphs and embeddings based on filters."""
    print("Loading search data...")
    base_path = Path(CONFIG["BASE_OUTPUT_DIR"])
    if not base_path.exists():
        raise FileNotFoundError(f"Base directory not found: {CONFIG['BASE_OUTPUT_DIR']}")

    if chumash and parsha:
        index_dir = get_output_dir(chumash, parsha)
        return [load_single_index(index_dir)]
    elif chumash:
        chumash_dir = base_path / chumash.lower()
        if not chumash_dir.exists():
            raise FileNotFoundError(f"Chumash directory not found: {chumash}")
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
    """Finds top-k similar paragraphs using cosine similarity."""
    similarities = [(1.0 - cosine(query_emb, emb), i) for i, emb in enumerate(stored_embeds)]
    similarities.sort(reverse=True)
    top_k = min(top_k, len(similarities))
    return [(paragraphs[idx], sim, chumash, parsha) for sim, idx in similarities[:top_k]]

def prepare_search_results(matches: List[Tuple[Dict[str, Any], float, str, str]], query: str) -> Tuple[str, Dict[str, Any]]:
    """Prepares search results for both text file and Claude."""
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
        "title": "Search Results",
        "context": f"These are the most relevant paragraphs for the query: {query}",
        "citations": {"enabled": True}
    }
    return "\n".join(text_content), claude_doc

async def search_and_analyze(query: str, chumash: Optional[str] = None, parsha: Optional[str] = None, top_k: int = 20, skip_claude: bool = False) -> Tuple[str, Dict[str, Any]]:
    """Searches for similar paragraphs and optionally uses Claude for analysis."""
    try:
        sources_data = load_search_data(chumash, parsha)
        if not sources_data:
            raise ValueError("No search data found matching the specified criteria")

        # --- Embedding Model (ONLY for the query) ---
        print("Initializing embedding model for query...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = AutoTokenizer.from_pretrained(CONFIG["MODEL_NAME"])
        model = AutoModel.from_pretrained(CONFIG["MODEL_NAME"]).to(device)
        model.eval()

        def get_query_embedding(text: str) -> np.ndarray:
            """Generates embedding for the query text ONLY."""
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=CONFIG["MAX_TOKENS"])
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs)
            return outputs.last_hidden_state.mean(dim=1)[0].cpu().numpy()
        # --- End Embedding Model ---

        query_emb = get_query_embedding(query)  # Get embedding for the query
        all_matches = []

        for paragraphs, stored_embeds, src_chumash, src_parsha in tqdm(sources_data, desc="Processing sources"):
            matches = find_similar_paragraphs(query_emb, paragraphs, stored_embeds, src_chumash, src_parsha, top_k)
            all_matches.extend(matches)

        if not all_matches:
            print("\nNo matches found!")
            return ""

        text_content, claude_doc = prepare_search_results(all_matches, query)
        return text_content, claude_doc

    except Exception as e:
        print(f"\nError during search: {e}")
        raise

@app.route("/", methods=["GET"])
def home():
    return render_template("home.html")

@app.route("/search", methods=["POST"])
async def handle_search():
    data = request.get_json()
    query = data.get("query")
    chumash = data.get("chumash")
    parsha = data.get("parsha")
    top_k = int(data.get("top_k", 20))
    
    try:
        result = await asyncio.run(search_and_analyze(query, chumash, parsha, top_k, skip_claude=True))
        return jsonify({
            "results": result[0],
            "analysis": result[1]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    load_dotenv()
    app.run(debug=True)
```

```html
<!-- templates/home.html -->
<!DOCTYPE html>
<html>
<head>
    <title>Torah Search</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
        }
        form {
            display: flex;
            flex-direction: column;
            gap: 10px;
        }
        label {
            font-weight: bold;
        }
        input, select {
            padding: 5px;
        }
        button {
            padding: 5px 10px;
            background-color: #4CAF50;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
        }
        button:hover {
            background-color: #45a049;
        }
        #results {
            margin-top: 20px;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 4px;
        }
        pre {
            white-space: pre-wrap;
            word-wrap: break-word;
        }
    </style>
</head>
<body>
    <h1>Torah Search Engine</h1>
    <form id="searchForm">
        <label for="query">Query:</label>
        <input type="text" id="query" name="query" required>
        
        <label for="chumash">Chumash:</label>
        <select id="chumash" name="chumash">
            <option value="">All</option>
            <option value="Bereishis">Bereishis</option>
            <option value="Shemos">Shemos</option>
            <option value="Vayikra">Vayikra</option>
            <option value="Bamidbar">Bamidbar</option>
            <option value="Devarim">Devarim</option>
        </select>
        
        <label for="parsha">Parsha:</label>
        <select id="parsha" name="parsha">
            <option value="">All</option>
        </select>
        
        <label for="top_k">Top Matches:</label>
        <input type="number" id="top_k" name="top_k" value="20">
        
        <button type="submit">Search</button>
    </form>

    <div id="results"></div>

    <script>
        document.getElementById('searchForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            const formData = new FormData(e.target);
            const data = Object.fromEntries(formData.entries());
            
            const response = await fetch('/search', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(data),
            });
            
            const result = await response.json();
            
            if (result.error) {
                alert(result.error);
            } else {
                const resultsDiv = document.getElementById('results');
                resultsDiv.innerHTML = `
                    <h2>Results</h2>
                    <pre>${result.results}</pre>
                    <h2>Analysis</h2>
                    <pre>${result.analysis}</pre>
                `;
            }
        });
    </script>
</body>
</html>
```

To run the application:

1. Install the required packages:
   ```bash
   pip install flask transformers torch scipy tqdm python-dotenv anthropic
   ```

2. Create the templates directory and add the `home.html` file.

3. Run the Flask app:
   ```bash
   python app.py
   ```
