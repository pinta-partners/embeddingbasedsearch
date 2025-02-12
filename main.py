from flask import Flask, render_template, request, jsonify
import asyncio
import os
from pathlib import Path
import numpy as np
from scipy.spatial.distance import cosine
from tqdm import tqdm
from dotenv import load_dotenv
import torch
from transformers import AutoTokenizer, AutoModel

# Configuration
CONFIG = {
    "BASE_OUTPUT_DIR": "torah_search_data",
    "MAX_TOKENS": 512,
    "MIN_CHARS": 50,
    "MODEL_NAME": "dicta-il/BEREL_2.0",
}

app = Flask(__name__)

def get_output_dir(chumash: str, parsha: str) -> Path:
    return Path(CONFIG["BASE_OUTPUT_DIR"]) / chumash / parsha

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
        top_k = int(data.get("top_k", 20))

        result = search_and_analyze(query, chumash, parsha, top_k, skip_claude=True)
        return jsonify({
            "results": result[0],
            "analysis": result[1]
        })
    except Exception as e:
        print(f"Search error: {str(e)}")
        return jsonify({"error": str(e)}), 500

def search_and_analyze(query, chumash, parsha, top_k, skip_claude=True):
    # Placeholder for actual search and analysis logic.  This needs to be implemented.
    #  This function should return a tuple: (results, analysis)
    results = [{"text": f"Result {i} for query: {query}", "score": i} for i in range(top_k)]
    analysis = {"query": query, "chumash": chumash, "parsha": parsha}
    return results, analysis


if __name__ == "__main__":
    load_dotenv()
    app.run(host='0.0.0.0', port=3000)