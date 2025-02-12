
from flask import Flask, render_template, request, jsonify
import asyncio
import argparse
import json
import os
import sys
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

@app.route("/search", methods=["POST"])
async def handle_search():
    data = request.get_json()
    query = data.get("query")
    chumash = data.get("chumash")
    parsha = data.get("parsha")
    top_k = int(data.get("top_k", 20))
    
    try:
        from searcher import search_and_analyze
        result = await search_and_analyze(query, chumash, parsha, top_k, skip_claude=True)
        return jsonify({
            "results": result[0],
            "analysis": result[1]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    load_dotenv()
    app.run(host='0.0.0.0', port=3000)
