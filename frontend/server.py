# server.py
# Flask backend that connects index.html to the fine-tuned Azuha model.
#
# Folder: frontend/server.py
#
# ── Install ──
#   pip install flask flask-cors
#
# ── Run ──
#   cd /path/to/Azuha-LLM
#   python frontend/server.py
#
# ── Then ──
#   Open http://localhost:5000 in your browser
#   OR open index.html and change:
#       const API_URL   = 'http://localhost:5000/api/chat';
#       const DEMO_MODE = false;
#
# ── What this file does ──
#   1. Loads the fine-tuned Azuha model once at startup
#   2. Serves index.html at GET /
#   3. Accepts POST /api/chat  →  { "message": "user text" }
#   4. Returns              →  { "response": "azuha reply" }
#
# ── Model path ──
#   Update MODEL_PATH below to point to your saved .pth file.
#   Default assumes you run this from the project root.

import sys
import torch
import tiktoken
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# ─── PROJECT ROOT ─────────────────────────────────────────────────────
# This file lives in frontend/ — one level down from project root.
# parents[1] goes up to Azuha-LLM/
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from llm_architecture_step3.gpt import GPTModel
from pretraining_step4.gpt_generate import generate, text_to_token_ids, token_ids_to_text
from fine_tunning_step5.instruction_dataset import format_input
from fine_tunning_step5.lora import apply_lora

# ─── CONFIG — change these if needed ─────────────────────────────────

# Path to your saved model — update this to match your actual filename
MODEL_PATH = project_root / "fine_tunning_step5" / "gpt2-small124M-lora-r4-sft.pth"

# LoRA rank — must match what was used in finetune_train.py
LORA_RANK = 4

# Generation settings
MAX_NEW_TOKENS = 150    # max tokens Azuha generates per response
TEMPERATURE    = 0.7    # 0.0 = deterministic, 0.7 = natural — do not use 0.0, causes loops
TOP_K          = 40     # sample from top-k tokens only

# Server settings
HOST = "0.0.0.0"        # 0.0.0.0 = accessible from your network
PORT = 5000

# Model architecture — must match finetune_train.py exactly
BASE_CONFIG = {
    "vocab_size":      50257,
    "context_length":  1024,
    "drop_rate":       0.0,
    "qkv_bias":        True
}
MODEL_CONFIG = {
    "emb_dim":   768,
    "n_layers":  12,
    "n_heads":   12
}


# ─── LOAD MODEL ONCE AT STARTUP ───────────────────────────────────────

device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = tiktoken.get_encoding("gpt2")

print(f"\n{'─'*50}")
print(f"  Azuha server starting")
print(f"{'─'*50}")
print(f"  Device    : {device}")
print(f"  Model     : {MODEL_PATH.name}")
print(f"  LoRA rank : {LORA_RANK}")

if not MODEL_PATH.exists():
    print(f"\n  ERROR: model file not found at {MODEL_PATH}")
    print(f"  Run finetune_train.py first to generate the model file.")
    sys.exit(1)

cfg   = {**BASE_CONFIG, **MODEL_CONFIG}
model = GPTModel(cfg)
apply_lora(model, rank=LORA_RANK)     # inject LoRA structure before loading weights
model.load_state_dict(
    torch.load(str(MODEL_PATH), map_location=device, weights_only=True)
)
model.to(device)
model.eval()

print(f"  Status    : ready")
print(f"{'─'*50}\n")


# ─── FLASK APP ────────────────────────────────────────────────────────

# static_folder points to the frontend/ directory
# so Flask can serve index.html at GET /
FRONTEND_DIR = Path(__file__).resolve().parent

app = Flask(__name__, static_folder=str(FRONTEND_DIR))
CORS(app)   # allow the frontend to call this from any origin


# ── serve index.html ──────────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


# ── health check ──────────────────────────────────────────────────────
@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "model":  MODEL_PATH.name,
        "device": str(device),
        "lora_rank": LORA_RANK
    })


# ── main chat endpoint ─────────────────────────────────────────────────
# Matches the frontend's API_URL = '/api/chat'
# Expects: POST  { "message": "user text here" }
# Returns:       { "response": "azuha reply here" }
@app.route("/api/chat", methods=["POST"])
def chat():
    # validate request
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "request body must be JSON"}), 400
    if "message" not in data:
        return jsonify({"error": "missing 'message' field"}), 400

    user_message = data["message"].strip()
    if not user_message:
        return jsonify({"error": "message cannot be empty"}), 400

    if len(user_message) > 2000:
        return jsonify({"error": "message too long — keep under 2000 characters"}), 400

    # format prompt in User/Azuha style — matches training format exactly
    # format_input returns: "User: <message>\nAzuha:"
    prompt = format_input(user_message)

    # generate response
    try:
        with torch.no_grad():
            token_ids = generate(
                model=model,
                idx=text_to_token_ids(prompt, tokenizer).to(device),
                max_new_tokens=MAX_NEW_TOKENS,
                context_size=BASE_CONFIG["context_length"],
                temperature=TEMPERATURE,
                top_k=TOP_K,
                eos_id=50256    # stop at end-of-text token
            )

        # decode and strip the prompt prefix — return only Azuha's reply
        full_text = token_ids_to_text(token_ids, tokenizer)
        response  = full_text[len(prompt):].strip()

        # guard against empty response
        if not response:
            response = "I did not generate a response. Try rephrasing your question."

        return jsonify({"response": response})

    except Exception as e:
        print(f"Generation error: {e}")
        return jsonify({"error": "generation failed", "detail": str(e)}), 500


# ─── RUN ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"  Open http://localhost:{PORT} in your browser")
    print(f"  Or set API_URL = 'http://localhost:{PORT}/api/chat' in index.html\n")
    app.run(host=HOST, port=PORT, debug=False)
