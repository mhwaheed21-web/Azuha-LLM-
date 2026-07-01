# finetune_generate.py
# Chapter 7 — generate responses from your fine-tuned model
#
# What this file does:
#   Loads the saved fine-tuned model (gpt2-medium355M-sft.pth)
#   and generates responses to instructions you give it.
#   Run this AFTER finetune_train.py has completed and saved the model.
#
# Imports from YOUR existing files:
#   gpt.py          → GPTModel
#   gpt_generate.py → generate, text_to_token_ids, token_ids_to_text
#
# Imports from NEW files:
#   instruction_dataset.py → format_input

import sys
import torch
import tiktoken
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from llm_architecture_step3.gpt import GPTModel
from pretrain_step4.gpt_generate import generate, text_to_token_ids, token_ids_to_text
from finetune_step4.instruction_dataset import format_input


# ─────────────────────────────────────────────
# MODEL CONFIG — must be identical to finetune_train.py
# ─────────────────────────────────────────────

BASE_CONFIG = {
    "vocab_size": 50257,
    "context_length": 1024,
    "drop_rate": 0.0,
    "qkv_bias": True
}

MODEL_CONFIGS = {
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
}

CHOOSE_MODEL = "gpt2-medium (355M)"


# ─────────────────────────────────────────────
# LOAD FINE-TUNED MODEL FROM DISK
# ─────────────────────────────────────────────

def load_finetuned_model(model_path, device):
    """
    Loads the fine-tuned model weights from the .pth file
    saved by finetune_train.py.
    """
    cfg = {**BASE_CONFIG, **MODEL_CONFIGS[CHOOSE_MODEL]}
    model = GPTModel(cfg)
    model.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=True)
    )
    model.to(device)
    model.eval()
    print(f"Loaded fine-tuned model from {model_path}")
    return model


# ─────────────────────────────────────────────
# GENERATE A RESPONSE
# ─────────────────────────────────────────────

def generate_response(model, tokenizer, device,
                      instruction, input_text="",
                      max_new_tokens=100,
                      temperature=0.0,
                      top_k=None):
    """
    Takes a plain instruction string, formats it into Alpaca prompt style,
    generates a response, and returns only the response text (not the prompt).

    Args:
        instruction   : the instruction / question string
        input_text    : optional extra input (can be empty string)
        max_new_tokens: how many tokens to generate
        temperature   : 0.0 = greedy (deterministic),
                        >0  = random sampling (more creative)
                        I am not certain of the ideal value — 0.7 or 1.0
                        are common starting points, verify from your book.
        top_k         : if set, only sample from top k tokens.
                        50 is a common value but verify.
    """
    entry = {"instruction": instruction, "input": input_text}
    prompt = format_input(entry)

    token_ids = generate(
        model=model,
        idx=text_to_token_ids(prompt, tokenizer).to(device),
        max_new_tokens=max_new_tokens,
        context_size=BASE_CONFIG["context_length"],
        temperature=temperature,
        top_k=top_k,
        eos_id=50256   # stop at end-of-text token
    )

    full_text    = token_ids_to_text(token_ids, tokenizer)
    response     = full_text[len(prompt):].strip()
    return response


# ─────────────────────────────────────────────
# BATCH EVALUATE ON TEST SET
# prints model responses vs expected responses
# so you can judge quality
# ─────────────────────────────────────────────

def evaluate_on_test_set(model, tokenizer, device, test_data, num_samples=10):
    """
    Generates responses for the first num_samples entries of test_data
    and prints them alongside the expected output.
    Used for qualitative evaluation (Chapter 7 stage 3).
    """
    print("\n" + "="*60)
    print("QUALITATIVE EVALUATION ON TEST SET")
    print("="*60)

    for i, entry in enumerate(test_data[:num_samples]):
        prompt   = format_input(entry)
        expected = entry["output"]
        response = generate_response(
            model, tokenizer, device,
            instruction=entry["instruction"],
            input_text=entry["input"],
            max_new_tokens=100,
            temperature=0.0
        )

        print(f"\n--- Example {i+1} ---")
        print(f"Instruction : {entry['instruction']}")
        if entry["input"]:
            print(f"Input       : {entry['input']}")
        print(f"Expected    : {expected}")
        print(f"Generated   : {response}")
        print()


# ─────────────────────────────────────────────
# MAIN — run this file to chat with your model
# ─────────────────────────────────────────────

if __name__ == "__main__":
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = tiktoken.get_encoding("gpt2")

    # path to the model saved by finetune_train.py
    model_path = "gpt2-medium355M-sft.pth"
    model = load_finetuned_model(model_path, device)

    # ── interactive mode ──
    # type your own instructions and see the model respond
    print("\n" + "="*60)
    print("FINE-TUNED ASSISTANT — type 'quit' to exit")
    print("="*60)

    while True:
        instruction = input("\nInstruction: ").strip()
        if instruction.lower() in ("quit", "exit", "q"):
            break
        if not instruction:
            continue

        extra_input = input("Input (press Enter to skip): ").strip()

        response = generate_response(
            model=model,
            tokenizer=tokenizer,
            device=device,
            instruction=instruction,
            input_text=extra_input,
            max_new_tokens=150,
            temperature=0.0,   # change to 0.7 for more varied responses
            top_k=50
        )
        print(f"\nResponse: {response}")
