# finetune_generate.py
# Chapter 7 — generate responses from your fine-tuned model
#
# What this file does:
#   Loads the saved fine-tuned model (gpt2-small124M-lora-r4-sft.pth)
#   and generates responses to instructions you give it.
#   Run this AFTER finetune_train.py has completed and saved the model.
#
# Imports from YOUR existing files:
#   gpt.py              → GPTModel
#   gpt_generate.py     → generate, text_to_token_ids, token_ids_to_text
#   instruction_dataset.py → format_input
#   lora.py             → apply_lora

import sys
import torch
import tiktoken
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from llm_architecture_step3.gpt import GPTModel
from pretraining_step4.gpt_generate import generate, text_to_token_ids, token_ids_to_text
from fine_tunning_step5.instruction_dataset import format_input
from fine_tunning_step5.lora import apply_lora


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
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
}

CHOOSE_MODEL = "gpt2-small (124M)"

# must match the rank used during training
LORA_RANK = 4


# ─────────────────────────────────────────────
# LOAD FINE-TUNED MODEL FROM DISK
# ─────────────────────────────────────────────

def load_finetuned_model(model_path, device):
    """
    Loads the LoRA fine-tuned model weights from the .pth file
    saved by finetune_train.py.

    IMPORTANT: apply_lora() must be called before load_state_dict()
    because the saved .pth file contains LoRA weights (lora.A and lora.B
    inside LinearWithLoRA wrappers). Without apply_lora() first, the
    model structure will not match the saved file and it will crash.
    """
    cfg = {**BASE_CONFIG, **MODEL_CONFIGS[CHOOSE_MODEL]}
    model = GPTModel(cfg)

    # inject LoRA structure first so model matches the saved file
    apply_lora(model, rank=LORA_RANK)

    # now load the saved weights into the LoRA-injected model
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
                        0.7 or 1.0 are common starting points.
        top_k         : if set, only sample from top k tokens.
                        50 is a common value.
    """
    entry  = {"instruction": instruction, "input": input_text}
    prompt = format_input(entry)

    token_ids = generate(
        model=model,
        idx=text_to_token_ids(prompt, tokenizer).to(device),
        max_new_tokens=max_new_tokens,
        context_size=BASE_CONFIG["context_length"],
        temperature=temperature,
        top_k=top_k,
        eos_id=50256
    )

    full_text = token_ids_to_text(token_ids, tokenizer)
    response  = full_text[len(prompt):].strip()
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
    # uses path relative to this file so it always works
    # regardless of which directory you run from
    model_path = Path(__file__).resolve().parent / "gpt2-small124M-lora-r4-sft.pth"

    model = load_finetuned_model(str(model_path), device)

    # ── interactive mode ──
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
            temperature=0.0,
            top_k=50
        )
        print(f"\nResponse: {response}")