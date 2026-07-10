# Azuha — Personal AI Assistant

> I built a GPT-style large language model completely from scratch using PyTorch and fine-tuned it with LoRA to serve as my own personal AI assistant. Every component — tokenization, multi-head attention, transformer blocks, pretraining, and fine-tuning — is written by me, not wrapped around an existing API.

---

## Table of Contents

- [What is this](#what-is-this)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Model Specifications](#model-specifications)
- [Training Data](#training-data)
- [LoRA Implementation](#lora-implementation)
- [Getting Started](#getting-started)
- [Configurable Values](#configurable-values)
- [Training Pipeline](#training-pipeline)
- [Monitoring Training](#monitoring-training)
- [Keeping Azuha Updated](#keeping-azuha-updated)
- [Continued Fine-Tuning](#continued-fine-tuning)
- [Chat Format](#chat-format)
- [References](#references)

---

## What is this

I followed Sebastian Raschka's book *Build a Large Language Model From Scratch*, implementing every piece in PyTorch before moving to the next. Once the architecture was complete I replaced the book's Alpaca-style fine-tuning with a conversational `User:/Azuha:` format, combined with LoRA fine-tuning faithful to the original Hu et al. 2021 paper.

The result is a working personal assistant that:

- Knows who I am — my background, projects, skills, goals, and personality
- Responds in a natural conversational format rather than formal instruction-response style
- Was fine-tuned by training only 147,456 parameters out of 163 million — 0.09% — using LoRA

This is not a demo or tutorial reproduction. The fine-tuning pipeline, LoRA implementation, personal dataset generation, and conversational format are decisions I made independently on top of the book's foundation.

---

## Architecture

```
                    ┌──────────────────────────────────┐
                    │         User Input (text)         │
                    └─────────────────┬────────────────┘
                                      │
                    ┌─────────────────▼────────────────┐
                    │     tiktoken GPT-2 Tokenizer      │
                    │   text → token IDs (vocab 50257)  │
                    └─────────────────┬────────────────┘
                                      │
                    ┌─────────────────▼────────────────┐
                    │          GPTModel (gpt.py)        │
                    │                                   │
                    │  tok_emb  [vocab_size, emb_dim]   │
                    │  pos_emb  [context_len, emb_dim]  │
                    │        ↓ (sum + dropout)          │
                    │  ┌──────────────────────────┐    │
                    │  │   TransformerBlock × 12   │    │
                    │  │                          │    │
                    │  │  LayerNorm               │    │
                    │  │  MultiHeadAttention ←────┼────┼── LoRA injected here
                    │  │    W_query (frozen)       │    │   (W_query, W_value)
                    │  │    W_key   (frozen)       │    │
                    │  │    W_value (frozen) ←─────┼────┼── LoRA injected here
                    │  │    out_proj (frozen)      │    │
                    │  │  Residual connection      │    │
                    │  │  LayerNorm               │    │
                    │  │  FeedForward (frozen)    │    │
                    │  │  Residual connection      │    │
                    │  └──────────────────────────┘    │
                    │        ↓                         │
                    │  Final LayerNorm                 │
                    │  out_head Linear [emb_dim→50257] │
                    └─────────────────┬────────────────┘
                                      │ logits [batch, seq, 50257]
                    ┌─────────────────▼────────────────┐
                    │    Token Sampling / Generation    │
                    │   greedy (temp=0) or top-k + temp │
                    └─────────────────┬────────────────┘
                                      │
                    ┌─────────────────▼────────────────┐
                    │     tiktoken decode → text        │
                    └──────────────────────────────────┘
```

### LoRA injection detail

```
Original W_query forward:
    output = x @ W_query.T             ← frozen, not updated

LinearWithLoRA forward:
    output = x @ W_query.T             ← frozen
           + (x @ A.T @ B.T) * (1/r)  ← trainable: A and B only

A: shape [rank, in_features]  — Gaussian init
B: shape [out_features, rank] — zero init (LoRA output = 0 at start)
```

Applied to `W_query` and `W_value` in every TransformerBlock. `W_key`, `out_proj`, and all FeedForward layers remain frozen.

---

## Project Structure

```
Azuha-LLM/
│
├── data_preperation_step1/
│   ├── dataloader.py               # BPE tokenization, sliding window, DataLoader
│   └── the-verdict.txt             # Small text used for pretraining experiments
│
├── attention_mechanism_step2/
│   └── multihead_attention.py      # multi-head self-attention, causal mask
│
├── llm_architecture_step3/
│   └── gpt.py                      #GPTModel, TransformerBlock, generate_text_simple
│
├── pretraining_step4/
│   ├── gpt_download.py             # Downloads OpenAI GPT-2 weights (124M / 355M / 774M / 1558M)
│   ├── gpt_generate.py             # Loads weights, temperature sampling, top-k, load_weights_into_gpt
│   └── gpt_train.py                # Pretraining loop on raw text, saves model.pth
│
├── fine_tunning_step5/
│   ├── profile.json                # Private (git-ignored)
│   ├── generate_dataset.py         # Private (git-ignored)
│   ├── instruction_dataset.py      # Loads and merges all three datasets, AzuhaDataset class
│   ├── lora.py                     # LoRALayer, LinearWithLoRA, apply_lora, count_parameters
│   ├── finetune_train.py           # Fine-tuning training loop with LoRA support
│   ├── finetune_generate.py        # Interactive chat with the fine-tuned model
│   └── hamza_azuha_dataset.json    # Auto-generated personal dataset (git-ignored)
│
|──requirement.txt
└── README.md
```

---

## Model Specifications

| Component | Value |
|---|---|
| Base model | GPT-2 small |
| Total parameters | 163,184,640 |
| Pretrained weights | OpenAI GPT-2 small (downloaded via `gpt_download.py`) |
| Embedding dimension | 768 |
| Attention heads | 12 |
| Transformer layers | 12 |
| Context length | 1,024 tokens |
| Vocabulary size | 50,257 — GPT-2 BPE |
| Fine-tuning method | LoRA — Hu et al. 2021 |
| LoRA rank | 4 |
| LoRA scaling | 1/r — original paper formulation |
| LoRA target layers | W_query and W_value (paper Section 5.1) |
| LoRA dropout | None — per paper |
| Trainable parameters after LoRA | 147,456 — 0.09% of total |
| Training format | `User: ...\nAzuha: ...` |

---

## Training Data

Three sources are merged before training:

| Dataset | Size | License | Purpose |
|---|---|---|---|
| `HuggingFaceTB/everyday-conversations-llama3.1-2k` | ~2,260 pairs | Apache 2.0 | Natural everyday conversation behavior — greetings, casual exchanges, common questions |
| `HuggingFaceH4/ultrachat_200k` | 20,000 sampled | MIT | General conversational intelligence and instruction following |
| `hamza_azuha_dataset.json` | ~110 entries × 50 repeats | Personal | Azuha's identity, my personal background, projects, and skills (Increased and fine_tunned in furure) |

The personal dataset is repeated 50 times so the model learns personal facts reliably despite being a small fraction of the total data. Both external datasets are downloaded from HuggingFace and cached locally on first run.

All three sources are extracted into a flat list of `{"user": ..., "assistant": ...}` pairs and formatted consistently as:

```
User: <user message>
Azuha: <assistant message>
```

---

## LoRA Implementation

My LoRA implementation in `lora.py` follows the original paper exactly based on my reading of Hu et al. 2021. Every design decision below is sourced from the paper.

### LoRALayer

```python
class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, rank):
        self.A = nn.Parameter(torch.randn(rank, in_features))  # Gaussian init — Section 3
        self.B = nn.Parameter(torch.zeros(out_features, rank)) # Zero init   — Section 3

    def forward(self, x):
        return (x @ self.A.T @ self.B.T) * (1.0 / self.rank)  # 1/r scaling — Section 3
```

### LinearWithLoRA

```python
class LinearWithLoRA(nn.Module):
    def forward(self, x):
        return self.linear(x) + self.lora(x)   # frozen + trainable
```

### apply_lora

```python
def apply_lora(model, rank):
    for param in model.parameters():       # freeze everything — Chapter 6 approach
        param.requires_grad = False

    for block in model.trf_blocks:         # inject into W_query and W_value
        att = block.att                    # only these two — paper Section 5.1
        att.W_query = LinearWithLoRA(att.W_query, rank=rank)
        att.W_value = LinearWithLoRA(att.W_value, rank=rank)
```

After injection with rank=4 on GPT-2 small:

```
Trainable parameters :    147,456
Total parameters     : 163,184,640
Trainable %          :     0.0904%
```

---

## Getting Started

### Requirements

```bash
pip install torch tiktoken matplotlib datasets
```

A CUDA-capable GPU is strongly recommended. I trained on a Tesla T4 via Google Colab. CPU training is possible but will be very slow on the full dataset.

### Step 1 — Generate the personal dataset

```bash
python fine_tunning_step5/generate_dataset.py
```

Reads `profile.json` and writes `hamza_azuha_dataset.json`. Run this before training.

### Step 2 — Verify the data pipeline

```bash
python fine_tunning_step5/instruction_dataset.py
```

This downloads everyday-conversations and a sample of ultrachat, merges them with personal data, and prints the first batch shape. Datasets are cached locally after first download.

### Step 3 — Verify LoRA injection

```bash
python fine_tunning_step5/lora.py
```

Should print parameter counts before and after injection, then confirm the forward pass shape is `torch.Size([2, 8, 50257])`.

### Step 4 — Run fine-tuning

```bash
python fine_tunning_step5/finetune_train.py
```

Downloads GPT-2 small weights on first run (~498 MB). Applies LoRA, trains, and saves the model.

### Step 5 — Chat with Azuha

```bash
python fine_tunning_step5/finetune_generate.py
```

---

## Configurable Values

### `data_preperation_step1/dataloader.py`

| Variable | Default | Description |
|---|---|---|
| `vocab_size` | `50257` | GPT-2 BPE vocabulary size — do not change unless swapping tokenizer |
| `output_dim` | `256` | Embedding size in the test block — must match model emb_dim |
| `context_length` | `1024` | Maximum sequence length |
| `batch_size` | `8` | Sequences per batch |
| `max_length` | `4` | Tokens per sample in the test block |
| `stride` | `4` | Sliding window step — equal to max_length means no overlap between samples |

---

### `pretraining_step4/gpt_train.py`

| Variable | Default | Description |
|---|---|---|
| `context_length` | `256` | Shortened to 256 for small training text — full GPT-2 uses 1024 |
| `emb_dim` | `768` | Embedding dimension — tied to the GPT-2 variant |
| `n_heads` | `12` | Number of attention heads |
| `n_layers` | `12` | Transformer blocks stacked |
| `drop_rate` | `0.1` | Dropout during pretraining |
| `learning_rate` | `5e-4` | AdamW learning rate |
| `num_epochs` | `10` | Training epochs |
| `batch_size` | `2` | Reduce to 1 if out of memory |
| `weight_decay` | `0.1` | AdamW weight decay |

---

### `pretraining_step4/gpt_generate.py`

| Variable | Default | Description |
|---|---|---|
| `CHOOSE_MODEL` | `"gpt2-small (124M)"` | GPT-2 variant — small / medium (355M) / large (774M) / xl (1558M) |
| `INPUT_PROMPT` | `"Every effort moves you"` | Starting text for generation |
| `max_new_tokens` | `25` | Tokens to generate |
| `top_k` | `50` | Sample from top-k tokens only |
| `temperature` | `1.0` | 0.0 = deterministic, higher = more creative |

---

### `fine_tunning_step5/instruction_dataset.py`

| Variable | Default | Description |
|---|---|---|
| `use_everyday` | `True` | Include everyday-conversations dataset |
| `use_ultrachat` | `True` | Include ultrachat_200k dataset |
| `ultrachat_samples` | `20000` | Conversations sampled from ultrachat — reduce if training is slow |
| `personal_repeat` | `50` | Times personal data is repeated — increase if Azuha forgets personal facts |
| `allowed_max_length` | `1024` | Truncate sequences longer than this — cannot exceed context_length |
| `batch_size` | `8` | DataLoader batch size |
| `pad_token_id` | `50256` | GPT-2 end-of-text token used for padding |
| `ignore_index` | `-100` | PyTorch cross_entropy skips targets with this value |

---

### `fine_tunning_step5/lora.py`

| Variable | Default | Description |
|---|---|---|
| `rank` | `4` | Inner dimension of A and B matrices — the only value to change. Lower = fewer params / faster. Higher = better fit. Paper used 4 and 8 |
| Scaling | `1/r` | Fixed — original paper formulation, not alpha/r |
| Target layers | `W_query, W_value` | Fixed — per paper Section 5.1 |
| A initialization | `torch.randn` | Gaussian — fixed per paper Section 3 |
| B initialization | `torch.zeros` | Fixed per paper — ensures LoRA output is zero at init |

---

### `fine_tunning_step5/finetune_train.py`

| Variable | Default | Description |
|---|---|---|
| `CHOOSE_MODEL` | `"gpt2-small (124M)"` | GPT-2 variant to fine-tune |
| `LORA_RANK` | `4` | Passed to `apply_lora()` — controls trainable parameter count |
| `USE_LORA` | `True` | Set `False` to do full fine-tuning — all 163M parameters updated |
| `batch_size` | `2` | Reduce to 1 on limited GPU memory |
| `NUM_EPOCHS` | `2` | Fine-tuning epochs — monitor val_loss to decide when to stop |
| `EVAL_FREQ` | `200` | Print train/val loss every N steps |
| `EVAL_ITER` | `5` | Batches averaged per evaluation |
| `lr` | `5e-5` | AdamW learning rate — use `1e-5` for continued fine-tuning |
| `weight_decay` | `0.1` | AdamW regularization |
| `models_dir` | local path | Directory where GPT-2 weights are stored — update to your path |
| `ultrachat_samples` | `20000` | First value to reduce if training is too slow |

---

### `fine_tunning_step5/finetune_generate.py`

| Variable | Default | Description |
|---|---|---|
| `CHOOSE_MODEL` | `"gpt2-small (124M)"` | Must match `finetune_train.py` exactly |
| `LORA_RANK` | `4` | Must match `finetune_train.py` exactly |
| `max_new_tokens` | `150` | Maximum response length in tokens |
| `temperature` | `0.0` | 0.0 = deterministic, 0.7 = more varied responses |
| `top_k` | `50` | Restrict sampling to top-k tokens — set `None` to disable |
| `model_path` | auto-resolved | Resolved relative to script using `Path(__file__).resolve().parent` |

---

## Training Pipeline

```
python finetune_train.py
│
├── 1. Load tokenizer           tiktoken GPT-2 encoding
├── 2. Load datasets
│   ├── everyday-conversations  ~2,260 pairs (Apache 2.0)
│   ├── ultrachat_200k          20,000 sampled pairs (MIT)
│   └── hamza_azuha_dataset     ~110 × 50 = 5,500 personal pairs
│       total: ~29,000 pairs shuffled
│
├── 3. Split                    85% train / 10% test / 5% val
│
├── 4. AzuhaDataset             tokenize all pairs in __init__
│                               format: "User: ...\nAzuha: ..."
│
├── 5. custom_collate_fn        pad to batch max length
│                               targets: shift right by 1
│                               padding positions: set to -100 (ignored by loss)
│
├── 6. Load GPT-2 weights       download_and_load_gpt2 → load_weights_into_gpt
│
├── 7. apply_lora (rank=4)      freeze all 163M params
│                               inject LinearWithLoRA into W_query + W_value × 12 blocks
│                               147,456 params now trainable
│
├── 8. AdamW optimizer          only receives params where requires_grad=True
│                               lr=5e-5, weight_decay=0.1
│
├── 9. Training loop
│   for epoch in NUM_EPOCHS:
│     for batch in train_loader:
│       zero_grad → forward → cross_entropy loss → backward → step
│       every 200 steps: evaluate train_loss and val_loss
│     print sample response from val_data[0]
│
└── 10. Save                    gpt2-small124M-lora-r4-sft.pth
        plot                    finetune_loss.pdf
```

---

## Monitoring Training

I watch the gap between training loss and validation loss — not the absolute values alone.

```
Step 000000:  Train 3.524  Val 3.739  gap 0.215  — starting point
Step 000200:  Train 2.687  Val 2.812  gap 0.125  — learning well
Step 000400:  Train 2.005  Val 2.112  gap 0.107  — gap closing, healthy
```

Signs of overfitting — val loss rises while train loss keeps falling:

```
Step 000800:  Train 1.502  Val 2.187  gap 0.685  — watch this
Step 001000:  Train 1.301  Val 2.340  gap 1.039  — overfitting confirmed, stop here
```

With LoRA rank=4, overfitting is unlikely since only 0.09% of parameters train. The more common outcome is a plateau — both train and val loss stop improving. That is the signal to stop.

---

## Keeping Azuha Updated

`profile.json` is the single source of truth for Azuha's personal knowledge. When anything changes in my life or work, I update that file and regenerate the dataset.

```bash
# 1. Edit profile.json
# 2. Regenerate
python fine_tunning_step5/generate_dataset.py
# 3. Continue fine-tuning from last checkpoint — see below
```

Fields in `profile.json`:

| Section | What to update |
|---|---|
| `personal` | Age, contact info, city |
| `education` | Year, graduation status |
| `professional` | Role, period, summary |
| `projects` | Add new project with name, year, description |
| `technical_skills` | Add new languages, frameworks, tools |
| `certificates` | Add new certificates |
| `career_goals` | Update as goals evolve |
| `hobbies` | Sports, music, anything personal |

---

## Continued Fine-Tuning

To continue fine-tuning from a previously saved model rather than starting from GPT-2 weights again, uncomment this block in `finetune_train.py` and comment out the GPT-2 weight loading section:

```python
# load previously fine-tuned model for continued training
model = GPTModel(cfg)
apply_lora(model, rank=LORA_RANK)    # inject LoRA structure first
model.load_state_dict(               # then load saved weights
    torch.load(
        "gpt2-small124M-lora-r4-sft.pth",
        map_location=device,
        weights_only=True
    )
)
```

Use a lower learning rate for continued fine-tuning:

```python
lr=1e-5   # instead of 5e-5 — smaller adjustments on already-trained model
```

Save under a new filename to keep versions separate:

```python
save_path = "azuha-v2-lora-r4.pth"
```

---

## Chat Format

```
User: What is your name?
Azuha: My name is Azuha. I am Hamza's personal AI assistant.

User: Who built you?
Azuha: I was built by Muhammad Hamza Waheed, a 3rd year Electrical
       Engineering student at NUST, completely from scratch using
       PyTorch and LoRA fine-tuning.

User: What is Hamza working on?
Azuha: Hamza is currently building me — a GPT-style large language
       model from scratch. He has also worked on ECG arrhythmia
       classification, an EEG signal processing pipeline, an
       AI-powered CV maker deployed on Azure, and a multi-agent
       RAG knowledge system.

User: What sports does Hamza play?
Azuha: Hamza plays football and table tennis.
```

---

## References

- Raschka, S. — *Build a Large Language Model From Scratch*, Manning Publications
- Hu, E. et al. (2021) — *LoRA: Low-Rank Adaptation of Large Language Models*
- OpenAI — GPT-2 model weights and BPE tokenizer
- HuggingFace — `HuggingFaceTB/everyday-conversations-llama3.1-2k` (Apache 2.0)
- HuggingFace — `HuggingFaceH4/ultrachat_200k` (MIT)

---

*PyTorch · tiktoken · LoRA · GPT-2 · HuggingFace Datasets*