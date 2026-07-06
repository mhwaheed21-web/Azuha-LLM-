# lora.py
# LoRA implementation faithful to the original paper:
#   Hu et al. (2021) — "LoRA: Low-Rank Adaptation of Large Language Models"
#   All design decisions below are sourced directly from your reading of that paper.
#
# Exact choices confirmed from the paper:
#   - Applied to   : W_query and W_value only
#   - A init       : random Gaussian (normal)
#   - B init       : zeros
#   - Scaling      : 1/r  (original paper — not alpha/r)
#   - LoRA dropout : none
#   - MLP layers   : frozen, not adapted
#
# WHERE TO CONTROL LORA SIZE:
#   Only ONE value to set: RANK (r)
#   Find it in finetune_train.py as LORA_RANK.
#   Lower r = fewer trainable params, faster, less memory.
#   Higher r = more trainable params, better fit, more memory.
#   The paper used r = 4 and r = 8 in their experiments.
#   I cannot confirm what the optimal value is for your specific
#   use case — experiment and monitor validation loss.
#
# Folder: finetune_step4/lora.py
# Imported by: finetune_step4/finetune_train.py

import torch
import torch.nn as nn
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


# ─────────────────────────────────────────────
# 1. LORA LAYER
#    Implements the low-rank branch: (1/r) * B @ A @ x
#    This is added alongside the frozen original linear layer.
# ─────────────────────────────────────────────

class LoRALayer(nn.Module):
    """
    The LoRA branch for one Linear layer.

    From the paper (Section 3):
        h = W0 x + (1/r) * B A x

    Where:
        W0 : frozen original weight — NOT inside this class
        A  : shape [rank, in_features]  — Gaussian init
        B  : shape [out_features, rank] — zero init

    B is zero at init so the LoRA output starts at exactly zero,
    meaning the model starts identical to the pretrained model.
    Training then gradually shifts A and B away from zero.

    Note on matrix orientation:
        nn.Linear stores weight as [out_features, in_features].
        To match that convention and make the matmul work cleanly
        through PyTorch's F.linear, we store:
            A: [rank, in_features]
            B: [out_features, rank]
        So the forward pass is: x @ A.T @ B.T
        which equals (B A x) from the paper's column-vector notation.

    Args:
        in_features  : input size of the original Linear
        out_features : output size of the original Linear
        rank         : r in the paper — the inner dimension of A and B
    """
    def __init__(self, in_features, out_features, rank):
        super().__init__()
        self.rank = rank

        # A: Gaussian (normal) initialization — confirmed from paper Section 3
        self.A = nn.Parameter(torch.randn(rank, in_features))

        # B: zero initialization — confirmed from paper Section 3
        # "so that delta_W = BA is zero at the beginning of training"
        self.B = nn.Parameter(torch.zeros(out_features, rank))

    def forward(self, x):
        # x @ A.T  →  [..., rank]
        # @ B.T    →  [..., out_features]
        # * (1/r)  →  scale as per paper Section 3
        return (x @ self.A.T @ self.B.T) * (1.0 / self.rank)


# ─────────────────────────────────────────────
# 2. LINEAR WITH LORA
#    Wraps a frozen nn.Linear and adds the LoRA branch in parallel.
# ─────────────────────────────────────────────

class LinearWithLoRA(nn.Module):
    """
    Wraps an existing frozen nn.Linear and adds a LoRALayer alongside it.

    Forward pass (from paper equation):
        output = W0 x + (1/r) * B A x
               = frozen_linear(x) + lora(x)

    The frozen_linear weights are NOT updated during training.
    Only lora.A and lora.B are updated.

    Args:
        linear : the nn.Linear to wrap — its weights will be frozen here
        rank   : r from the paper
    """
    def __init__(self, linear, rank):
        super().__init__()

        self.linear = linear
        self.lora   = LoRALayer(
            in_features  = linear.in_features,
            out_features = linear.out_features,
            rank         = rank
        )

        # freeze the original linear weights
        # confirmed from paper — W0 does not receive gradient updates
        for param in self.linear.parameters():
            param.requires_grad = False

    def forward(self, x):
        # frozen path + LoRA path
        return self.linear(x) + self.lora(x)


# ─────────────────────────────────────────────
# 3. APPLY LORA TO GPTMODEL
#    Step 1: freeze all model parameters (from Chapter 6 of your book)
#    Step 2: inject LinearWithLoRA into W_query and W_value only
#            (confirmed from paper Section 5.1 and 6.1)
#    Step 3: MLP/feedforward layers stay frozen — not adapted
#            (confirmed from paper Section 3.1)
# ─────────────────────────────────────────────

def apply_lora(model, rank):
    """
    Injects LoRA into W_query and W_value of every TransformerBlock.

    Freezing approach taken directly from your book (Chapter 6, page 186):
        for param in model.parameters():
            param.requires_grad = False

    Then W_query and W_value in each block's MultiHeadAttention are
    replaced with LinearWithLoRA wrappers. The LoRALayer inside each
    wrapper has requires_grad=True by default for A and B.

    W_key, out_proj, and all FeedForward layers stay frozen.
    This matches the paper's Section 3.1 and 5.1 exactly.

    Args:
        model : your GPTModel instance — modified in place
        rank  : r from the paper
                ─────────────────────────────────────────
                THIS IS YOUR MAIN TUNING KNOB.
                The paper used r = 4 and r = 8.
                I am not certain which is best for your
                use case — monitor val_loss to decide.
                ─────────────────────────────────────────

    Returns:
        model : same object, modified in place
    """

    # STEP 1 — freeze everything
    # source: your book Chapter 6, page 186, confirmed from PDF
    for param in model.parameters():
        param.requires_grad = False

    # STEP 2 — inject LoRA into W_query and W_value only
    # source: paper Section 5.1 — "we only apply LoRA to Wq and Wv"
    for block in model.trf_blocks:
        att = block.att  # MultiHeadAttention instance

        # replace W_query
        att.W_query = LinearWithLoRA(att.W_query, rank=rank)

        # replace W_value
        att.W_value = LinearWithLoRA(att.W_value, rank=rank)

        # W_key, out_proj — left frozen, not adapted (paper Section 6.1)

    return model


# ─────────────────────────────────────────────
# 4. COUNT PARAMETERS — run after apply_lora()
#    to confirm how many params will actually train
# ─────────────────────────────────────────────

def count_parameters(model):
    """
    Prints trainable vs total parameter counts.
    Call this right after apply_lora() to verify the reduction.
    The percentage should be very small — I believe roughly 1-5%
    for typical rank values, but you should verify this yourself
    by running count_parameters() on your specific model and rank.
    """
    trainable = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    total = sum(
        p.numel() for p in model.parameters()
    )
    pct = 100.0 * trainable / total

    print(f"Trainable parameters : {trainable:,}")
    print(f"Total parameters     : {total:,}")
    print(f"Trainable %          : {pct:.4f}%")

    return trainable, total


# ─────────────────────────────────────────────
# QUICK TEST — run this file directly to verify
# python finetune_step4/lora.py
# ─────────────────────────────────────────────

if __name__ == "__main__":
    from llm_architecture_step3.gpt import GPTModel

    # use small (124M) config just for the test
    cfg = {
        "vocab_size"     : 50257,
        "context_length" : 1024,
        "emb_dim"        : 768,
        "n_heads"        : 12,
        "n_layers"       : 12,
        "drop_rate"      : 0.0,
        "qkv_bias"       : True
    }

    model = GPTModel(cfg)

    print("=== BEFORE LoRA ===")
    count_parameters(model)

    apply_lora(model, rank=4)

    print("\n=== AFTER LoRA (rank=4) ===")
    count_parameters(model)

    # confirm forward pass still works after injection
    dummy_input = torch.randint(0, 50257, (2, 8))
    output = model(dummy_input)
    print(f"\nForward pass shape : {output.shape}")
    print("Expected           : torch.Size([2, 8, 50257])")
    print("LoRA injection test passed." if output.shape == torch.Size([2, 8, 50257])
          else "SHAPE MISMATCH — something is wrong.")
