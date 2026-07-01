# instruction_dataset.py
# Chapter 7 — data preparation for instruction fine-tuning
#
# What this file does:
#   1. Loads the Alpaca dataset from HuggingFace (52,002 entries)
#   2. Formats each entry into Alpaca prompt style
#   3. Splits into train / val / test sets
#   4. Defines InstructionDataset (replaces GPTDatasetV1 from Chapter 2)
#   5. Defines custom_collate_fn (pads batches, masks padding with -100)
#   6. Creates and returns train / val / test DataLoaders
#
# Imports from YOUR existing files:
#   None — this file is self-contained, imported BY finetune_train.py
#
# NOTE: requires `pip install datasets` for HuggingFace dataset loading.
# NOTE: The Alpaca dataset is licensed CC BY-NC 4.0 (non-commercial use only).
#       Verify this matches your intended use before proceeding.

import torch
import tiktoken
from torch.utils.data import Dataset, DataLoader
from functools import partial


# ─────────────────────────────────────────────
# 1. LOAD ALPACA DATASET FROM HUGGINGFACE
# ─────────────────────────────────────────────

def load_alpaca_data():
    """
    Downloads and returns the Alpaca dataset as a plain Python list of dicts.
    Each dict has keys: 'instruction', 'input', 'output', 'text'
    Requires: pip install datasets
    I am not 100% certain this API is unchanged — verify against
    the official HuggingFace datasets documentation.
    """
    from datasets import load_dataset
    dataset = load_dataset("tatsu-lab/alpaca")
    data = dataset["train"].to_list()
    print(f"Loaded {len(data)} entries from Alpaca dataset.")
    return data


# ─────────────────────────────────────────────
# 2. ALPACA PROMPT FORMAT (exactly as Chapter 7)
# ─────────────────────────────────────────────

def format_input(entry):
    """
    Formats a dataset entry into the Alpaca prompt style.
    This is the exact format_input function from Chapter 7 of the book.
    The ### Input: section is omitted if entry['input'] is empty.
    """
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )
    input_text = (
        f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""
    )
    return instruction_text + input_text


# ─────────────────────────────────────────────
# 3. TRAIN / VAL / TEST SPLIT
# ─────────────────────────────────────────────

def split_data(data, train_frac=0.85, test_frac=0.10):
    """
    Splits data into train, test, and validation sets.
    Follows the same ratios as Chapter 7:
        train: 85%, test: 10%, val: remaining 5%
    """
    train_portion = int(len(data) * train_frac)
    test_portion  = int(len(data) * test_frac)

    train_data = data[:train_portion]
    test_data  = data[train_portion:train_portion + test_portion]
    val_data   = data[train_portion + test_portion:]

    print(f"Training set:   {len(train_data)} entries")
    print(f"Validation set: {len(val_data)} entries")
    print(f"Test set:       {len(test_data)} entries")

    return train_data, val_data, test_data


# ─────────────────────────────────────────────
# 4. INSTRUCTION DATASET CLASS
#    Replaces GPTDatasetV1 from Chapter 2
# ─────────────────────────────────────────────

class InstructionDataset(Dataset):
    """
    Tokenizes and stores all instruction-response pairs.
    Each entry is the full formatted text:
        format_input(entry) + "\n\n### Response:\n" + entry['output']
    Pre-tokenized in __init__ so the DataLoader doesn't redo it per batch.
    """
    def __init__(self, data, tokenizer):
        self.data = data
        self.encoded_texts = []

        for entry in data:
            instruction_plus_input = format_input(entry)
            response_text = f"\n\n### Response:\n{entry['output']}"
            full_text = instruction_plus_input + response_text
            self.encoded_texts.append(
                tokenizer.encode(full_text)
            )

    def __getitem__(self, index):
        return self.encoded_texts[index]

    def __len__(self):
        return len(self.data)


# ─────────────────────────────────────────────
# 5. CUSTOM COLLATE FUNCTION
#    This is the key Chapter 7 addition vs Chapter 2
# ─────────────────────────────────────────────

def custom_collate_fn(
    batch,
    pad_token_id=50256,
    ignore_index=-100,
    allowed_max_length=None,
    device="cpu"
):
    """
    Pads all sequences in a batch to the same length.
    Replaces extra padding tokens in targets with -100 so
    PyTorch's cross_entropy ignores them (ignore_index=-100 by default).
    One padding token (50256) is kept in targets so the model learns
    when to stop generating.

    Steps (exactly as Chapter 7 figure 7.6):
        2.3  pad to same length with token 50256
        2.4  create target = input shifted right by 1
        2.5  replace all but first padding token in target with -100
    """
    batch_max_length = max(len(item) + 1 for item in batch)
    inputs_lst, targets_lst = [], []

    for item in batch:
        new_item = item.copy()
        new_item += [pad_token_id]  # append one end-of-text

        # pad to batch max length
        padded = (
            new_item +
            [pad_token_id] * (batch_max_length - len(new_item))
        )

        inputs  = torch.tensor(padded[:-1])   # drop last token for input
        targets = torch.tensor(padded[1:])    # shift right by 1 for target

        # replace all but the FIRST padding token in targets with -100
        # so cross_entropy ignores them
        mask = targets == pad_token_id
        indices = torch.nonzero(mask).squeeze()
        if indices.numel() > 1:
            targets[indices[1:]] = ignore_index

        # optional: truncate to model's max context length
        if allowed_max_length is not None:
            inputs  = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]

        inputs_lst.append(inputs)
        targets_lst.append(targets)

    inputs_tensor  = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)
    return inputs_tensor, targets_tensor


# ─────────────────────────────────────────────
# 6. CREATE DATALOADERS
# ─────────────────────────────────────────────

def create_dataloaders(train_data, val_data, test_data,
                       tokenizer, device, batch_size=8):
    """
    Creates and returns train, val, test DataLoaders.
    Uses the custom_collate_fn with device and allowed_max_length=1024
    pre-filled via functools.partial (same approach as Chapter 7).
    """
    customized_collate = partial(
        custom_collate_fn,
        device=device,
        allowed_max_length=1024   # GPT-2's context length
    )

    train_loader = DataLoader(
        InstructionDataset(train_data, tokenizer),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=customized_collate,
        num_workers=0
    )
    val_loader = DataLoader(
        InstructionDataset(val_data, tokenizer),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=customized_collate,
        num_workers=0
    )
    test_loader = DataLoader(
        InstructionDataset(test_data, tokenizer),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=customized_collate,
        num_workers=0
    )

    return train_loader, val_loader, test_loader


# ─────────────────────────────────────────────
# QUICK TEST — run this file directly to verify
# ─────────────────────────────────────────────

if __name__ == "__main__":
    tokenizer = tiktoken.get_encoding("gpt2")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = load_alpaca_data()
    train_data, val_data, test_data = split_data(data)
    train_loader, val_loader, test_loader = create_dataloaders(
        train_data, val_data, test_data, tokenizer, device
    )

    # print first batch shape to confirm it works
    for inputs, targets in train_loader:
        print("inputs shape: ", inputs.shape)
        print("targets shape:", targets.shape)
        break

    # print one formatted example so you can see the prompt format
    print("\n--- Example formatted entry ---")
    example = format_input(train_data[0])
    print(example)
    print(f"\n\n### Response:\n{train_data[0]['output']}")
