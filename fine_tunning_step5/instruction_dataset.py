# # instruction_dataset.py
# # Chapter 7 — data preparation for instruction fine-tuning
# #
# # What this file does:
# #   1. Loads the Alpaca dataset from HuggingFace (52,002 entries)
# #   2. Formats each entry into Alpaca prompt style
# #   3. Splits into train / val / test sets
# #   4. Defines InstructionDataset (replaces GPTDatasetV1 from Chapter 2)
# #   5. Defines custom_collate_fn (pads batches, masks padding with -100)
# #   6. Creates and returns train / val / test DataLoaders
# #
# # Imports from YOUR existing files:
# #   None — this file is self-contained, imported BY finetune_train.py
# #
# # NOTE: requires `pip install datasets` for HuggingFace dataset loading.
# # NOTE: The Alpaca dataset is licensed CC BY-NC 4.0 (non-commercial use only).
# #       Verify this matches your intended use before proceeding.

# import torch
# import json
# import tiktoken
# from torch.utils.data import Dataset, DataLoader
# from functools import partial


# # ─────────────────────────────────────────────
# # 1. LOAD ALPACA DATASET FROM HUGGINGFACE
# # ─────────────────────────────────────────────

# def load_alpaca_data():
#     """
#     Downloads and returns the Alpaca dataset as a plain Python list of dicts.
#     Each dict has keys: 'instruction', 'input', 'output', 'text'
#     Requires: pip install datasets
#     I am not 100% certain this API is unchanged — verify against
#     the official HuggingFace datasets documentation.
#     """
#     from datasets import load_dataset
#     dataset = load_dataset("tatsu-lab/alpaca")
#     data = dataset["train"].to_list()
#     print(f"Loaded {len(data)} entries from Alpaca dataset.")
#     return data



# # def load_custom_data(file_path):
# #     """
# #     Loads your personal JSON entries.
# #     File must be a list of dicts with keys:
# #     instruction, input, output
# #     """
# #     with open(file_path, "r", encoding="utf-8") as f:
# #         custom_data = json.load(f)
# #     print(f"Loaded {len(custom_data)} custom entries.")
# #     return custom_data


# # def load_and_merge_data(custom_data_path=None):
# #     """
# #     Loads Alpaca and optionally merges your custom data.
# #     """
# #     # load alpaca
# #     data = load_alpaca_data()

# #     # merge custom entries if provided
# #     if custom_data_path:
# #         custom = load_custom_data(custom_data_path)
# #         data = data + custom   # append your entries at the end
# #         print(f"Total after merge: {len(data)} entries.")

# #     return data
# # ─────────────────────────────────────────────
# # 2. ALPACA PROMPT FORMAT (exactly as Chapter 7)
# # ─────────────────────────────────────────────

# def format_input(entry):
#     """
#     Formats a dataset entry into the Alpaca prompt style.
#     This is the exact format_input function from Chapter 7 of the book.
#     The ### Input: section is omitted if entry['input'] is empty.
#     """
#     instruction_text = (
#         f"Below is an instruction that describes a task. "
#         f"Write a response that appropriately completes the request."
#         f"\n\n### Instruction:\n{entry['instruction']}"
#     )
#     input_text = (
#         f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""
#     )
#     return instruction_text + input_text


# # ─────────────────────────────────────────────
# # 3. TRAIN / VAL / TEST SPLIT
# # ─────────────────────────────────────────────

# def split_data(data, train_frac=0.85, test_frac=0.10):
#     """
#     Splits data into train, test, and validation sets.
#     Follows the same ratios as Chapter 7:
#         train: 85%, test: 10%, val: remaining 5%
#     """
#     train_portion = int(len(data) * train_frac)
#     test_portion  = int(len(data) * test_frac)

#     train_data = data[:train_portion]
#     test_data  = data[train_portion:train_portion + test_portion]
#     val_data   = data[train_portion + test_portion:]

#     print(f"Training set:   {len(train_data)} entries")
#     print(f"Validation set: {len(val_data)} entries")
#     print(f"Test set:       {len(test_data)} entries")

#     return train_data, val_data, test_data


# # ─────────────────────────────────────────────
# # 4. INSTRUCTION DATASET CLASS
# #    Replaces GPTDatasetV1 from Chapter 2
# # ─────────────────────────────────────────────

# class InstructionDataset(Dataset):
#     """
#     Tokenizes and stores all instruction-response pairs.
#     Each entry is the full formatted text:
#         format_input(entry) + "\n\n### Response:\n" + entry['output']
#     Pre-tokenized in __init__ so the DataLoader doesn't redo it per batch.
#     """
#     def __init__(self, data, tokenizer):
#         self.data = data
#         self.encoded_texts = []

#         for entry in data:
#             instruction_plus_input = format_input(entry)
#             response_text = f"\n\n### Response:\n{entry['output']}"
#             full_text = instruction_plus_input + response_text
#             self.encoded_texts.append(
#                 tokenizer.encode(full_text)
#             )

#     def __getitem__(self, index):
#         return self.encoded_texts[index]

#     def __len__(self):
#         return len(self.data)


# # ─────────────────────────────────────────────
# # 5. CUSTOM COLLATE FUNCTION
# #    This is the key Chapter 7 addition vs Chapter 2
# # ─────────────────────────────────────────────

# def custom_collate_fn(
#     batch,
#     pad_token_id=50256,
#     ignore_index=-100,
#     allowed_max_length=None,
#     device="cpu"
# ):
#     """
#     Pads all sequences in a batch to the same length.
#     Replaces extra padding tokens in targets with -100 so
#     PyTorch's cross_entropy ignores them (ignore_index=-100 by default).
#     One padding token (50256) is kept in targets so the model learns
#     when to stop generating.

#     Steps (exactly as Chapter 7 figure 7.6):
#         2.3  pad to same length with token 50256
#         2.4  create target = input shifted right by 1
#         2.5  replace all but first padding token in target with -100
#     """
#     batch_max_length = max(len(item) + 1 for item in batch)
#     inputs_lst, targets_lst = [], []

#     for item in batch:
#         new_item = item.copy()
#         new_item += [pad_token_id]  # append one end-of-text

#         # pad to batch max length
#         padded = (
#             new_item +
#             [pad_token_id] * (batch_max_length - len(new_item))
#         )

#         inputs  = torch.tensor(padded[:-1])   # drop last token for input
#         targets = torch.tensor(padded[1:])    # shift right by 1 for target

#         # replace all but the FIRST padding token in targets with -100
#         # so cross_entropy ignores them
#         mask = targets == pad_token_id
#         indices = torch.nonzero(mask).squeeze()
#         if indices.numel() > 1:
#             targets[indices[1:]] = ignore_index

#         # optional: truncate to model's max context length
#         if allowed_max_length is not None:
#             inputs  = inputs[:allowed_max_length]
#             targets = targets[:allowed_max_length]

#         inputs_lst.append(inputs)
#         targets_lst.append(targets)

#     inputs_tensor  = torch.stack(inputs_lst).to(device)
#     targets_tensor = torch.stack(targets_lst).to(device)
#     return inputs_tensor, targets_tensor


# # ─────────────────────────────────────────────
# # 6. CREATE DATALOADERS
# # ─────────────────────────────────────────────

# def create_dataloaders(train_data, val_data, test_data,
#                        tokenizer, device, batch_size=8):
#     """
#     Creates and returns train, val, test DataLoaders.
#     Uses the custom_collate_fn with device and allowed_max_length=1024
#     pre-filled via functools.partial (same approach as Chapter 7).
#     """
#     customized_collate = partial(
#         custom_collate_fn,
#         device=device,
#         allowed_max_length=1024   # GPT-2's context length
#     )

#     train_loader = DataLoader(
#         InstructionDataset(train_data, tokenizer),
#         batch_size=batch_size,
#         shuffle=True,
#         drop_last=True,
#         collate_fn=customized_collate,
#         num_workers=0
#     )
#     val_loader = DataLoader(
#         InstructionDataset(val_data, tokenizer),
#         batch_size=batch_size,
#         shuffle=False,
#         drop_last=False,
#         collate_fn=customized_collate,
#         num_workers=0
#     )
#     test_loader = DataLoader(
#         InstructionDataset(test_data, tokenizer),
#         batch_size=batch_size,
#         shuffle=False,
#         drop_last=False,
#         collate_fn=customized_collate,
#         num_workers=0
#     )

#     return train_loader, val_loader, test_loader


# # ─────────────────────────────────────────────
# # QUICK TEST — run this file directly to verify
# # ─────────────────────────────────────────────

# if __name__ == "__main__":
#     tokenizer = tiktoken.get_encoding("gpt2")
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     data = load_alpaca_data()
#     # data = load_and_merge_data(
#     #     custom_data_path="my_custom_data.json"
#     # )
#     train_data, val_data, test_data = split_data(data)
#     train_loader, val_loader, test_loader = create_dataloaders(
#         train_data, val_data, test_data, tokenizer, device
#     )

#     # print first batch shape to confirm it works
#     for inputs, targets in train_loader:
#         print("inputs shape: ", inputs.shape)
#         print("targets shape:", targets.shape)
#         break

#     # print one formatted example so you can see the prompt format
#     print("\n--- Example formatted entry ---")
#     example = format_input(train_data[0])
#     print(example)
#     print(f"\n\n### Response:\n{train_data[0]['output']}")













# instruction_dataset.py
# Chapter 7 — data preparation for instruction fine-tuning
#
# Loads and merges THREE data sources:
#   1. OpenAssistant oasst1  — 84,400 rows, Apache 2.0 license
#      Filtered to: English only, non-deleted, best-ranked assistant replies
#   2. Alpaca                — 52,002 instruction-response pairs, CC BY-NC 4.0
#   3. Your custom JSON      — personal data about Hamza and Azuha
#
# Final merged dataset is split into train / val / test and
# wrapped into PyTorch DataLoaders ready for fine-tuning.
#
# Key difference from previous version:
#   oasst1 is a CONVERSATION TREE, not a flat list.
#   Each prompter message can have multiple assistant replies.
#   We extract only: English, non-deleted, rank=0 (highest quality) pairs.
#
# Folder: fine_tunning_step5/instruction_dataset.py

import json
import random
import torch
import tiktoken
from pathlib import Path
from functools import partial
from torch.utils.data import Dataset, DataLoader


# ─────────────────────────────────────────────
# 1. LOAD OASST1
#    Extracts clean English prompter→assistant pairs
#    from the conversation tree structure.
# ─────────────────────────────────────────────

def load_oasst1_data(lang="en", min_quality=0.6):
    """
    Loads OpenAssistant oasst1 and extracts clean instruction-response pairs.

    oasst1 is a conversation tree. Each row is either a 'prompter' message
    or an 'assistant' message. We extract pairs by:
        1. Filter to English only (lang == 'en')
        2. Remove deleted messages
        3. Keep only assistant messages with rank == 0
           (rank 0 = the highest-voted reply to that prompt)
        4. Find the parent prompter message for each assistant reply
        5. Return as flat list of {instruction, input, output} dicts

    Args:
        lang        : language filter — 'en' for English only
        min_quality : minimum quality score filter (0.0 to 1.0)
                      based on the 'quality' label in the dataset
                      I am not certain this threshold is optimal — verify
                      by inspecting the data and adjusting.

    Returns:
        list of dicts with keys: instruction, input, output
    """
    from datasets import load_dataset

    print("Loading oasst1 dataset...")
    dataset = load_dataset("OpenAssistant/oasst1")

    # combine train and validation into one pool for extraction
    all_rows = dataset["train"].to_list() + dataset["validation"].to_list()
    print(f"Total oasst1 rows: {len(all_rows)}")

    # build a lookup dict: message_id -> row
    # so we can find parent messages quickly
    id_to_row = {row["message_id"]: row for row in all_rows}

    pairs = []
    skipped = 0

    for row in all_rows:
        # only process assistant messages
        if row["role"] != "assistant":
            continue

        # skip deleted messages
        if row.get("deleted", False):
            continue

        # only English
        if row.get("lang", "") != lang:
            continue

        # rank == 0 means this is the best-ranked assistant reply
        # for this particular prompt (lower rank = better)
        # None means no ranking was done — skip those
        if row.get("rank") != 0:
            continue

        # find the parent prompter message
        parent_id = row.get("parent_id")
        if not parent_id or parent_id not in id_to_row:
            skipped += 1
            continue

        parent = id_to_row[parent_id]

        # parent must also be English, non-deleted, and a prompter
        if parent.get("role") != "prompter":
            continue
        if parent.get("deleted", False):
            continue
        if parent.get("lang", "") != lang:
            continue

        # optional quality filter using the labels field
        quality_score = None
        labels = row.get("labels")
        if labels and isinstance(labels, dict):
            names  = labels.get("name", [])
            values = labels.get("value", [])
            if "quality" in names:
                idx = names.index("quality")
                if idx < len(values):
                    quality_score = values[idx]

        if quality_score is not None and quality_score < min_quality:
            skipped += 1
            continue

        pairs.append({
            "instruction": parent["text"].strip(),
            "input": "",
            "output": row["text"].strip()
        })

    print(f"Extracted {len(pairs)} clean oasst1 pairs (skipped {skipped})")
    return pairs


# ─────────────────────────────────────────────
# 2. LOAD ALPACA
# ─────────────────────────────────────────────

def load_alpaca_data():
    """
    Loads the Alpaca dataset from HuggingFace.
    Returns a list of dicts with keys: instruction, input, output.
    Cached after first download — no re-download on subsequent runs.
    License: CC BY-NC 4.0 (non-commercial use only).
    """
    from datasets import load_dataset
    print("Loading Alpaca dataset...")
    dataset = load_dataset("tatsu-lab/alpaca")
    data = dataset["train"].to_list()
    print(f"Loaded {len(data)} Alpaca entries.")
    return data


# ─────────────────────────────────────────────
# 3. LOAD CUSTOM PERSONAL DATA
# ─────────────────────────────────────────────

def load_custom_data(file_path):
    """
    Loads your personal Hamza/Azuha dataset from a JSON file.
    Generated by generate_dataset.py.
    Format: list of {instruction, input, output} dicts.
    """
    path = Path(file_path)
    if not path.exists():
        print(f"Custom data file not found: {file_path}")
        print("Skipping custom data — run generate_dataset.py first.")
        return []

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Loaded {len(data)} custom personal entries from {file_path}")
    return data


# ─────────────────────────────────────────────
# 4. MERGE ALL THREE SOURCES
# ─────────────────────────────────────────────

def load_and_merge_data(
    use_oasst1=True,
    use_alpaca=True,
    custom_data_path=None,
    personal_repeat=20,
    oasst1_lang="en",
    oasst1_min_quality=0.6,
    shuffle=True
):
    """
    Loads and merges oasst1, Alpaca, and custom personal data.

    Args:
        use_oasst1       : whether to include oasst1
        use_alpaca       : whether to include Alpaca
        custom_data_path : path to your personal JSON file (or None to skip)
        personal_repeat  : how many times to repeat personal entries
                           to ensure the model learns them despite being
                           a small fraction of the total data.
                           I am not certain of the ideal value — start at 20
                           and adjust based on whether the model remembers
                           personal facts after fine-tuning.
        oasst1_lang      : language filter for oasst1
        oasst1_min_quality: minimum quality threshold for oasst1 pairs
        shuffle          : whether to shuffle the merged dataset

    Returns:
        merged list of {instruction, input, output} dicts
    """
    merged = []

    if use_oasst1:
        oasst1_data = load_oasst1_data(
            lang=oasst1_lang,
            min_quality=oasst1_min_quality
        )
        merged += oasst1_data

    if use_alpaca:
        alpaca_data = load_alpaca_data()
        merged += alpaca_data

    if custom_data_path:
        custom_data = load_custom_data(custom_data_path)
        if custom_data:
            # repeat personal entries so model learns them reliably
            repeated = custom_data * personal_repeat
            merged += repeated
            print(f"Personal data: {len(custom_data)} entries × {personal_repeat} = {len(repeated)} total")

    print(f"\nTotal merged entries: {len(merged)}")

    if shuffle:
        random.shuffle(merged)

    return merged


# ─────────────────────────────────────────────
# 5. ALPACA PROMPT FORMAT — same as before
# ─────────────────────────────────────────────

def format_input(entry):
    """
    Formats a dataset entry into the Alpaca prompt style.
    Works for all three data sources since they all share
    the same {instruction, input, output} structure.
    """
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )
    input_text = (
        f"\n\n### Input:\n{entry['input']}" if entry.get("input") else ""
    )
    return instruction_text + input_text


# ─────────────────────────────────────────────
# 6. TRAIN / VAL / TEST SPLIT
# ─────────────────────────────────────────────

def split_data(data, train_frac=0.85, test_frac=0.10):
    """
    Splits into train (85%), test (10%), val (5%).
    Same ratios as Chapter 7 of the book.
    """
    train_portion = int(len(data) * train_frac)
    test_portion  = int(len(data) * test_frac)

    train_data = data[:train_portion]
    test_data  = data[train_portion:train_portion + test_portion]
    val_data   = data[train_portion + test_portion:]

    print(f"Training set  : {len(train_data)} entries")
    print(f"Validation set: {len(val_data)} entries")
    print(f"Test set      : {len(test_data)} entries")

    return train_data, val_data, test_data


# ─────────────────────────────────────────────
# 7. INSTRUCTION DATASET CLASS
# ─────────────────────────────────────────────

class InstructionDataset(Dataset):
    """
    Tokenizes and stores all instruction-response pairs.
    Replaces GPTDatasetV1 from Chapter 2.
    All tokenization done in __init__ — not repeated per batch.
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
# 8. CUSTOM COLLATE FUNCTION
# ─────────────────────────────────────────────

def custom_collate_fn(
    batch,
    pad_token_id=50256,
    ignore_index=-100,
    allowed_max_length=None,
    device="cpu"
):
    """
    Pads sequences to same length. Replaces all but the first
    padding token in targets with -100 so cross_entropy ignores them.
    This is the Chapter 7 collate approach — unchanged.
    """
    batch_max_length = max(len(item) + 1 for item in batch)
    inputs_lst, targets_lst = [], []

    for item in batch:
        new_item = item.copy()
        new_item += [pad_token_id]
        padded = (
            new_item +
            [pad_token_id] * (batch_max_length - len(new_item))
        )

        inputs  = torch.tensor(padded[:-1])
        targets = torch.tensor(padded[1:])

        mask = targets == pad_token_id
        indices = torch.nonzero(mask).squeeze()
        if indices.numel() > 1:
            targets[indices[1:]] = ignore_index

        if allowed_max_length is not None:
            inputs  = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]

        inputs_lst.append(inputs)
        targets_lst.append(targets)

    return (
        torch.stack(inputs_lst).to(device),
        torch.stack(targets_lst).to(device)
    )


# ─────────────────────────────────────────────
# 9. CREATE DATALOADERS
# ─────────────────────────────────────────────

def create_dataloaders(train_data, val_data, test_data,
                       tokenizer, device, batch_size=8):
    """
    Creates train, val, test DataLoaders.
    Uses functools.partial to pre-fill device and allowed_max_length
    into custom_collate_fn before passing to DataLoader.
    """
    customized_collate = partial(
        custom_collate_fn,
        device=device,
        allowed_max_length=1024 #reduce to 512 if you run out of GPU memory
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
# python fine_tunning_step5/instruction_dataset.py
# ─────────────────────────────────────────────

if __name__ == "__main__":
    tokenizer = tiktoken.get_encoding("gpt2")
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # adjust custom_data_path to where your generated file is
    data = load_and_merge_data(
        use_oasst1=True,
        use_alpaca=True,
        custom_data_path=Path(__file__).resolve().parent / "my_custom_data.json",
        personal_repeat=50
    )

    train_data, val_data, test_data = split_data(data)
    train_loader, val_loader, test_loader = create_dataloaders(
        train_data, val_data, test_data,
        tokenizer=tokenizer,
        device=device,
        batch_size=2
    )

    # print first batch shape
    for inputs, targets in train_loader:
        print("inputs shape :", inputs.shape)
        print("targets shape:", targets.shape)
        break

    # print one example to verify format
    print("\n--- Example formatted entry ---")
    print(format_input(train_data[0]))
    print(f"\n### Response:\n{train_data[0]['output']}")