# # finetune_train.py
# # Chapter 7 — instruction fine-tuning training loop
# #
# # What this file does:
# #   1. Loads GPT-2 medium (355M) pretrained weights
# #   2. Sets up train/val/test dataloaders from Alpaca dataset
# #   3. Runs the fine-tuning training loop
# #   4. Saves the fine-tuned model as gpt2-medium355M-sft.pth
# #
# # Imports from YOUR existing files:
# #   gpt.py          → GPTModel, generate_text_simple
# #   gpt_download.py → download_and_load_gpt2
# #   gpt_generate.py → load_weights_into_gpt, text_to_token_ids, token_ids_to_text
# #
# # Imports from NEW files:
# #   instruction_dataset.py → load_alpaca_data, split_data, create_dataloaders,
# #                            format_input
# #
# # NOTE: The book recommends gpt2-medium (355M) over gpt2-small (124M) for
# #       instruction fine-tuning. The 124M model is described as lacking
# #       sufficient capacity for good results.
# # NOTE: The 355M model download is approximately 1.42 GB (confirmed from book PDF).
# #       Verify your available disk space before running.

# import sys
# import torch
# import tiktoken
# import matplotlib.pyplot as plt
# from pathlib import Path

# # ── add project root to sys.path so imports from sibling folders work ──
# project_root = Path(__file__).resolve().parents[1]
# if str(project_root) not in sys.path:
#     sys.path.insert(0, str(project_root))

# from llm_architecture_step3.gpt import GPTModel, generate_text_simple
# from pretraining_step4.gpt_download import download_and_load_gpt2
# from pretraining_step4.gpt_generate import load_weights_into_gpt, text_to_token_ids, token_ids_to_text
# from fine_tunning_step5.instruction_dataset import (
#     load_alpaca_data, split_data, create_dataloaders, format_input
# )


# # ─────────────────────────────────────────────
# # MODEL CONFIG — must match GPT-2 medium weights exactly
# # ─────────────────────────────────────────────

# BASE_CONFIG = {
#     "vocab_size": 50257,
#     "context_length": 1024,
#     "drop_rate": 0.0,      # no dropout during fine-tuning
#     "qkv_bias": True
# }

# MODEL_CONFIGS = {
#     "gpt2-small (124M)":  {"emb_dim": 768,  "n_layers": 12, "n_heads": 12},
#     "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
#     "gpt2-large (774M)":  {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
#     "gpt2-xl (1558M)":    {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
# }

# CHOOSE_MODEL = "gpt2-small (124M)"


# # ─────────────────────────────────────────────
# # LOSS FUNCTIONS — same structure as gpt_train.py
# # but loss is computed across ALL tokens (not just last)
# # because we want the model to learn the full response
# # ─────────────────────────────────────────────

# def calc_loss_batch(input_batch, target_batch, model, device):
#     """
#     Cross entropy loss across all tokens.
#     The -100 tokens in target_batch are automatically ignored
#     by PyTorch's cross_entropy (ignore_index=-100 is the default).
#     This is the key difference from Chapter 6's classification loss
#     which only looked at the last token.
#     """
#     input_batch  = input_batch.to(device)
#     target_batch = target_batch.to(device)
#     logits = model(input_batch)
#     loss = torch.nn.functional.cross_entropy(
#         logits.flatten(0, 1),
#         target_batch.flatten()
#     )
#     return loss


# def calc_loss_loader(data_loader, model, device, num_batches=None):
#     total_loss = 0.0
#     if len(data_loader) == 0:
#         return float("nan")
#     num_batches = (
#         len(data_loader) if num_batches is None
#         else min(num_batches, len(data_loader))
#     )
#     for i, (input_batch, target_batch) in enumerate(data_loader):
#         if i < num_batches:
#             total_loss += calc_loss_batch(
#                 input_batch, target_batch, model, device
#             ).item()
#         else:
#             break
#     return total_loss / num_batches


# def evaluate_model(model, train_loader, val_loader, device, eval_iter):
#     model.eval()
#     with torch.no_grad():
#         train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
#         val_loss   = calc_loss_loader(val_loader,   model, device, num_batches=eval_iter)
#     model.train()
#     return train_loss, val_loss


# # ─────────────────────────────────────────────
# # GENERATE SAMPLE DURING TRAINING
# # prints a response to a validation prompt so you
# # can visually inspect quality each epoch
# # ─────────────────────────────────────────────

# def generate_and_print_sample(model, tokenizer, device, val_data):
#     model.eval()
#     input_text  = format_input(val_data[0])
#     token_ids   = text_to_token_ids(input_text, tokenizer).to(device)
#     context_size = model.pos_emb.weight.shape[0]

#     with torch.no_grad():
#         output_ids = generate_text_simple(
#             model=model,
#             idx=token_ids,
#             max_new_tokens=50,
#             context_size=context_size
#         )

#     generated_text  = token_ids_to_text(output_ids, tokenizer)
#     response        = generated_text[len(input_text):].strip()
#     print(f"\n[Sample response]\nInstruction: {val_data[0]['instruction']}")
#     print(f"Response:    {response}")
#     model.train()


# # ─────────────────────────────────────────────
# # TRAINING LOOP — follows same structure as
# # train_model_simple in gpt_train.py
# # ─────────────────────────────────────────────

# def train_model(model, train_loader, val_loader, optimizer,
#                 device, num_epochs, eval_freq, eval_iter,
#                 tokenizer, val_data):
#     train_losses, val_losses, tokens_seen = [], [], []
#     global_step = -1
#     total_tokens = 0

#     for epoch in range(num_epochs):
#         model.train()

#         for input_batch, target_batch in train_loader:

#             # ===== Added checks =====
#             if global_step == -1:
#                 print("Input batch device:", input_batch.device)
#                 print("Target batch device:", target_batch.device)
#                 print("Model device:", next(model.parameters()).device)
#                 print("GPU memory allocated:",
#                       torch.cuda.memory_allocated()/1024**2, "MB")
#                 print("GPU memory reserved:",
#                       torch.cuda.memory_reserved()/1024**2, "MB")
#             # ========================

#             optimizer.zero_grad()
#             loss = calc_loss_batch(input_batch, target_batch, model, device)
#             loss.backward()
#             optimizer.step()

#             total_tokens += input_batch.numel()
#             global_step  += 1

#             if global_step % eval_freq == 0:
#                 train_loss, val_loss = evaluate_model(
#                     model, train_loader, val_loader, device, eval_iter
#                 )
#                 train_losses.append(train_loss)
#                 val_losses.append(val_loss)
#                 tokens_seen.append(total_tokens)
#                 print(
#                     f"Ep {epoch+1} (Step {global_step:06d}): "
#                     f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}"
#                 )

#         # print a sample response after every epoch
#         generate_and_print_sample(model, tokenizer, device, val_data)

#     return train_losses, val_losses, tokens_seen


# # def train_model(model, train_loader, val_loader, optimizer,
# #                 device, num_epochs, eval_freq, eval_iter,
# #                 tokenizer, val_data):
# #     train_losses, val_losses, tokens_seen = [], [], []
# #     global_step = -1
# #     total_tokens = 0

# #     for epoch in range(num_epochs):
# #         model.train()

# #         for input_batch, target_batch in train_loader:
# #             optimizer.zero_grad()
# #             loss = calc_loss_batch(input_batch, target_batch, model, device)
# #             loss.backward()
# #             optimizer.step()

# #             total_tokens += input_batch.numel()
# #             global_step  += 1

# #             if global_step % eval_freq == 0:
# #                 train_loss, val_loss = evaluate_model(
# #                     model, train_loader, val_loader, device, eval_iter
# #                 )
# #                 train_losses.append(train_loss)
# #                 val_losses.append(val_loss)
# #                 tokens_seen.append(total_tokens)
# #                 print(
# #                     f"Ep {epoch+1} (Step {global_step:06d}): "
# #                     f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}"
# #                 )

# #         # print a sample response after every epoch
# #         generate_and_print_sample(model, tokenizer, device, val_data)

# #     return train_losses, val_losses, tokens_seen


# # ─────────────────────────────────────────────
# # PLOT LOSSES
# # ─────────────────────────────────────────────

# def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses,
#                 save_path="finetune_loss.pdf"):
#     fig, ax1 = plt.subplots()
#     ax1.plot(epochs_seen, train_losses, label="Training loss")
#     ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")
#     ax1.set_xlabel("Epochs")
#     ax1.set_ylabel("Loss")
#     ax1.legend(loc="upper right")
#     ax2 = ax1.twiny()
#     ax2.plot(tokens_seen, train_losses, alpha=0)
#     ax2.set_xlabel("Tokens seen")
#     fig.tight_layout()
#     plt.savefig(save_path)
#     print(f"Loss plot saved to {save_path}")


# # ─────────────────────────────────────────────
# # MAIN — run this file to start fine-tuning
# # ─────────────────────────────────────────────




# if __name__ == "__main__":

#     # ── device ──
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Device: {device}")

#     print("CUDA available:", torch.cuda.is_available())
#     print("Current device:", torch.cuda.current_device())
#     print("GPU:", torch.cuda.get_device_name(0))
#     print("Memory allocated:",
#         torch.cuda.memory_allocated()/1024**2, "MB")
#     print("Memory reserved:",
#         torch.cuda.memory_reserved()/1024**2, "MB")

#     # ── tokenizer ──
#     tokenizer = tiktoken.get_encoding("gpt2")

#     # ── data ──
#     data = load_alpaca_data()
#     train_data, val_data, test_data = split_data(data)
#     train_loader, val_loader, test_loader = create_dataloaders(
#         train_data, val_data, test_data,
#         tokenizer=tokenizer,
#         device=device,
#         batch_size=8    # reduce to 4 if you run out of GPU memory
#     )

#     # ── model config ──
#     cfg = {**BASE_CONFIG, **MODEL_CONFIGS[CHOOSE_MODEL]}
#     model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")

#     # ── load pretrained GPT-2 weights ──
#     print(f"\nLoading pretrained weights: {CHOOSE_MODEL}")
#     settings, params = download_and_load_gpt2(
#         model_size=model_size,
#         models_dir="/home/hamza/WORK-linux/LLM/pretraining_step4/gpt2"
#     )
#     model = GPTModel(cfg)
#     load_weights_into_gpt(model, params)
#     model.to(device)

#     # ===== Added checks =====
#     print("Model device:", next(model.parameters()).device)
#     print("Memory allocated after model.to(device):",
#           torch.cuda.memory_allocated()/1024**2, "MB")
#     print("Memory reserved after model.to(device):",
#           torch.cuda.memory_reserved()/1024**2, "MB")
#     # ========================

#     print("Pretrained weights loaded successfully.")

#     # ── optimizer ──
#     optimizer = torch.optim.AdamW(
#         model.parameters(),
#         lr=5e-5,
#         weight_decay=0.1
#     )

#     # ── training settings ──
#     NUM_EPOCHS = 2
#     EVAL_FREQ  = 200
#     EVAL_ITER  = 5

#     # ── train ──
#     print("\nStarting fine-tuning...")
#     train_losses, val_losses, tokens_seen = train_model(
#         model=model,
#         train_loader=train_loader,
#         val_loader=val_loader,
#         optimizer=optimizer,
#         device=device,
#         num_epochs=NUM_EPOCHS,
#         eval_freq=EVAL_FREQ,
#         eval_iter=EVAL_ITER,
#         tokenizer=tokenizer,
#         val_data=val_data
#     )

#     # ── save model ──
#     save_path = "gpt2-medium355M-sft.pth"
#     torch.save(model.state_dict(), save_path)
#     print(f"\nFine-tuned model saved to {save_path}")

#     # ── plot ──
#     epochs_tensor = torch.linspace(0, NUM_EPOCHS, len(train_losses))
#     plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)










# # if __name__ == "__main__":

# #     # ── device ──
# #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# #     print(f"Device: {device}")
    

# #     print("CUDA available:", torch.cuda.is_available())
# #     print("Current device:", torch.cuda.current_device())
# #     print("GPU:", torch.cuda.get_device_name(0))
# #     print("Memory allocated:",
# #         torch.cuda.memory_allocated()/1024**2, "MB")
# #     print("Memory reserved:",
# #         torch.cuda.memory_reserved()/1024**2, "MB")

# #     # ── tokenizer ──
# #     tokenizer = tiktoken.get_encoding("gpt2")

# #     # ── data ──
# #     data = load_alpaca_data()
# #     train_data, val_data, test_data = split_data(data)
# #     train_loader, val_loader, test_loader = create_dataloaders(
# #         train_data, val_data, test_data,
# #         tokenizer=tokenizer,
# #         device=device,
# #         batch_size=8    # reduce to 4 if you run out of GPU memory
# #     )

# #     # ── model config ──
# #     cfg = {**BASE_CONFIG, **MODEL_CONFIGS[CHOOSE_MODEL]}
# #     model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")

# #     # ── load pretrained GPT-2 weights ──
# #     # NOTE: downloads ~1.42 GB on first run
# #     print(f"\nLoading pretrained weights: {CHOOSE_MODEL}")
# #     settings, params = download_and_load_gpt2(
# #         model_size=model_size,
# #         models_dir="/home/hamza/WORK-linux/LLM/pretraining_step4/gpt2"
# #     )
# #     model = GPTModel(cfg)
# #     load_weights_into_gpt(model, params)
# #     model.to(device)
# #     print("Pretrained weights loaded successfully.")

# #     # ── optimizer ──
# #     # learning rate 5e-5 and weight_decay 0.1 are the values
# #     # used in Chapter 7 of the book — verify from your PDF
# #     optimizer = torch.optim.AdamW(
# #         model.parameters(),
# #         lr=5e-5,
# #         weight_decay=0.1
# #     )

# #     # ── training settings ──
# #     # NOTE: 2 epochs is a starting point for Alpaca's 52k entries.
# #     # The book uses similar settings on 1,100 entries. With 52k entries
# #     # even 1 epoch is significant. Adjust based on your GPU and time.
# #     # I am not certain of the ideal epoch count — monitor val_loss.
# #     NUM_EPOCHS = 2
# #     EVAL_FREQ  = 200   # evaluate every 200 steps
# #     EVAL_ITER  = 5     # use 5 batches for each eval estimate

# #     # ── train ──
# #     print("\nStarting fine-tuning...")
# #     train_losses, val_losses, tokens_seen = train_model(
# #         model=model,
# #         train_loader=train_loader,
# #         val_loader=val_loader,
# #         optimizer=optimizer,
# #         device=device,
# #         num_epochs=NUM_EPOCHS,
# #         eval_freq=EVAL_FREQ,
# #         eval_iter=EVAL_ITER,
# #         tokenizer=tokenizer,
# #         val_data=val_data
# #     )

# #     # ── save model ──
# #     # exact filename the book uses (confirmed from PDF)
# #     save_path = "gpt2-medium355M-sft.pth"
# #     torch.save(model.state_dict(), save_path)
# #     print(f"\nFine-tuned model saved to {save_path}")

# #     # ── plot ──
# #     epochs_tensor = torch.linspace(0, NUM_EPOCHS, len(train_losses))
# #     plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)





#-----------------------------------------------------------------------------------------
#the code below imports 2 datasets from internet and merges them with a custom dataset.
#but the format of data is more as a assistant 
#the double commented part in both above and below is the part where some changes were made to see cuda working and some stuff like that its simple version is commented and other is in use
#----------------------------------------------------------------------------------------









# # finetune_train.py
# # Chapter 7 — instruction fine-tuning training loop
# #
# # What this file does:
# #   1. Loads GPT-2 medium (355M) pretrained weights
# #   2. Sets up train/val/test dataloaders from Alpaca dataset
# #   3. Runs the fine-tuning training loop
# #   4. Saves the fine-tuned model as gpt2-medium355M-sft.pth
# #
# # Imports from YOUR existing files:
# #   gpt.py          → GPTModel, generate_text_simple
# #   gpt_download.py → download_and_load_gpt2
# #   gpt_generate.py → load_weights_into_gpt, text_to_token_ids, token_ids_to_text
# #
# # Imports from NEW files:
# #   instruction_dataset.py → load_alpaca_data, split_data, create_dataloaders,
# #                            format_input
# #
# # NOTE: The book recommends gpt2-medium (355M) over gpt2-small (124M) for
# #       instruction fine-tuning. The 124M model is described as lacking
# #       sufficient capacity for good results.
# # NOTE: The 355M model download is approximately 1.42 GB (confirmed from book PDF).
# #       Verify your available disk space before running.

# import sys
# import torch
# import tiktoken
# import matplotlib.pyplot as plt
# from pathlib import Path

# # ── add project root to sys.path so imports from sibling folders work ──
# project_root = Path(__file__).resolve().parents[1]
# if str(project_root) not in sys.path:
#     sys.path.insert(0, str(project_root))

# from llm_architecture_step3.gpt import GPTModel, generate_text_simple
# from pretraining_step4.gpt_download import download_and_load_gpt2
# from pretraining_step4.gpt_generate import load_weights_into_gpt, text_to_token_ids, token_ids_to_text
# # from fine_tunning_step5.instruction_dataset import (
# #     load_alpaca_data, split_data, create_dataloaders, format_input
# # )


# from fine_tunning_step5.instruction_dataset import (
#     load_and_merge_data, split_data, create_dataloaders, format_input
# )


# # ═══════════════════════════════════════════════════════
# # ADDED — import LoRA functions from lora.py
# # only these two functions are needed from lora.py
# # ═══════════════════════════════════════════════════════
# from fine_tunning_step5.lora import apply_lora, count_parameters


# # ─────────────────────────────────────────────
# # MODEL CONFIG — must match GPT-2 medium weights exactly
# # ─────────────────────────────────────────────

# BASE_CONFIG = {
#     "vocab_size": 50257,
#     "context_length": 1024,
#     "drop_rate": 0.0,      # no dropout during fine-tuning
#     "qkv_bias": True
# }

# MODEL_CONFIGS = {
#     "gpt2-small (124M)":  {"emb_dim": 768,  "n_layers": 12, "n_heads": 12},
#     "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
#     "gpt2-large (774M)":  {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
#     "gpt2-xl (1558M)":    {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
# }

# CHOOSE_MODEL = "gpt2-small (124M)"

# # ═══════════════════════════════════════════════════════
# # ADDED — LoRA rank setting
# # this is the only value you change to control LoRA size
# # paper used r=4 and r=8 — start with 4
# # set USE_LORA = False to run full fine-tuning instead
# # ═══════════════════════════════════════════════════════
# LORA_RANK = 4
# USE_LORA  = True


# # ─────────────────────────────────────────────
# # LOSS FUNCTIONS — same structure as gpt_train.py
# # but loss is computed across ALL tokens (not just last)
# # because we want the model to learn the full response
# # ─────────────────────────────────────────────

# def calc_loss_batch(input_batch, target_batch, model, device):
#     """
#     Cross entropy loss across all tokens.
#     The -100 tokens in target_batch are automatically ignored
#     by PyTorch's cross_entropy (ignore_index=-100 is the default).
#     This is the key difference from Chapter 6's classification loss
#     which only looked at the last token.
#     """
#     input_batch  = input_batch.to(device)
#     target_batch = target_batch.to(device)
#     logits = model(input_batch)
#     loss = torch.nn.functional.cross_entropy(
#         logits.flatten(0, 1),
#         target_batch.flatten()
#     )
#     return loss


# def calc_loss_loader(data_loader, model, device, num_batches=None):
#     total_loss = 0.0
#     if len(data_loader) == 0:
#         return float("nan")
#     num_batches = (
#         len(data_loader) if num_batches is None
#         else min(num_batches, len(data_loader))
#     )
#     for i, (input_batch, target_batch) in enumerate(data_loader):
#         if i < num_batches:
#             total_loss += calc_loss_batch(
#                 input_batch, target_batch, model, device
#             ).item()
#         else:
#             break
#     return total_loss / num_batches


# def evaluate_model(model, train_loader, val_loader, device, eval_iter):
#     model.eval()
#     with torch.no_grad():
#         train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
#         val_loss   = calc_loss_loader(val_loader,   model, device, num_batches=eval_iter)
#     model.train()
#     return train_loss, val_loss


# # ─────────────────────────────────────────────
# # GENERATE SAMPLE DURING TRAINING
# # prints a response to a validation prompt so you
# # can visually inspect quality each epoch
# # ─────────────────────────────────────────────

# def generate_and_print_sample(model, tokenizer, device, val_data):
#     model.eval()
#     input_text  = format_input(val_data[0])
#     token_ids   = text_to_token_ids(input_text, tokenizer).to(device)
#     context_size = model.pos_emb.weight.shape[0]

#     with torch.no_grad():
#         output_ids = generate_text_simple(
#             model=model,
#             idx=token_ids,
#             max_new_tokens=50,
#             context_size=context_size
#         )

#     generated_text  = token_ids_to_text(output_ids, tokenizer)
#     response        = generated_text[len(input_text):].strip()
#     print(f"\n[Sample response]\nInstruction: {val_data[0]['instruction']}")
#     print(f"Response:    {response}")
#     model.train()


# # ─────────────────────────────────────────────
# # TRAINING LOOP — follows same structure as
# # train_model_simple in gpt_train.py
# # ─────────────────────────────────────────────

# def train_model(model, train_loader, val_loader, optimizer,
#                 device, num_epochs, eval_freq, eval_iter,
#                 tokenizer, val_data):
#     train_losses, val_losses, tokens_seen = [], [], []
#     global_step = -1
#     total_tokens = 0

#     for epoch in range(num_epochs):
#         model.train()

#         for input_batch, target_batch in train_loader:

#             # ===== Added checks =====
#             if global_step == -1:
#                 print("Input batch device:", input_batch.device)
#                 print("Target batch device:", target_batch.device)
#                 print("Model device:", next(model.parameters()).device)
#                 print("GPU memory allocated:",
#                       torch.cuda.memory_allocated()/1024**2, "MB")
#                 print("GPU memory reserved:",
#                       torch.cuda.memory_reserved()/1024**2, "MB")
#             # ========================

#             optimizer.zero_grad()
#             loss = calc_loss_batch(input_batch, target_batch, model, device)
#             loss.backward()
#             optimizer.step()

#             total_tokens += input_batch.numel()
#             global_step  += 1

#             if global_step % eval_freq == 0:
#                 train_loss, val_loss = evaluate_model(
#                     model, train_loader, val_loader, device, eval_iter
#                 )
#                 train_losses.append(train_loss)
#                 val_losses.append(val_loss)
#                 tokens_seen.append(total_tokens)
#                 print(
#                     f"Ep {epoch+1} (Step {global_step:06d}): "
#                     f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}"
#                 )

#         # print a sample response after every epoch
#         generate_and_print_sample(model, tokenizer, device, val_data)

#     return train_losses, val_losses, tokens_seen


# # def train_model(model, train_loader, val_loader, optimizer,
# #                 device, num_epochs, eval_freq, eval_iter,
# #                 tokenizer, val_data):
# #     train_losses, val_losses, tokens_seen = [], [], []
# #     global_step = -1
# #     total_tokens = 0

# #     for epoch in range(num_epochs):
# #         model.train()

# #         for input_batch, target_batch in train_loader:
# #             optimizer.zero_grad()
# #             loss = calc_loss_batch(input_batch, target_batch, model, device)
# #             loss.backward()
# #             optimizer.step()

# #             total_tokens += input_batch.numel()
# #             global_step  += 1

# #             if global_step % eval_freq == 0:
# #                 train_loss, val_loss = evaluate_model(
# #                     model, train_loader, val_loader, device, eval_iter
# #                 )
# #                 train_losses.append(train_loss)
# #                 val_losses.append(val_loss)
# #                 tokens_seen.append(total_tokens)
# #                 print(
# #                     f"Ep {epoch+1} (Step {global_step:06d}): "
# #                     f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}"
# #                 )

# #         # print a sample response after every epoch
# #         generate_and_print_sample(model, tokenizer, device, val_data)

# #     return train_losses, val_losses, tokens_seen


# # ─────────────────────────────────────────────
# # PLOT LOSSES
# # ─────────────────────────────────────────────

# def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses,
#                 save_path="finetune_loss.pdf"):
#     fig, ax1 = plt.subplots()
#     ax1.plot(epochs_seen, train_losses, label="Training loss")
#     ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")
#     ax1.set_xlabel("Epochs")
#     ax1.set_ylabel("Loss")
#     ax1.legend(loc="upper right")
#     ax2 = ax1.twiny()
#     ax2.plot(tokens_seen, train_losses, alpha=0)
#     ax2.set_xlabel("Tokens seen")
#     fig.tight_layout()
#     plt.savefig(save_path)
#     print(f"Loss plot saved to {save_path}")


# # ─────────────────────────────────────────────
# # MAIN — run this file to start fine-tuning
# # ─────────────────────────────────────────────

# if __name__ == "__main__":

#     # ── device ──
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Device: {device}")

#     print("CUDA available:", torch.cuda.is_available())
#     print("Current device:", torch.cuda.current_device())
#     print("GPU:", torch.cuda.get_device_name(0))
#     print("Memory allocated:",
#         torch.cuda.memory_allocated()/1024**2, "MB")
#     print("Memory reserved:",
#         torch.cuda.memory_reserved()/1024**2, "MB")

#     # ── tokenizer ──
#     tokenizer = tiktoken.get_encoding("gpt2")

    

#     data = load_and_merge_data(
#     use_oasst1=True,
#     use_alpaca=True,
#     custom_data_path=Path(__file__).resolve().parent / "my_custom_data.json",
#     personal_repeat=50)
#     train_data, val_data, test_data = split_data(data)

    
#     train_loader, val_loader, test_loader = create_dataloaders(
#         train_data, val_data, test_data,
#         tokenizer=tokenizer,
#         device=device,
#         batch_size=2    # reduce to 4 if you run out of GPU memory
#     )

#     # ── model config ──
#     cfg = {**BASE_CONFIG, **MODEL_CONFIGS[CHOOSE_MODEL]}
#     model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")

#     # ── load pretrained GPT-2 weights ──
#     print(f"\nLoading pretrained weights: {CHOOSE_MODEL}")
#     settings, params = download_and_load_gpt2(
#         model_size=model_size,
#         models_dir="/home/hamza/WORK-linux/LLM/pretraining_step4/gpt2"
#     )
#     model = GPTModel(cfg)
#     load_weights_into_gpt(model, params)
    
# # to finetune pretrained model
#     # model = GPTModel(cfg)
#     # apply_lora(model, rank=LORA_RANK)   # inject LoRA structure first
#     # model.load_state_dict(
#     #     torch.load("gpt2-small124M-lora-r4-sft.pth",
#     #             map_location=device, weights_only=True)
#     # )

#     model.to(device)

#     # ===== Added checks =====
#     print("Model device:", next(model.parameters()).device)
#     print("Memory allocated after model.to(device):",
#           torch.cuda.memory_allocated()/1024**2, "MB")
#     print("Memory reserved after model.to(device):",
#           torch.cuda.memory_reserved()/1024**2, "MB")
#     # ========================

#     print("Pretrained weights loaded successfully.")

#     # ═══════════════════════════════════════════════════════
#     # ADDED — apply LoRA if USE_LORA is True
#     # this is the only block added inside __main__
#     # it sits right after weights are loaded and before
#     # the optimizer is created — order matters because
#     # apply_lora() freezes weights and the optimizer must
#     # only see the trainable LoRA params after that
#     #if a pretrained model is loaded, you can skip this step and just uncomment the model loading lines above
#     # ═══════════════════════════════════════════════════════
#     if USE_LORA:
#         print(f"\nApplying LoRA (rank={LORA_RANK})...")
#         apply_lora(model, rank=LORA_RANK)
#         print("Parameter count after LoRA:")
#         count_parameters(model)

#     # ── optimizer ──
#     # ═══════════════════════════════════════════════════════
#     # CHANGED — when USE_LORA is True, optimizer only receives
#     # parameters where requires_grad=True (the LoRA A and B matrices)
#     # when USE_LORA is False, model.parameters() passes everything
#     # as before — no other change to the optimizer block
#     # ═══════════════════════════════════════════════════════
#     optimizer = torch.optim.AdamW(
#         filter(lambda p: p.requires_grad, model.parameters()),
#         lr=5e-5,
#         weight_decay=0.1
#     )

#     # ── training settings ──
#     NUM_EPOCHS = 2
#     EVAL_FREQ  = 200
#     EVAL_ITER  = 5

#     # ── train ──
#     print("\nStarting fine-tuning...")
#     train_losses, val_losses, tokens_seen = train_model(
#         model=model,
#         train_loader=train_loader,
#         val_loader=val_loader,
#         optimizer=optimizer,
#         device=device,
#         num_epochs=NUM_EPOCHS,
#         eval_freq=EVAL_FREQ,
#         eval_iter=EVAL_ITER,
#         tokenizer=tokenizer,
#         val_data=val_data
#     )

#     # ── save model ──
#     # ═══════════════════════════════════════════════════════
#     # CHANGED — save filename reflects whether LoRA was used
#     # so you can keep both versions on disk separately
#     # ═══════════════════════════════════════════════════════
#     save_path = (
#         f"gpt2-small124M-lora-r{LORA_RANK}-sft.pth"
#         if USE_LORA else
#         "gpt2-medium355M-sft.pth"
#     )
#     torch.save(model.state_dict(), save_path)
#     print(f"\nFine-tuned model saved to {save_path}")

#     # ── plot ──
#     epochs_tensor = torch.linspace(0, NUM_EPOCHS, len(train_losses))
#     plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)


# # if __name__ == "__main__":

# #     # ── device ──
# #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# #     print(f"Device: {device}")
    

# #     print("CUDA available:", torch.cuda.is_available())
# #     print("Current device:", torch.cuda.current_device())
# #     print("GPU:", torch.cuda.get_device_name(0))
# #     print("Memory allocated:",
# #         torch.cuda.memory_allocated()/1024**2, "MB")
# #     print("Memory reserved:",
# #         torch.cuda.memory_reserved()/1024**2, "MB")

# #     # ── tokenizer ──
# #     tokenizer = tiktoken.get_encoding("gpt2")

# #     # ── data ──
# #     data = load_alpaca_data()
# #     train_data, val_data, test_data = split_data(data)
# #     train_loader, val_loader, test_loader = create_dataloaders(
# #         train_data, val_data, test_data,
# #         tokenizer=tokenizer,
# #         device=device,
# #         batch_size=8    # reduce to 4 if you run out of GPU memory
# #     )

# #     # ── model config ──
# #     cfg = {**BASE_CONFIG, **MODEL_CONFIGS[CHOOSE_MODEL]}
# #     model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")

# #     # ── load pretrained GPT-2 weights ──
# #     # NOTE: downloads ~1.42 GB on first run
# #     print(f"\nLoading pretrained weights: {CHOOSE_MODEL}")
# #     settings, params = download_and_load_gpt2(
# #         model_size=model_size,
# #         models_dir="/home/hamza/WORK-linux/LLM/pretraining_step4/gpt2"
# #     )
# #     model = GPTModel(cfg)
# #     load_weights_into_gpt(model, params)
# #     model.to(device)
# #     print("Pretrained weights loaded successfully.")

# #     # ── optimizer ──
# #     # learning rate 5e-5 and weight_decay 0.1 are the values
# #     # used in Chapter 7 of the book — verify from your PDF
# #     optimizer = torch.optim.AdamW(
# #         model.parameters(),
# #         lr=5e-5,
# #         weight_decay=0.1
# #     )

# #     # ── training settings ──
# #     # NOTE: 2 epochs is a starting point for Alpaca's 52k entries.
# #     # The book uses similar settings on 1,100 entries. With 52k entries
# #     # even 1 epoch is significant. Adjust based on your GPU and time.
# #     # I am not certain of the ideal epoch count — monitor val_loss.
# #     NUM_EPOCHS = 2
# #     EVAL_FREQ  = 200   # evaluate every 200 steps
# #     EVAL_ITER  = 5     # use 5 batches for each eval estimate

# #     # ── train ──
# #     print("\nStarting fine-tuning...")
# #     train_losses, val_losses, tokens_seen = train_model(
# #         model=model,
# #         train_loader=train_loader,
# #         val_loader=val_loader,
# #         optimizer=optimizer,
# #         device=device,
# #         num_epochs=NUM_EPOCHS,
# #         eval_freq=EVAL_FREQ,
# #         eval_iter=EVAL_ITER,
# #         tokenizer=tokenizer,
# #         val_data=val_data
# #     )

# #     # ── save model ──
# #     # exact filename the book uses (confirmed from PDF)
# #     save_path = "gpt2-medium355M-sft.pth"
# #     torch.save(model.state_dict(), save_path)
# #     print(f"\nFine-tuned model saved to {save_path}")

# #     # ── plot ──
# #     epochs_tensor = torch.linspace(0, NUM_EPOCHS, len(train_losses))
# #     plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)














#-----------------------------------------------------------------------------------------
#the code below imports 2 datasets from internet and merges them with a custom dataset.
#but the format of data is more as a assistant 
#-----------------------------------------------------------------------------------------







# finetune_train.py
# Chapter 7 — instruction fine-tuning training loop
#
# What this file does:
#   1. Loads GPT-2 medium (355M) pretrained weights
#   2. Sets up train/val/test dataloaders from Alpaca dataset
#   3. Runs the fine-tuning training loop
#   4. Saves the fine-tuned model as gpt2-medium355M-sft.pth
#
# Imports from YOUR existing files:
#   gpt.py          → GPTModel, generate_text_simple
#   gpt_download.py → download_and_load_gpt2
#   gpt_generate.py → load_weights_into_gpt, text_to_token_ids, token_ids_to_text
#
# Imports from NEW files:
#   instruction_dataset.py → load_alpaca_data, split_data, create_dataloaders,
#                            format_input
#
# NOTE: The book recommends gpt2-medium (355M) over gpt2-small (124M) for
#       instruction fine-tuning. The 124M model is described as lacking
#       sufficient capacity for good results.
# NOTE: The 355M model download is approximately 1.42 GB (confirmed from book PDF).
#       Verify your available disk space before running.

import sys
import torch
import tiktoken
import matplotlib.pyplot as plt
from pathlib import Path

# ── add project root to sys.path so imports from sibling folders work ──
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from llm_architecture_step3.gpt import GPTModel, generate_text_simple
from pretraining_step4.gpt_download import download_and_load_gpt2
from pretraining_step4.gpt_generate import load_weights_into_gpt, text_to_token_ids, token_ids_to_text
# from fine_tunning_step5.instruction_dataset import (
#     load_alpaca_data, split_data, create_dataloaders, format_input
# )

# ═══════════════════════════════════════════════════════
# CHANGED — updated import to match new instruction_dataset.py
# use_everyday and use_ultrachat replaced use_oasst1 and use_alpaca
# format_input signature also changed — now takes user_message string
# ═══════════════════════════════════════════════════════
from fine_tunning_step5.instruction_dataset import (
    load_and_merge_data, split_data, create_dataloaders, format_input
)

# ═══════════════════════════════════════════════════════
# ADDED — import LoRA functions from lora.py
# only these two functions are needed from lora.py
# ═══════════════════════════════════════════════════════
from fine_tunning_step5.lora import apply_lora, count_parameters


# ─────────────────────────────────────────────
# MODEL CONFIG — must match GPT-2 medium weights exactly
# ─────────────────────────────────────────────

BASE_CONFIG = {
    "vocab_size": 50257,
    "context_length": 1024,
    "drop_rate": 0.0,      # no dropout during fine-tuning
    "qkv_bias": True
}

MODEL_CONFIGS = {
    "gpt2-small (124M)":  {"emb_dim": 768,  "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)":  {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)":    {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}

CHOOSE_MODEL = "gpt2-small (124M)"

# ═══════════════════════════════════════════════════════
# ADDED — LoRA rank setting
# this is the only value you change to control LoRA size
# paper used r=4 and r=8 — start with 4
# set USE_LORA = False to run full fine-tuning instead
# ═══════════════════════════════════════════════════════
LORA_RANK = 4
USE_LORA  = True


# ─────────────────────────────────────────────
# LOSS FUNCTIONS — same structure as gpt_train.py
# but loss is computed across ALL tokens (not just last)
# because we want the model to learn the full response
# ─────────────────────────────────────────────

def calc_loss_batch(input_batch, target_batch, model, device):
    """
    Cross entropy loss across all tokens.
    The -100 tokens in target_batch are automatically ignored
    by PyTorch's cross_entropy (ignore_index=-100 is the default).
    This is the key difference from Chapter 6's classification loss
    which only looked at the last token.
    """
    input_batch  = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1),
        target_batch.flatten()
    )
    return loss


def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.0
    if len(data_loader) == 0:
        return float("nan")
    num_batches = (
        len(data_loader) if num_batches is None
        else min(num_batches, len(data_loader))
    )
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            total_loss += calc_loss_batch(
                input_batch, target_batch, model, device
            ).item()
        else:
            break
    return total_loss / num_batches


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss   = calc_loss_loader(val_loader,   model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


# ─────────────────────────────────────────────
# GENERATE SAMPLE DURING TRAINING
# prints a response to a validation prompt so you
# can visually inspect quality each epoch
# ─────────────────────────────────────────────

def generate_and_print_sample(model, tokenizer, device, val_data):
    model.eval()
    # ═══════════════════════════════════════════════════════
    # CHANGED — data now has 'user' key instead of 'instruction'
    # format_input now takes user_message string directly
    # ═══════════════════════════════════════════════════════
    input_text   = format_input(val_data[0]["user"])
    token_ids    = text_to_token_ids(input_text, tokenizer).to(device)
    context_size = model.pos_emb.weight.shape[0]

    with torch.no_grad():
        output_ids = generate_text_simple(
            model=model,
            idx=token_ids,
            max_new_tokens=50,
            context_size=context_size
        )

    generated_text = token_ids_to_text(output_ids, tokenizer)
    response       = generated_text[len(input_text):].strip()
    # ═══════════════════════════════════════════════════════
    # CHANGED — display label matches new User/Azuha format
    # ═══════════════════════════════════════════════════════
    print(f"\n[Sample response]\nUser : {val_data[0]['user']}")
    print(f"Azuha: {response}")
    model.train()


# ─────────────────────────────────────────────
# TRAINING LOOP — follows same structure as
# train_model_simple in gpt_train.py
# ─────────────────────────────────────────────

def train_model(model, train_loader, val_loader, optimizer,
                device, num_epochs, eval_freq, eval_iter,
                tokenizer, val_data):
    train_losses, val_losses, tokens_seen = [], [], []
    global_step = -1
    total_tokens = 0

    for epoch in range(num_epochs):
        model.train()

        for input_batch, target_batch in train_loader:

            # ===== Added checks =====
            if global_step == -1:
                print("Input batch device:", input_batch.device)
                print("Target batch device:", target_batch.device)
                print("Model device:", next(model.parameters()).device)
                print("GPU memory allocated:",
                      torch.cuda.memory_allocated()/1024**2, "MB")
                print("GPU memory reserved:",
                      torch.cuda.memory_reserved()/1024**2, "MB")
            # ========================

            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()

            total_tokens += input_batch.numel()
            global_step  += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                tokens_seen.append(total_tokens)
                print(
                    f"Ep {epoch+1} (Step {global_step:06d}): "
                    f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}"
                )

        # print a sample response after every epoch
        generate_and_print_sample(model, tokenizer, device, val_data)

    return train_losses, val_losses, tokens_seen


# ─────────────────────────────────────────────
# PLOT LOSSES
# ─────────────────────────────────────────────

def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses,
                save_path="finetune_loss.pdf"):
    fig, ax1 = plt.subplots()
    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    ax2 = ax1.twiny()
    ax2.plot(tokens_seen, train_losses, alpha=0)
    ax2.set_xlabel("Tokens seen")
    fig.tight_layout()
    plt.savefig(save_path)
    print(f"Loss plot saved to {save_path}")


# ─────────────────────────────────────────────
# MAIN — run this file to start fine-tuning
# ─────────────────────────────────────────────

if __name__ == "__main__":

    # ── device ──
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("CUDA available:", torch.cuda.is_available())
    print("Current device:", torch.cuda.current_device())
    print("GPU:", torch.cuda.get_device_name(0))
    print("Memory allocated:",
        torch.cuda.memory_allocated()/1024**2, "MB")
    print("Memory reserved:",
        torch.cuda.memory_reserved()/1024**2, "MB")

    # ── tokenizer ──
    tokenizer = tiktoken.get_encoding("gpt2")

    # ═══════════════════════════════════════════════════════
    # CHANGED — data loading block updated to new parameters
    # use_everyday and use_ultrachat replace use_oasst1 and use_alpaca
    # custom_data_path points to hamza_azuha_dataset.json
    # personal_repeat kept at 50 as you had it
    # ═══════════════════════════════════════════════════════
    data = load_and_merge_data(
        use_everyday=True,
        use_ultrachat=True,
        ultrachat_samples=20000,
        custom_data_path=Path(__file__).resolve().parent / "hamza_azuha_dataset.json",
        personal_repeat=50
    )
    train_data, val_data, test_data = split_data(data)

    train_loader, val_loader, test_loader = create_dataloaders(
        train_data, val_data, test_data,
        tokenizer=tokenizer,
        device=device,
        batch_size=2    # reduce to 4 if you run out of GPU memory
    )

    # ── model config ──
    cfg = {**BASE_CONFIG, **MODEL_CONFIGS[CHOOSE_MODEL]}
    model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")

    # ── load pretrained GPT-2 weights ──
    print(f"\nLoading pretrained weights: {CHOOSE_MODEL}")
    settings, params = download_and_load_gpt2(
        model_size=model_size,
        models_dir="/home/hamza/WORK-linux/LLM/pretraining_step4/gpt2"
    )
    model = GPTModel(cfg)
    load_weights_into_gpt(model, params)

    # to finetune pretrained model
    # model = GPTModel(cfg)
    # apply_lora(model, rank=LORA_RANK)   # inject LoRA structure first
    # model.load_state_dict(
    #     torch.load("gpt2-small124M-lora-r4-sft.pth",
    #             map_location=device, weights_only=True)
    # )

    model.to(device)

    # ===== Added checks =====
    print("Model device:", next(model.parameters()).device)
    print("Memory allocated after model.to(device):",
          torch.cuda.memory_allocated()/1024**2, "MB")
    print("Memory reserved after model.to(device):",
          torch.cuda.memory_reserved()/1024**2, "MB")
    # ========================

    print("Pretrained weights loaded successfully.")

    # ═══════════════════════════════════════════════════════
    # ADDED — apply LoRA if USE_LORA is True
    # this is the only block added inside __main__
    # it sits right after weights are loaded and before
    # the optimizer is created — order matters because
    # apply_lora() freezes weights and the optimizer must
    # only see the trainable LoRA params after that
    # if a pretrained model is loaded, you can skip this step
    # and just uncomment the model loading lines above
    # ═══════════════════════════════════════════════════════
    if USE_LORA:
        print(f"\nApplying LoRA (rank={LORA_RANK})...")
        apply_lora(model, rank=LORA_RANK)
        print("Parameter count after LoRA:")
        count_parameters(model)

    # ── optimizer ──
    # ═══════════════════════════════════════════════════════
    # CHANGED — when USE_LORA is True, optimizer only receives
    # parameters where requires_grad=True (the LoRA A and B matrices)
    # when USE_LORA is False, model.parameters() passes everything
    # as before — no other change to the optimizer block
    # ═══════════════════════════════════════════════════════
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=5e-5,
        weight_decay=0.1
    )

    # ── training settings ──
    NUM_EPOCHS = 2
    EVAL_FREQ  = 200
    EVAL_ITER  = 5

    # ── train ──
    print("\nStarting fine-tuning...")
    train_losses, val_losses, tokens_seen = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        num_epochs=NUM_EPOCHS,
        eval_freq=EVAL_FREQ,
        eval_iter=EVAL_ITER,
        tokenizer=tokenizer,
        val_data=val_data
    )

    # ── save model ──
    # ═══════════════════════════════════════════════════════
    # CHANGED — save filename reflects whether LoRA was used
    # so you can keep both versions on disk separately
    # ═══════════════════════════════════════════════════════
    save_path = (
        f"gpt2-small124M-lora-r{LORA_RANK}-sft.pth"
        if USE_LORA else
        "gpt2-medium355M-sft.pth"
    )
    torch.save(model.state_dict(), save_path)
    print(f"\nFine-tuned model saved to {save_path}")

    # ── plot ──
    epochs_tensor = torch.linspace(0, NUM_EPOCHS, len(train_losses))
    plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)



