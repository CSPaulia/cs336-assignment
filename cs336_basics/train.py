import os
import logging

import torch
import wandb

from cs336_basics.tokenizer import train_bpe_from_file, BPETokenizer, save_bpe, load_bpe
from cs336_basics.linear import Linear
from cs336_basics.embedding import Embedding
from cs336_basics.normalization import RMSNorm
from cs336_basics.activation import SwiGLU, Softmax
from cs336_basics.position_embedding import RotaryPositionalEmbedding
from cs336_basics.attention import ScaledDotProductAttention, CausalMultiHeadSelfAttention
from cs336_basics.transformer import TransformerBlock, Transformer
from cs336_basics.loss import cross_entropy
from cs336_basics.optimizer import AdamW
from cs336_basics.schedule import cosine_lr_schedule
from cs336_basics.grad_clip import gradient_clipping
from cs336_basics.dataloader import dataloading
from cs336_basics.checkpoint import save_checkpoint, load_checkpoint
from cs336_basics.generate import generate_text

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("logs/training.log"),
            logging.StreamHandler()
        ]
    )

def train(
        model,
        optimizer,
        token_ids,
        iterations: int = 2000,
        batch_size: int = 64,
        context_length: int = 256,
        learning_rate: float = 1e-3,
        eval_every: int = 100,
        val_token_ids: torch.Tensor = None,
    ):
    for it in range(iterations):
        lr = cosine_lr_schedule(
            it,
            lr_max=learning_rate,
            lr_min=learning_rate * 0.1,
            t_warmup=100,
            t_cosine=iterations,
        )
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        x, y = dataloading(
            token_ids,
            batch_size=batch_size,
            context_length=context_length,
            device=model.device
        )

        logits = model(x)
        loss = cross_entropy(logits.view(-1, logits.shape[-1]), y.view(-1))

        optimizer.zero_grad()
        loss.backward()

        gradient_clipping(model.parameters(), max_norm=1.0)
        optimizer.step()

        wandb.log({"train/loss": loss.item(), "train/lr": lr, "train/iteration": it})

        if (it + 1) % eval_every == 0:
            logging.info(f"Iteration {it + 1}/{iterations}, Loss: {loss.item():.4f}, LR: {lr:.6f}")
            if val_token_ids is not None:
                val_loss = evaluate(model, val_token_ids, batch_size=batch_size, context_length=context_length)
                wandb.log({"val/loss": val_loss, "val/iteration": it + 1})
                model.train()

        if (it + 1) % 500 == 0:
            logging.info(f"Saving checkpoint at iteration {it + 1}")
            os.makedirs("cs336_basics/ckpts", exist_ok=True)
            save_checkpoint(model, optimizer, it + 1, f"cs336_basics/ckpts/checkpoint_{it + 1}.pt")

def evaluate(
        model, 
        token_ids, 
        batch_size=64, 
        context_length=256
    ):
    model.eval()
    total_loss = 0.0
    count = 0

    with torch.no_grad():
        for start in range(0, len(token_ids) - context_length, batch_size):
            input_sequences = torch.stack([token_ids[i:i+context_length] for i in range(start, min(start + batch_size, len(token_ids) - context_length))])
            target_sequences = torch.stack([token_ids[i+1:i+context_length+1] for i in range(start, min(start + batch_size, len(token_ids) - context_length))])

            logits = model(input_sequences)
            loss = cross_entropy(logits.view(-1, logits.shape[-1]), target_sequences.view(-1))
            total_loss += loss.item() * input_sequences.size(0)
            count += input_sequences.size(0)

    avg_loss = total_loss / count if count > 0 else 0
    logging.info(f"Validation Loss: {avg_loss:.4f}")
    return avg_loss

def generate(
        model,
        tokenizer,
        prompt: str, 
        max_length: int=100):
    generated_text = generate_text(model, tokenizer, prompt, max_length=max_length)
    logging.info(f"Generated text: {generated_text}")

def main(
        train_file_path: str = "data/TinyStoriesV2-GPT4-train.txt",
        val_file_path: str = "data/TinyStoriesV2-GPT4-valid.txt",
    ):
    setup_logging()

    file_name = os.path.basename(train_file_path)
    if not os.path.exists(f"data/tokenizer/{file_name}-vocab.pkl") or not os.path.exists(f"data/tokenizer/{file_name}-merges.pkl"):
        vocab, merges = train_bpe_from_file(
            train_file_path, 
            vocab_size=10000, 
            special_tokens=["<|endoftext|>"]
        )
        save_bpe(vocab, merges, f"data/tokenizer/{file_name}-vocab.pkl", f"data/tokenizer/{file_name}-merges.pkl")
    else:
        vocab, merges = load_bpe(f"data/tokenizer/{file_name}-vocab.pkl", f"data/tokenizer/{file_name}-merges.pkl")
    
    logging.info(f"Tokenizer trained/loaded with vocab size: {len(vocab)} and merges size: {len(merges)}")
    tokenizer = BPETokenizer(vocab, merges, special_tokens=["<|endoftext|>"])

    logging.info(f"Loading training dataset from {train_file_path}")
    with open(train_file_path, "r") as f:
        train_dataset = f.read()

    if not os.path.exists(f"data/tokenizer/{file_name}-train_token_ids.pt"):
        train_token_ids = tokenizer.encode(train_dataset)
        with open(f"data/tokenizer/{file_name}-train_token_ids.pt", "wb") as f:
            torch.save(train_token_ids, f)
    else:
        with open(f"data/tokenizer/{file_name}-train_token_ids.pt", "rb") as f:
            train_token_ids = torch.load(f)
    train_token_ids = torch.tensor(train_token_ids, dtype=torch.long)

    logging.info(f"Loading validation dataset from {val_file_path}")
    with open(val_file_path, "r") as f:
        val_dataset = f.read()

    if not os.path.exists(f"data/tokenizer/{file_name}-val_token_ids.pt"):
        val_token_ids = tokenizer.encode(val_dataset)
        with open(f"data/tokenizer/{file_name}-val_token_ids.pt", "wb") as f:
            torch.save(val_token_ids, f)
    else:
        with open(f"data/tokenizer/{file_name}-val_token_ids.pt", "rb") as f:
            val_token_ids = torch.load(f)
    val_token_ids = torch.tensor(val_token_ids, dtype=torch.long)

    logging.info(f"Model initialization")
    model = Transformer(
        vocab_size=10000,
        context_length=256,
        num_layers=4,
        num_heads=16,
        d_model=512,
        d_ff=1344,
        rope_theta=10000.0,
        device="cuda" if torch.cuda.is_available() else "cpu",
        dtype=torch.float32
    )

    logging.info(f"Optimizer and training setup")
    lr = 1e-3
    batch_size = 64
    optimizer = AdamW(
        model.parameters(),
        lr=lr,
        betas=(0.9, 0.95),
        weight_decay=0.01
    )

    wandb.init(
        project="cs336-assignment1",
        name=f"lr-{lr}-bs-{batch_size}-owt",
        config={
            "vocab_size": 10000,
            "context_length": 256,
            "num_layers": 4,
            "num_heads": 16,
            "d_model": 512,
            "d_ff": 1344,
            "batch_size": batch_size,
            "learning_rate": lr,
            "iterations": 2000,
            "optimizer": "AdamW",
            "weight_decay": 0.01,
            "betas": (0.9, 0.95),
        }
    )

    train(
        model=model,
        optimizer=optimizer,
        token_ids=train_token_ids,
        iterations=20000,
        batch_size=batch_size,
        context_length=256,
        learning_rate=lr,
        eval_every=10000,
        val_token_ids=val_token_ids,
    )

    final_val_loss = evaluate(
        model=model,
        token_ids=val_token_ids,
        batch_size=batch_size,
        context_length=256
    )
    wandb.log({"val/final_loss": final_val_loss})

    generate(
        model=model,
        tokenizer=tokenizer,
        prompt="Once upon a time",
        max_length=100
    )

    wandb.finish()

if __name__ == "__main__":
    main(
        train_file_path="data/owt_train.txt",
        val_file_path="data/owt_val.txt"
    )

