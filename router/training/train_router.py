from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="train a one logit router on (query, draft, y) rows")
    parser.add_argument("--pairs", type=Path, required=True, help="jsonl with query, draft, y")
    parser.add_argument("--base", default="microsoft/deberta-v3-xsmall")
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-input-tokens", type=int, default=4096)
    args = parser.parse_args()

    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    rows = [json.loads(line) for line in args.pairs.read_text(encoding="utf-8").splitlines() if line.strip()]
    tokenizer = AutoTokenizer.from_pretrained(args.base)
    model = AutoModelForSequenceClassification.from_pretrained(args.base, num_labels=1)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    def batches():
        for start in range(0, len(rows), 16):
            chunk = rows[start : start + 16]
            encoded = tokenizer(
                [r["query"] for r in chunk],
                [r["draft"] for r in chunk],
                truncation=True,
                max_length=args.max_input_tokens,
                padding=True,
                return_tensors="pt",
            )
            labels = torch.tensor([float(r["y"]) for r in chunk])
            yield {k: v.to(device) for k, v in encoded.items()}, labels.to(device)

    model.train()
    for epoch in range(args.epochs):
        total = 0.0
        for encoded, labels in DataLoader(list(batches()), batch_size=None):
            optimizer.zero_grad()
            logits = model(**encoded).logits.reshape(-1)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            total += float(loss)
        print(f"epoch {epoch + 1}: loss {total:.4f}")
    model.save_pretrained(args.out, safe_serialization=True)
    tokenizer.save_pretrained(args.out)
    print(f"saved to {args.out}")


if __name__ == "__main__":
    main()
