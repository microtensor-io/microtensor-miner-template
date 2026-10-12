from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="fine tune the base model on your SQL data")
    parser.add_argument("--base", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    parser.add_argument("--data", required=True, help="jsonl with prompt and completion")
    parser.add_argument("--out", required=True)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-5)
    args = parser.parse_args()
    print(
        "fill in your trainer here (transformers Trainer or TRL SFT); record every run under "
        f"training/runs/<id>/ with config.json, metrics.jsonl, env.json and final_eval.json. "
        f"base={args.base} data={args.data} out={args.out} epochs={args.epochs} lr={args.lr}"
    )


if __name__ == "__main__":
    main()
