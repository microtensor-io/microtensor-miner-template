from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def main() -> None:
    parser = argparse.ArgumentParser(description="fine tune an allowed base on grader filtered pairs")
    parser.add_argument("--base", required=True, help="an allowed base, for example Qwen/Qwen2.5-Coder-1.5B-Instruct")
    parser.add_argument("--data", required=True, help="jsonl of {prompt, completion} from distill.py")
    parser.add_argument("--eval", default="", help="held out jsonl of the same shape")
    parser.add_argument("--out", required=True, help="where the merged model and tokenizer go")
    parser.add_argument("--run-id", required=True, help="folder name under training/runs/")
    parser.add_argument("--method", default="lora", choices=["full", "lora", "qlora"])
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--lr", type=float, default=0.0, help="default 1e-5 full, 1.5e-4 lora and qlora")
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--accumulate", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fp16", action="store_true", help="use on cards without bf16, such as the T4")
    args = parser.parse_args()

    lr = args.lr or (1e-5 if args.method == "full" else 1.5e-4)
    out = Path(args.out)
    run_dir = Path(__file__).resolve().parents[1] / "runs" / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    dtype = torch.float16 if args.fp16 else torch.bfloat16

    tokenizer = AutoTokenizer.from_pretrained(args.base)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant = None
    if args.method == "qlora":
        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        )
    model = AutoModelForCausalLM.from_pretrained(
        args.base,
        dtype=dtype,
        quantization_config=quant,
        device_map="auto",
        attn_implementation="sdpa",
    )

    data = load_dataset("json", data_files={"train": args.data, **({"eval": args.eval} if args.eval else {})})
    peft_config = None
    if args.method != "full":
        peft_config = LoraConfig(
            r=args.rank,
            lora_alpha=args.rank * 2,
            lora_dropout=0.05,
            target_modules=list(TARGETS),
            task_type="CAUSAL_LM",
        )

    config = SFTConfig(
        output_dir=str(run_dir / "checkpoints"),
        num_train_epochs=args.epochs,
        learning_rate=lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.accumulate,
        gradient_checkpointing=True,
        max_length=args.max_length,
        completion_only_loss=True,
        packing=False,
        bf16=not args.fp16,
        fp16=args.fp16,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch" if args.eval else "no",
        load_best_model_at_end=bool(args.eval),
        metric_for_best_model="eval_loss" if args.eval else None,
        seed=args.seed,
        report_to="none",
    )
    trainer = SFTTrainer(
        model=model,
        args=config,
        train_dataset=data["train"],
        eval_dataset=data["eval"] if args.eval else None,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    started = time.time()
    trainer.train()
    hours = (time.time() - started) / 3600

    trained = trainer.model
    if args.method != "full":
        trained = trained.merge_and_unload()
    if args.method == "qlora":
        trained = trained.to(dtype)
    trained.save_pretrained(str(out), safe_serialization=True)
    tokenizer.save_pretrained(str(out))

    (run_dir / "config.json").write_text(
        json.dumps(
            {
                "run_id": args.run_id,
                "base_model": args.base,
                "method": args.method,
                "rank": args.rank if args.method != "full" else None,
                "epochs": args.epochs,
                "learning_rate": lr,
                "max_length": args.max_length,
                "batch_size": args.batch * args.accumulate,
                "seed": args.seed,
                "data": args.data,
                "eval": args.eval,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "metrics.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in trainer.state.log_history), encoding="utf-8"
    )
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""
    (run_dir / "env.json").write_text(
        json.dumps(
            {
                "libraries": {
                    "torch": torch.__version__,
                    "transformers": __import__("transformers").__version__,
                    "trl": __import__("trl").__version__,
                    "peft": __import__("peft").__version__,
                },
                "gpu": gpu,
                "profile": "",
                "python": platform.python_version(),
                "hours": round(hours, 2),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    final = read_json(run_dir / "final_eval.json")
    final.setdefault("note", "fill from score_compress.py on the held out items")
    (run_dir / "final_eval.json").write_text(json.dumps(final, indent=2) + "\n", encoding="utf-8")
    print(f"merged model at {out}; run record at {run_dir}")


if __name__ == "__main__":
    main()
