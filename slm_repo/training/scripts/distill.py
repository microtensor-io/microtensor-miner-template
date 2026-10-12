from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="generate teacher answers for your prompts")
    parser.add_argument("--teacher", required=True, help="the teacher model id and version")
    parser.add_argument("--prompts", required=True, help="jsonl with schema and question")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    print(
        "call the teacher through your own account on your own prompts, keep its answers and "
        f"record the teacher in training/distillation/teacher.json. teacher={args.teacher} "
        f"prompts={args.prompts} out={args.out}"
    )


if __name__ == "__main__":
    main()
