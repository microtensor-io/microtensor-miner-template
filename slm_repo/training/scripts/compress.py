from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="quantise the fine tuned model for the arena")
    parser.add_argument("--model", required=True)
    parser.add_argument("--method", default="awq", choices=["awq", "gptq", "fp8", "none"])
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    print(
        "quantise with the method the arena allows and record it in training/compression/; "
        f"model={args.model} method={args.method} bits={args.bits} out={args.out}"
    )


if __name__ == "__main__":
    main()
