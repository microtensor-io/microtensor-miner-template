from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import subprocess
from pathlib import Path
from typing import Any


def calibration(path: str, count: int, seed: int) -> list[str]:
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    texts = [str(r.get("prompt", "")) + str(r.get("completion", "")) for r in rows if r.get("prompt")]
    random.Random(seed).shuffle(texts)
    return texts[:count]


def quantise_awq(model: str, out: Path, samples: list[str], bits: int, group: int) -> None:
    from awq import AutoAWQForCausalLM
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model)
    quantiser = AutoAWQForCausalLM.from_pretrained(model, safetensors=True)
    quantiser.quantize(
        tokenizer,
        quant_config={"zero_point": True, "q_group_size": group, "w_bit": bits, "version": "GEMM"},
        calib_data=samples,
    )
    quantiser.save_quantized(str(out), safetensors=True)
    tokenizer.save_pretrained(str(out))


def quantise_gptq(model: str, out: Path, samples: list[str], bits: int, group: int) -> None:
    from gptqmodel import GPTQModel, QuantizeConfig
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model)
    quantiser = GPTQModel.load(model, QuantizeConfig(bits=bits, group_size=group, desc_act=False))
    quantiser.quantize(samples, batch_size=1)
    quantiser.save(str(out))
    tokenizer.save_pretrained(str(out))


def quantise_gguf(model: str, out: Path, kind: str, llama_cpp: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    f16 = out / "model-f16.gguf"
    subprocess.run(
        ["python", str(llama_cpp / "convert_hf_to_gguf.py"), model, "--outtype", "f16", "--outfile", str(f16)],
        check=True,
    )
    binary = llama_cpp / "build" / "bin" / "llama-quantize"
    if not binary.exists():
        binary = llama_cpp / "llama-quantize"
    subprocess.run([str(binary), str(f16), str(out / "model.gguf"), kind], check=True)
    f16.unlink()
    for name in ("tokenizer.json", "tokenizer_config.json", "config.json"):
        source = Path(model) / name
        if source.is_file():
            shutil.copy2(source, out / name)


def main() -> None:
    parser = argparse.ArgumentParser(description="compress the merged model to the format your envelope group takes")
    parser.add_argument("--model", required=True, help="the merged model folder from train.py")
    parser.add_argument("--out", required=True, help="the folder that becomes slm_repo/ weights")
    parser.add_argument("--method", default="awq", choices=["awq", "gptq", "gguf"])
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--gguf-type", default="Q4_K_M", choices=["Q4_K_M", "Q5_K_S", "Q5_K_M", "Q6_K", "Q8_0"])
    parser.add_argument("--llama-cpp", default="", help="path to a llama.cpp checkout with llama-quantize built")
    parser.add_argument("--calibration", default="", help="pairs.jsonl from distill.py, used as calibration text")
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.bits < 4:
        raise SystemExit("nothing below 4 bit is accepted in any envelope group")
    out = Path(args.out)
    samples: list[str] = []
    calibration_sha = ""
    if args.method in ("awq", "gptq"):
        if not args.calibration:
            raise SystemExit("awq and gptq need --calibration pairs.jsonl from your own training inputs")
        samples = calibration(args.calibration, args.samples, args.seed)
        calibration_sha = hashlib.sha256("\n".join(samples).encode("utf-8")).hexdigest()
    if args.method == "awq":
        quantise_awq(args.model, out, samples, args.bits, args.group_size)
    elif args.method == "gptq":
        quantise_gptq(args.model, out, samples, args.bits, args.group_size)
    else:
        if not args.llama_cpp:
            raise SystemExit("gguf needs --llama-cpp pointing at a llama.cpp checkout")
        quantise_gguf(args.model, out, args.gguf_type, Path(args.llama_cpp))

    record: dict[str, Any] = {
        "method": args.method,
        "bits": args.bits if args.method != "gguf" else None,
        "group_size": args.group_size if args.method != "gguf" else None,
        "gguf_type": args.gguf_type if args.method == "gguf" else None,
        "calibration_samples": len(samples),
        "calibration_data_sha256": calibration_sha,
        "source_model": args.model,
    }
    folder = Path(__file__).resolve().parents[1] / "compression"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "compression.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    before_after = folder / "before_after.json"
    if not before_after.is_file():
        before_after.write_text(
            json.dumps({"before": {}, "after": {}, "note": "fill both from score_compress.py on the held out items"}, indent=2)
            + "\n",
            encoding="utf-8",
        )
    print(f"compressed weights at {out}; record at {folder / 'compression.json'}")


if __name__ == "__main__":
    main()
