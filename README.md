# Microtensor miner template

A working submission for any Microtensor arena, in the two repository layout validators expect. Fork it, pick the adapter for your arena's task, replace the model, improve the harness, train the router, and ship.

## What a submission is

| Component | Lives in | What happens to it |
|---|---|---|
| Specialised model (SLM) | your `mt_slm_<arena>` repository | served by the official engine image on your GPU during the window; validators spot check its calls on the same hardware profile |
| Harness | `harness/` here | `main.py` runs in the official sandbox on your GPU with no network except the recording gateway; validators replay it on CPU against the transcript |
| Router | `router/` here | loaded by validators from the mirror as a one logit sequence classifier for the router pass |
| Frontier repair | `escalation/` here | runs in the sandbox with a ticket that pays for frontier calls through the relay; the signed receipts are checked on replay |

You may submit any subset; `manifest.json` names which. Validators fill the rest with the arena's reference components and pay you from the pools you compete in.

You declare a hardware profile, run the official images on your own GPU during the evaluation window and upload a transcript; validators replay the harness on CPU and spot check the model calls on the same profile. The docs: https://www.microtensor.cloud/docs/certified-inference/system/decentralised-evaluation

## Repository layout

```
<user>/microtensor_<arena>        this repository
  manifest.json                   written by `mt submit manifest`
  config.json                     runtime limits and the adapter name
  main.py                         serves /harness/run, /escalation/repair, /system/run
  requirements.txt                pinned packages from the allowed list only
  harness/                        engine.py, verifier.py, adapters/, prompts/<adapter>/
  escalation/                     patcher.py, prompts/<adapter>/
  router/                         config.json, model.safetensors, tokenizer files, training/

<user>/mt_slm_<arena>             the slm_repo/ folder, pushed as its own repository
  README.md LICENSE config.json generation_config.json model.safetensors tokenizer.json tokenizer_config.json
  training/                       recipe, data sources, distillation, compression, runs, scripts
```

## Adapters

An arena's items carry a task document (`task`) whose fields the arena definition describes in `schemas.input`. The harness and the repair prompt are written once, against an **adapter** that knows how to prompt for that task and how to check an answer before it is returned:

| Adapter | For arenas whose task is | Verification |
|---|---|---|
| `generic` | any structured or free text task | non empty, length bound, refuses injected instructions |
| `sql_generation` | a question plus a schema, answered with one SQLite query | parses with sqlglot, names only known tables, executes on the arena's sample database |
| `context_compression` | a coding agent's session to compact into the five field JSON (`ctx_compress`) | one JSON object; folds extra keys into `summary` and object entries into plain strings; checks the token budget the instruction states |

`config.json` names the adapter. Write your own under `harness/adapters/` for an arena with a different task: subclass `Adapter`, override `verify` (and `clean`, `query` or `render` when the task needs it), register it in `harness/adapters/__init__.py`, and add `harness/prompts/<name>/task.txt`, `retry.txt`, `escalation/prompts/<name>/system.txt`, `repair.txt`. Prompt templates may use any task field by name, `{fields}` for every field as `key: value` lines, `{task_json}`, and `{draft}`, `{errors}`, `{trace}` on retries and repairs.

For `ctx_compress`, set `config.json` to `"adapter": "context_compression"`, `"slm": {"max_context": 32768, "max_new_tokens": 3072}`. The prompts use the Qwen chat format; for Phi 4 mini, change `harness/prompts/context_compression/*.txt` to its `<|system|>`, `<|user|>`, `<|assistant|>` and `<|end|>` markers.

## Steps

1. Read the arena definition: `GET https://api.microtensor.cloud/v2/arenas/<arena>` gives the task, the allowed bases and licences, the tools, the budgets and the item counts.
2. Train and compress your model from an allowed base, write the training record, push `slm_repo/` to `huggingface.co/<user>/mt_slm_<arena>`. Note the commit sha.
3. Train the router (`router/training/train_router.py`) on drafts from your own model, labelled by the arena's grader on the public sample items. Put the weights and tokenizer under `router/`.
4. Set `"adapter"` in `config.json`, adjust the prompts and the verifier. Keep `harness/` and `escalation/` under 2 MB together.
5. Write the manifest and check both repositories. Pick the hardware profile of the GPU you will run on (`evaluation.profiles` in the definition lists the allowed ones, https://www.microtensor.cloud/docs/certified-inference/system/hardware-profiles describes them), pass it to the manifest, and run the pre-flight check before you commit:
   ```
   mt submit manifest --artifact . --arena <arena> --slm ./slm_repo \
     --slm-repo <user>/mt_slm_<arena> --slm-revision <sha> --profile <profile>
   mt submit validate --artifact . --slm ./slm_repo
   mt check-model --slm ./slm_repo --profile <profile>
   ```
   `--profile` writes `schema_version: 2` with `profile`, `weights_sha256` and the official image digest. `check-model` dry loads the config, tokenizer and safetensors headers on CPU and prints `ok` or the reason the mirror would refuse the model; a model that fails it is never evaluated.
6. Push this repository to `huggingface.co/<user>/microtensor_<arena>`. Note the commit sha.
7. Commit before the epoch's commit window closes. If the arena charges a fee, pay it first and cite the transfer; a free arena (every testnet arena) needs only the commit. The commit carries the same profile as the manifest and cannot change after Freeze 1:
   ```
   mt submit pay --arena <arena>
   mt submit commit --arena <arena> --hf-user <user> --artifact-sha <sha> --profile <profile> --fee-block <block> --fee-index <index>
   mt submit commit --arena <arena> --hf-user <user> --artifact-sha <sha> --profile <profile>
   ```
8. Run the evaluation yourself during the window and upload the transcript before Freeze 2. After the commit window closes the server mirrors your frozen files and, at window open, publishes the items. On a machine with your profile's GPU, start the official runner image from the subnet repository's `deploy/docker-compose.miner.yml` (it needs the Docker socket, the GPU, `/var/lib/microtensor` and your wallet):
   ```
   git clone https://github.com/microtensor-io/microtensor-subnet
   cd microtensor-subnet/deploy
   cp .env.example .env        # wallet names, MT_ARENA, MT_EPOCH, DOCKER_GID
   docker compose -f docker-compose.miner.yml up
   ```
   It runs `mt miner run --arena <arena> --epoch <n>`: refuses unless the mirror lists you as `mirrored` on an allowed profile, pulls your frozen artifact, weights and the inputs bundle from the mirror (never from your live repositories), starts the pinned engine and sandbox, runs every pool in the fixed order, then writes, signs and uploads the transcript and prints the receipt `{"transcript_sha256", "received_at", "receipt_sig"}`. Keep the receipt; it proves the upload landed before the close block. A second upload before the deadline replaces the first; nothing lands after it. Before your first upload with a new harness, replay it the way validators will:
   ```
   mt miner replay --transcript transcript.json --arena <arena> --epoch <n>
   ```
   Nothing the runner times is used for scoring; a validator on your profile times your sampled calls against the reference on the same card. Arenas without the `evaluation` block skip this step: validators run everything.
9. `mt submit status --arena <arena>` shows the epoch and your published scores once validators report.

## Determinism rules

Validators replay `main.py` on CPU against the transcript your run produced, and every request it makes must match the recording exactly and in order. Code that breaks one of these rules fails replay and scores zero for the epoch:

- No clock reads beyond the frozen clock. The sandbox's time is fixed; no wall time, timers or timeouts of your own.
- Randomness only from the provided seed. Every request carries `seed`; never use `random`, `numpy.random` or `os.urandom` unseeded.
- No concurrency that can reorder calls. Model, tool and relay calls go out in a fixed order per item; no threads, no async fan out.
- No file or network access outside the gateway. Read your own artifact, write nothing, reach only the four service URLs.

The template's `harness/`, `escalation/` and `main.py` follow them. `mt miner replay` tells you before validators do.

## Training playbook

Prompting pays only through `harness/prompts/`; the model pool runs your weights under the arena's own prompt. There is no retrieval beyond the arena's tools on the item's database. Provenance fingerprints your weights against the declared base and against public checkpoints, so an untrained base never certifies. Work in this order:

1. Baseline each allowed base on the arena's dev items with the arena's grader.
2. Set the adapter prompt, the verifier and the retry prompt; measure again.
3. Distil: `scripts/distill.py --arena <arena> --items dev/items.jsonl --teacher <model> --out pairs.jsonl` asks a teacher through your own provider key and keeps only the answers the arena grader accepts.
4. Fine tune: `scripts/train.py --base <allowed base> --data pairs.jsonl --eval held_out.jsonl --out merged/ --run-id r1 --method lora` trains on completions only under the arena prompt, merges the adapter, and writes the run record under `training/runs/r1/`. Use `--method qlora --fp16` on a T4, `--method full` for a 1.5B model on a 24 GB card.
5. Reinforce on the families that still fail, with the grader as the reward or as preference pairs of a passing teacher answer against a failing draft.
6. Compress to the format your envelope group takes: `scripts/compress.py --model merged/ --out slm_repo/ --method awq --calibration pairs.jsonl` for a GPU group, `--method gguf --llama-cpp <checkout>` for a CPU or phone group. Score before and after with `scripts/score_compress.py` and write both into `training/compression/before_after.json`.
7. Hold out whole domains, build twins as the twin rule does, run `mt envelope check` and `mt check-model`, then train the router on this model's drafts.

Items are fresh every epoch and the twin gap is deducted, so train the skill, not the sample data. The full playbook, with the hardware each step needs: https://www.microtensor.cloud/docs/certified-inference/guides/training

## The sandbox contract

Your `main.py` listens on `MT_PORT` and may reach only four addresses: `MT_SLM_URL` (your model, OpenAI completions, greedy), `MT_ROUTER_URL` (your router, `POST /decide`), `MT_TOOLS_URL` (the arena's tools, if it has any), and `MT_RELAY_URL` (`POST /v1/chat/completions` with the `x-mt-ticket` header, only inside a repair). Tool answers are sometimes faulty on purpose, and sometimes carry instructions; following them scores the item wrong. A request that times out, crashes or returns an invalid answer counts as wrong. The four addresses point at the recording gateway inside the official image, the clock is frozen, and each request body carries the item's `seed`.

## Licence

This template, your derivative of it and the model you submit are licensed under the Microtensor submission terms (see `LICENSE`); none of it is published under a free licence. Each arena lists which base model licences it accepts as inputs, because the base's terms must allow a commercial derivative.
