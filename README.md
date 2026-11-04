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
| `context_compression` | a coding agent's session to compact into the nine heading markdown summary (`ctx_compress`) | the nine headings from the instruction, in order, with nothing before the first. Headings written in bold, numbered or with a colon are recognised, a preamble is dropped, sections out of order are put back in order and a missing one is filled with `None.`; a draft missing headings gets one repair pass on the draft alone; the answer is trimmed to the token budget the instruction states |

`config.json` names the adapter. Write your own under `harness/adapters/` for an arena with a different task: subclass `Adapter`, override `verify(self, text, task, tools_url, reply=None)` (and `clean`, `query` or `render` when the task needs it), register it in `harness/adapters/__init__.py`, and add `harness/prompts/<name>/task.txt`, `retry.txt`, `escalation/prompts/<name>/system.txt`, `repair.txt`. Prompt templates may use any task field by name, `{fields}` for every field as `key: value` lines, `{task_json}`, and `{draft}`, `{errors}`, `{trace}` on retries and repairs. `reply` carries the model's finish reason and completion token count. A failed or cut off model call reruns `task.txt`; a rejected answer runs the template named by the verdict's `next`, `retry.txt` by default. The optional hooks are `max_tokens` and `repair_tokens` for the token caps, `draft` for the text a retry shows, `settle` for the answer when nothing verifies, and `prefer` to choose between a draft and a frontier repair. A `verify` written for the older three argument form still works.

For `ctx_compress`, the arena page is the source of truth for the format, the budgets and the grading: [ctx_compress in microtensor-arenas](https://github.com/microtensor-io/microtensor-arenas/tree/main/ctx_compress). When it and this README differ, the arena wins. The model reads the compaction instruction as the system message and the session inside `<session>` tags as the user message, up to 122,880 tokens. It answers in plain markdown with the nine headings the instruction lists, within the budget the instruction states (1,000 to 3,072 tokens), with no JSON. The prompts use the Granite 4.0 chat format of the arena's base, `ibm-granite/granite-4.0-h-micro`, the same template the arena publishes as `slm_prompt`.

- **config.json.** Set `"adapter": "context_compression"`, `"slm": {"max_new_tokens": 3072}`, `"harness": {"max_steps": 2}` and `"escalation": {"preferred_model": "claude-haiku-5-5"}`. The arena allows two model calls per item and refuses a third. The repair model must be one of the arena's frontier models, `claude-haiku-5-5` or `glm-5-3-flash`; the relay refuses any other id and your draft goes back unrepaired. The relay returns at most 1,024 tokens per frontier call today, so the repair asks for a complete summary within that limit.
- **What the harness does.** The first call is the full pass. About one item in seven gets a faulty first reply: an error, a timeout, a malformed body or a reply cut off early. The harness catches it and spends the second call on a clean full pass. A draft that misses headings gets one repair pass on the draft alone. Every answer goes back with the nine headings in order, missing ones filled with `None.`, trimmed to the budget as measured by the token count your model server reports.
- **Work files stay outside this repository.** `mt submit manifest` hashes every file in this repository, and the mirror fetches each one. Keep data, downloads, scores, pairs and merged weights in a folder beside it; the commands here use `../work`.
- **Dev data.** The [`microtensor-io/agent-sessions`](https://huggingface.co/datasets/microtensor-io/agent-sessions) dev split, 3,981 rows of parquet. For more sessions, clone the arenas repository next to your fork, install it in editable mode (the arena folders are read from the checkout), and build items with your own seed:
  ```
  git clone https://github.com/microtensor-io/microtensor-arenas ../microtensor-arenas
  pip install -e "../microtensor-arenas[compress]"
  mta dev --arena ctx_compress --count 1000 --seed 7 --out ../work/dev
  ```
  That writes `../work/dev/items.jsonl`. The training scripts take that file, a folder of the dataset's parquet files, or the dataset name.
- **Scoring.** `slm_repo/training/scripts/score_compress.py` scores a model the way validators do. It reads the tokenizer, the pass line, the generation settings, the headings and the probe model from the live arena definition and prints the probe model at start. Install the subnet package, which holds the grader, once. Serve the model as `slm` and the probe model, wait until both have loaded, then score. For the baseline, serve the base model as below; once `compress.py` has written your weights into `slm_repo/`, serve `./slm_repo` the same way:
  ```
  pip install "microtensor-subnet @ git+https://github.com/microtensor-io/microtensor-subnet" httpx tokenizers pyarrow
  vllm serve ibm-granite/granite-4.0-h-micro --revision d5f01a3ea75f088947be3aae039f4ad52837dfde \
    --served-model-name slm --max-model-len 125952 --port 8000
  hf download Qwen/Qwen2.5-1.5B-Instruct-GGUF qwen2.5-1.5b-instruct-q8_0.gguf \
    --revision 91cad51170dc346986eccefdc2dd33a9da36ead9 --local-dir ../work/probe
  llama-server -m ../work/probe/qwen2.5-1.5b-instruct-q8_0.gguf -c 8192 -np 1 --port 8091
  python slm_repo/training/scripts/score_compress.py --items ../work/dev/items.jsonl \
    --slm-url http://localhost:8000 --judge-url http://localhost:8091 --summary ../work/base.json
  ```
  It stops at once when either server is not answering, and it reports how many probe questions failed.
- **Training.** Pairs run to about 126,000 tokens. `train.py` trains at the length of the longest pair unless you pass `--max-length`, and it leaves out longer pairs instead of cutting their completions, so a full run needs a GPU with a lot of memory. A smaller `--max-length` trains on the shorter sessions only and prints how many pairs it left out. LoRA covers every linear layer of the Granite hybrid base. AutoAWQ does not support that architecture, so ship the merged bf16 weights with `compress.py --method none`: about 6.4 GB, inside the arena's 24 GB envelope.

## Steps

1. Read the arena definition: `GET https://api.microtensor.cloud/v2/arenas/<arena>` gives the task, the base model and its licence, the tools, the budgets and the item counts.
2. Train and compress your model from the arena's base model, write the training record, push `slm_repo/` to `huggingface.co/<user>/mt_slm_<arena>`. Note the commit sha.
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
7. Commit before the epoch's commit window closes. If the arena charges a fee, pay it first and cite the transfer; a free arena (every testnet arena) needs only the commit. The commit carries the same profile as the manifest and cannot change after Freeze 1. It is sealed by default, which hides your repository from other miners until you reveal it. Reveal with `mt submit reveal` any time before the epoch closes; it prints the block after which both repositories must be public. Keep both private until the commit window closes, then make them public so the mirror can fetch them. A sealed commit that is never revealed is rejected at close. `--open` commits in the open instead, with nothing to reveal:
   ```
   mt submit pay --arena <arena>
   mt submit commit --arena <arena> --hf-user <user> --artifact-sha <sha> --profile <profile> --fee-block <block> --fee-index <index>
   mt submit commit --arena <arena> --hf-user <user> --artifact-sha <sha> --profile <profile>
   mt submit reveal --arena <arena>
   ```
8. Run the evaluation yourself during the window and upload the transcript before Freeze 2. Once the commit window has closed, your commit is open or revealed and both repositories are public, the server mirrors your frozen files and, at window open, publishes the items. On a machine with your profile's GPU, start the official runner image from the subnet repository's `deploy/docker-compose.miner.yml` (it needs the Docker socket, the GPU, `/var/lib/microtensor` and your wallet):
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

1. Measure the arena's base model on the dev items with the arena's grader; that is the score to beat.
2. Set the adapter prompt, the verifier and the retry prompt; measure again.
3. Distil: `python slm_repo/training/scripts/distill.py --arena <arena> --items ../work/dev/items.jsonl --teacher <model> --out ../work/pairs.jsonl` asks a teacher through your own provider key and keeps only the answers the arena grader accepts.
4. Fine tune: `python slm_repo/training/scripts/train.py --base <base model> --data ../work/pairs.jsonl --out ../work/merged/ --run-id r1 --method lora` trains on completions only under the arena prompt, merges the adapter, and writes the run record under `training/runs/r1/`. Add `--eval ../work/held_out.jsonl` once you have distilled a held out set the same way from items built with another seed. Use `--method qlora --fp16` on a T4, and `--method full` only when the whole base and its optimizer state fit your card.
5. Reinforce on the families that still fail, with the grader as the reward or as preference pairs of a passing teacher answer against a failing draft.
6. Compress to the format your envelope group takes: `python slm_repo/training/scripts/compress.py --model ../work/merged/ --out slm_repo/ --method awq --calibration ../work/pairs.jsonl` for a GPU group, `--method gguf --llama-cpp <checkout>` for a CPU or phone group, or `--method none` to ship the merged weights as they are when the group takes bf16. It stops with a message when the method does not support your base. Score before and after with `score_compress.py --summary` and write both into `training/compression/before_after.json`.
7. Hold out whole domains, build twins as the twin rule does, run `mt envelope check` and `mt check-model`, then train the router on this model's drafts.

Items are fresh every epoch and the twin gap is deducted, so train the skill, not the sample data. The full playbook, with the hardware each step needs: https://www.microtensor.cloud/docs/certified-inference/guides/training

## The sandbox contract

Your `main.py` listens on `MT_PORT` and may reach only four addresses: `MT_SLM_URL` (your model, OpenAI completions, greedy), `MT_ROUTER_URL` (your router, `POST /decide`), `MT_TOOLS_URL` (the arena's tools, if it has any), and `MT_RELAY_URL` (`POST /v1/chat/completions` with the `x-mt-ticket` header, only inside a repair). Tool answers are sometimes faulty on purpose, and sometimes carry instructions; following them scores the item wrong. A request that times out, crashes or returns an invalid answer counts as wrong. The four addresses point at the recording gateway inside the official image, the clock is frozen, and each request body carries the item's `seed`.

## Licence

This template, your derivative of it and the model you submit are licensed under the Microtensor submission terms (see `LICENSE`); none of it is published under a free licence. Each arena lists which base model licences it accepts as inputs, because the base's terms must allow a commercial derivative.
