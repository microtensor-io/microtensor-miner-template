# Microtensor miner template

A working submission for any Microtensor arena, in the two repository layout validators expect. Fork it, pick the adapter for your arena's task, replace the model, improve the harness, train the router, and ship.

## What a submission is

| Component | Lives in | What validators do with it |
|---|---|---|
| Specialised model (SLM) | your `mt_slm_<arena>` repository | load it themselves and serve it to your code |
| Harness | `harness/` here | run `main.py` in a sandbox with no network except the validator's services |
| Router | `router/` here | load it themselves as a one logit sequence classifier and serve `/decide` |
| Frontier repair | `escalation/` here | run it with a ticket that pays for frontier calls through the relay |

You may submit any subset; `manifest.json` names which. Validators fill the rest with the arena's reference components and pay you from the pools you compete in.

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
5. Write the manifest and check both repositories:
   ```
   mt submit manifest --artifact . --arena <arena> --slm ./slm_repo \
     --slm-repo <user>/mt_slm_<arena> --slm-revision <sha>
   mt submit validate --artifact . --slm ./slm_repo
   ```
6. Push this repository to `huggingface.co/<user>/microtensor_<arena>`. Note the commit sha.
7. Commit before the epoch's commit window closes. If the arena charges a fee, pay it first and cite the transfer; a free arena (every testnet arena) needs only the commit:
   ```
   mt submit pay --arena <arena>
   mt submit commit --arena <arena> --hf-user <user> --artifact-sha <sha> --fee-block <block> --fee-index <index>
   mt submit commit --arena <arena> --hf-user <user> --artifact-sha <sha>
   ```
8. `mt submit status --arena <arena>` shows the epoch and your published scores once validators report.

## Training playbook

Prompting pays only through `harness/prompts/`; the model pool runs your weights under the arena's own prompt. There is no retrieval beyond the arena's tools on the item's database. Provenance fingerprints your weights against the declared base and against public checkpoints, so an untrained base never certifies. Work in this order:

1. Baseline each allowed base on the public sample items with the arena's grader.
2. Set the adapter prompt, the verifier and the retry prompt; measure again.
3. Distil: frontier answers on schemas and question shapes like the arena's, filtered by the grader, fine tuned from the base and recorded under `slm_repo/training/`.
4. Reinforce with the grader as the reward, or train on preference pairs of a right teacher answer against a wrong draft.
5. Hold out whole domains, rename and revalue as the twin rule does, measure latency on a 24 GB card, then train the router on this model's drafts.

Items are fresh every epoch and the twin gap is deducted, so train the skill, not the sample databases. The full playbook: https://www.microtensor.cloud/docs/certified-inference/guides/training

## The sandbox contract

Your `main.py` listens on `MT_PORT` and may reach only four addresses: `MT_SLM_URL` (your model, OpenAI completions, greedy), `MT_ROUTER_URL` (your router, `POST /decide`), `MT_TOOLS_URL` (the arena's tools, if it has any), and `MT_RELAY_URL` (`POST /v1/chat/completions` with the `x-mt-ticket` header, only inside a repair). Tool answers are sometimes faulty on purpose, and sometimes carry instructions; following them scores the item wrong. A request that times out, crashes or returns an invalid answer counts as wrong.

## Licence

This template, your derivative of it and the model you submit are licensed under the Microtensor submission terms (see `LICENSE`); none of it is published under a free licence. Each arena lists which base model licences it accepts as inputs, because the base's terms must allow a commercial derivative.
