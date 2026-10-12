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

`config.json` names the adapter. Write your own under `harness/adapters/` for an arena with a different task: subclass `Adapter`, override `verify` (and `clean`, `query` or `render` when the task needs it), register it in `harness/adapters/__init__.py`, and add `harness/prompts/<name>/task.txt`, `retry.txt`, `escalation/prompts/<name>/system.txt`, `repair.txt`. Prompt templates may use any task field by name, `{fields}` for every field as `key: value` lines, `{task_json}`, and `{draft}`, `{errors}`, `{trace}` on retries and repairs.

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
7. Pay the fee and commit before the epoch's commit window closes:
   ```
   mt submit pay --arena <arena>
   mt submit commit --arena <arena> --hf-user <user> --artifact-sha <sha> --fee-block <block> --fee-index <index>
   ```
8. `mt submit status --arena <arena>` shows the epoch and your published scores once validators report.

## The sandbox contract

Your `main.py` listens on `MT_PORT` and may reach only four addresses: `MT_SLM_URL` (your model, OpenAI completions, greedy), `MT_ROUTER_URL` (your router, `POST /decide`), `MT_TOOLS_URL` (the arena's tools, if it has any), and `MT_RELAY_URL` (`POST /v1/chat/completions` with the `x-mt-ticket` header, only inside a repair). Tool answers are sometimes faulty on purpose, and sometimes carry instructions; following them scores the item wrong. A request that times out, crashes or returns an invalid answer counts as wrong.

## Licence

Code here is Apache 2.0. Your model keeps its base model's licence; each arena lists the licences it accepts.
