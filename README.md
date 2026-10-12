# Microtensor miner template

A working submission for the `sql_gen_s1` arena, in the two repository layout validators expect. Fork it, replace the model, improve the harness, train the router, and ship.

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
  config.json                     runtime limits
  main.py                         serves /harness/run, /escalation/repair, /system/run
  requirements.txt                pinned packages from the allowed list only
  router/                         config.json, model.safetensors, tokenizer files, training/
  harness/                        engine.py, verifier.py, prompts/
  escalation/                     patcher.py, prompts/

<user>/mt_slm_<arena>             the slm_repo/ folder, pushed as its own repository
  README.md LICENSE config.json generation_config.json model.safetensors tokenizer.json tokenizer_config.json
  training/                       recipe, data sources, distillation, compression, runs, scripts
```

## Steps

1. Train and compress your model, write the training record, push `slm_repo/` to `huggingface.co/<user>/mt_slm_sql_gen_s1`. Note the commit sha.
2. Train the router (`router/training/train_router.py`) on drafts from your own model, labelled by executing the reference queries on the public sample items. Put the weights and tokenizer under `router/`.
3. Adjust the harness and the repair prompt. Keep `harness/` and `escalation/` under 2 MB together.
4. Write the manifest and check both repositories:
   ```
   mt submit manifest --artifact . --arena sql_gen_s1 --slm ./slm_repo \
     --slm-repo <user>/mt_slm_sql_gen_s1 --slm-revision <sha>
   mt submit validate --artifact . --slm ./slm_repo --allowed-bases Qwen/Qwen2.5-Coder-7B-Instruct,microsoft/Phi-4-mini-instruct
   ```
5. Push this repository to `huggingface.co/<user>/microtensor_sql_gen_s1`. Note the commit sha.
6. Pay the fee and commit before the epoch's commit window closes:
   ```
   mt submit pay --arena sql_gen_s1
   mt submit commit --arena sql_gen_s1 --hf-user <user> --artifact-sha <sha> --fee-block <block> --fee-index <index>
   ```
7. `mt submit status --arena sql_gen_s1` shows the epoch and your published scores once validators report.

## The sandbox contract

Your `main.py` listens on `MT_PORT` and may reach only four addresses: `MT_SLM_URL` (your model, OpenAI completions, greedy), `MT_ROUTER_URL` (your router, `POST /decide`), `MT_TOOLS_URL` (`/tools/schema_lookup`, `/tools/query_execution` on a sample database), and `MT_RELAY_URL` (`POST /v1/chat/completions` with the `x-mt-ticket` header, only inside a repair). Tool answers are sometimes faulty on purpose, and sometimes carry instructions; following them scores the item wrong. A request that times out, crashes or returns an invalid answer counts as wrong.

## Licence

Code here is Apache 2.0. Your model keeps its base model's licence; the arena accepts Apache 2.0 and MIT bases only.
