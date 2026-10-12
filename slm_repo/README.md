---
license: apache-2.0
base_model: Qwen/Qwen2.5-Coder-7B-Instruct
tags:
- microtensor
- sql_gen_s1
---

# mt_slm_sql_gen_s1

A specialised SQL generation model for the Microtensor `sql_gen_s1` arena.

| Field | Value |
|---|---|
| Base model | Qwen/Qwen2.5-Coder-7B-Instruct (Apache 2.0) |
| Method | see `training/recipe.json` |
| Arena | sql_gen_s1 |
| Scores | filled in from the ledger after the first epoch |

The root of this repository loads with `AutoModelForCausalLM`. Everything under `training/` is the record validators require and the archive keeps; it is never scored.
