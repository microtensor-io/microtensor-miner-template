---
license: other
license_name: microtensor-submission
license_link: https://www.microtensor.cloud/terms
base_model: <allowed base>
tags:
- microtensor
- <arena>
---

# mt_slm_<arena>

A specialised model for the Microtensor `<arena>` arena.

| Field | Value |
|---|---|
| Base model | an allowed base from the arena definition, with its licence |
| Method | see `training/recipe.json` |
| Arena | <arena> |
| Scores | filled in from the ledger after the first epoch |

The root of this repository loads with `AutoModelForCausalLM`. Everything under `training/` is the record validators require and the archive keeps; it is never scored.
