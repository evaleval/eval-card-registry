# Upstream data issues — to flag + track

Known-wrong records in the upstream sources that the registry corrects LOCALLY
(see `seed/models/enrichments/upstream_corrections.yaml` + `core.yaml`
`skip_source_ids`) pending an upstream fix. Once an upstream record is corrected,
remove the corresponding local override and this entry.

## evaleval/EEE_datastore

- **`openai/GPT-J-6B` and `openai/GPT-NeoX-20B`** — GPT-J and GPT-NeoX are
  **EleutherAI** models. The EEE datastore records evaluations under the
  `openai/` namespace (alongside the correct `EleutherAI/...` records); HF marks
  the `openai/` ids `unresolved_not_found_or_inaccessible`. The developer
  attribution is wrong upstream.
  - Local handling: map the wrong raw → the real EleutherAI repo
    (`upstream_corrections.yaml`); drop the tier3 mints of the wrong ids
    (`core.yaml` `skip_source_ids`).
  - **Action: flag to the EEE maintainers to fix the source records.**

## models.dev

- **`abacusai/Dracarys-72B-Instruct`** — models.dev gives this real Qwen2-72B
  repo the display/alias **"Llama 3.1 70B Dracarys 2"**, which denotes a
  different (Llama-3.1-70B) model. The HF record for this id is correct
  (`Dracarys-72B-Instruct`).
  - Local handling: override the display back to `Dracarys-72B-Instruct`
    (`upstream_corrections.yaml`). The stray models.dev alias is low-impact (not
    an EEE evaluation id); it clears when models.dev corrects the catalog entry
    or when the generator gains display/id size-consistency hygiene.
  - **Action: flag to models.dev to correct the catalog label.**
