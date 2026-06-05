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

- **Vendor-prefix noise (`alibaba/cogview4`, `openai/wizardlm-2-8x22b`)** —
  models.dev keys some models under a vendor that did not make them: CogView4 is
  ZhipuAI/Z.AI (real repo `THUDM/CogView4-6B`), and WizardLM-2 is Microsoft, not
  OpenAI. The wrong prefix had been baked into the curated `canonical_id`.
  - Local handling: re-key the canonicals to the correct developer
    (`zai/cogview4`, `microsoft/WizardLM-2-8x22B`) in `core.yaml`, keeping the
    wrong forms as resolvable aliases.
  - A wider, low-severity pattern: models.dev also prefixes many community
    finetunes with a base vendor (e.g. `meta/<TheDrummer-model>`,
    `nvidia/<google-or-mistral-model>`). These resolve CORRECTLY in the registry
    (the alias maps the wrong upstream string to the right canonical), so they
    are kept as aliases, not dropped — dropping would lose resolution of a real
    upstream string. No local data is wrong; the noise is purely upstream.
  - **Action: flag to models.dev to correct the vendor prefixes.**
