# Step 06 — VLM reconciler (Qwen2.5-0.5B + LoRA)

## Objective
Attach the optional LLM branch: project fusion tokens into a small LLM, LoRA-tune
it to emit a strict emotion label + reasoning, and use it specifically to
**reconcile high audio/video disagreement cases**. The deterministic fusion head
remains the primary metric source.

## In scope
- `src/av_mer/models/projector.py`: fusion tokens (+ p_audio, p_video,
  disagreement signal) → LLM prefix embeddings.
- `src/av_mer/models/llm_adapter.py`: load configurable LLM (default
  `Qwen/Qwen2.5-0.5B-Instruct`), attach LoRA/QLoRA, generate constrained output
  `<emotion>label</emotion>` + short reason.
- `src/av_mer/data/disagreement.py`: compute KL(p_audio‖p_video) per clip; flag
  high-disagreement subset.
- `src/av_mer/evaluation/generative.py`: parse label from text; invalid-format
  rate, refusal/empty rate, avg token count (protocol §3).
- `configs/vlm.yaml`; LM loss term wired behind flag.
- Tests: projector shapes; parser on valid/invalid strings; NO LLM download in unit tests.

## Out of scope
- Zero-shot cross-task (step-07). Changing the classifier as metric source.

## Input contract
- Trained soft-label backbone (step-05), fusion_embedding interface (step-03).
- Disagreement flags.

## Output contract
- VLM branch generating constrained labels; classifier logits kept separate.
- A targeted analysis: on high-disagreement clips, does the reconciler fix cases
  the fusion head got wrong? Report delta with counts, not just averages.
- Invalid-output accounting for every generative eval.

## Acceptance criteria
- [ ] Projector output shape matches LLM embedding dim (tested).
- [ ] Output parser + invalid-format accounting unit-tested.
- [ ] Constrained decoding allows only the 6 labels or strict schema.
- [ ] LM loss logged separately; toggling VLM off leaves step-05 behavior intact.
- [ ] Reconciler evaluated on the disagreement subset specifically.
- [ ] LoRA-only training; trainable param count reported (base frozen/quantized).

## Constraints
- Training/generation via `sbatch` on `dgx-*` only.
- Do NOT use generated explanations as supervised ground truth (AGENTS rule).
- VLM is not required for classifier training; keep it fully toggleable.

## Verification commands
```bash
python -m pytest tests/models/test_projector.py tests/evaluation/test_generative_parse.py -q
sbatch scripts/train_vlm.sbatch
```

## Notes and decisions
- Expectation: VLM likely does NOT raise in-domain accuracy. Its value = reconciling
  conflicts + explanation + enabling step-07 zero-shot. State this in results.
- Keep prompt templates in config/files, decoupled from model code (needed for step-07).
