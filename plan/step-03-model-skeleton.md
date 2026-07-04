# Step 03 — Model skeleton (encoders + fusion + heads)

## Objective
Implement the multimodal model as pure tensor-in / tensor-out modules with
validated shapes. **No training loop, no data loading, no LLM yet.**

## In scope
- `src/av_mer/models/audio_encoder.py`: wrap configurable speech encoder
  (default `microsoft/wavlm-base-plus`), frozen by default, output temporal tokens
  + mask. Model ID from config.
- `src/av_mer/models/video_encoder.py`: per-frame compact backbone (default
  EfficientNet-B0) → 2-layer Transformer temporal encoder → tokens + mask.
- `src/av_mer/models/fusion.py`: bidirectional cross-attention + gated pooling →
  fusion embedding.
- `src/av_mer/models/heads.py`: audio head, video head, fusion head → logits[B,6].
- `src/av_mer/models/pam_vlm.py`: assemble into one module returning the
  "Model output" dict (ARCHITECTURE §3): logits, audio_logits, video_logits,
  fusion_embedding. (generated_text stays None until step-06.)
- Shape/mask unit tests with random tensors (NO real weights download in tests).

## Out of scope
- Losses, optimizer, checkpointing (step-04). LLM branch (step-06).

## Input contract
- Collated batch tensors from step-02 (audio_values, masks, video_values, masks).
- Config: encoder IDs, hidden dims, num heads, freeze flags, dropout.

## Output contract
- Output dict with `logits[B,6]`, `audio_logits[B,6]`, `video_logits[B,6]`,
  `fusion_embedding[B,H]`. `loss=None`, `generated_text=None`.
- Every token sequence carries a padding mask; reduction rule documented in code.

## Acceptance criteria
- [ ] Forward pass on random batch returns correct shapes (tested, no download).
- [ ] Masks respected (padded positions do not affect pooled output — tested).
- [ ] Frozen-encoder mode exposes trainable param count; matches expectation.
- [ ] Encoder IDs / dims come from config, not literals in logic.
- [ ] CPU forward smoke passes on tiny dims.

## Constraints
- GPU/VRAM: tests CPU + tiny dims. Real WavLM load → compute node only.
- No network in unit tests: use a `--tiny`/mock encoder path for CPU tests, real
  encoder behind config for GPU runs.
- Keep classifier logits separate from any future generated text.

## Verification commands
```bash
python -m pytest tests/models/test_shapes.py tests/models/test_masks.py \
                 tests/models/test_param_count.py -q
```

## Notes and decisions
- Do NOT equate audio-token index with video-frame index (ARCHITECTURE §5).
- audio_logits/video_logits exist so step-05 can supervise them with soft labels.
- fusion_embedding is the interface step-06's projector will consume — keep it stable.
