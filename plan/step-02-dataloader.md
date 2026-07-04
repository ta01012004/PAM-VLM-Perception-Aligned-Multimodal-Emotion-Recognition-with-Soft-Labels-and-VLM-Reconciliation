# Step 02 — Dataloader, media decoding, collator

## Objective
Turn manifests into batched tensors: decode audio + video, crop faces, sample
frames deterministically, collate with masks. **No model, no training.**

## In scope
- `src/av_mer/data/audio_io.py`: load wav → mono, resample to `sample_rate` (cfg),
  return FloatTensor[num_samples] + true length.
- `src/av_mer/data/video_io.py`: decode `.flv` → sampled frames (deterministic in
  val/test), face detect + crop + align, resize to cfg size. Explicit policy +
  counter for face-detection failure (do not silently skip).
- `src/av_mer/data/dataset.py`: `CremaDataset` returning the "Dataset item before
  collation" dict (ARCHITECTURE §3), plus soft-label vectors for voice/face/av.
- `src/av_mer/data/collate.py`: pad audio + frames, build `audio_mask`,
  `video_mask`; produce "Collated batch" dict (ARCHITECTURE §3).
- Train-only augmentation hook (applied AFTER split; off in val/test).
- Tests using tiny synthetic wav/mp4 fixtures generated in a temp dir.

## Out of scope
- Encoders, fusion, LLM (step-03). Any .backward() (step-04).

## Input contract
- Manifests + soft-label tables from step-01.
- Config keys: `sample_rate`, `num_frames`, `frame_size`, `face_detector`, aug flags.

## Output contract
- Item dict: clip_id, actor_id, label, audio[FloatTensor], audio_sample_rate,
  video[FloatTensor F,C,H,W], soft_voice[6], soft_face[6], soft_av[6].
- Batch dict matches ARCHITECTURE §3 collated batch + soft-label tensors[B,6].
- Deterministic frame sampling in val/test (same clip → same frames).

## Acceptance criteria
- [ ] Audio loader: mono + correct sample rate, tested on synthetic wav.
- [ ] Video loader: fixed #frames, face-fail policy tested (fallback = center crop, counted).
- [ ] Collator: shapes + masks correct for variable-length inputs (tested).
- [ ] Determinism test: val sampling reproducible across two loads.
- [ ] CPU-only smoke: iterate 2 batches from a 4-clip fixture manifest.
- [ ] No real CREMA-D needed for unit tests.

## Constraints
- GPU/VRAM: none for tests. Real decoding of full data → later, on compute node.
- Deps: torchaudio/soundfile + decord or torchvision.io + a face detector
  (e.g. facenet-pytorch MTCNN). Justify each; pin versions.
- `.flv` decode is the top risk — add a standalone smoke script on 3 real clips
  (run on a compute node via sbatch, NOT login node).

## Verification commands
```bash
python -m pytest tests/data/test_audio_io.py tests/data/test_video_io.py \
                 tests/data/test_collate.py tests/data/test_dataset.py -q
# real-media smoke (compute node): sbatch scripts/smoke_decode.sbatch
```

## Notes and decisions
- Models must NOT open media paths in forward (AGENTS rule) — all IO lives here.
- Record undecodable media into a report file, not stderr noise.
- Face crop config-driven; keep raw-frame fallback so pipeline never crashes on 1 bad clip.
