# Step 01 — CREMA-D manifest + per-modality soft labels

## Objective
Produce actor-independent manifests and per-modality soft-label tables from raw
CREMA-D filenames and `finishedResponses.csv`. **Data only. No model code.**

## In scope
- `src/av_mer/data/filename_parser.py`: parse `1010_MTI_SAD_XX` →
  {clip_id, actor_id, sentence_id, emotion, emotion_id, intensity}.
- `src/av_mer/data/split.py`: actor-independent split (seeded), disjointness asserts,
  save actor lists + manifest checksum.
- `src/av_mer/data/votes.py`: read `finishedResponses.csv`, group by clip and
  `queryType` (1=voice, 2=face, 3=AV), build a 6-dim probability vector per
  (clip, condition) from `respEmo` vote counts (Laplace-smoothed).
- `scripts/build_manifest.py`: CLI that writes:
  - `data/splits/manifest_{train,val,test}.csv` (schema: ARCHITECTURE §3 manifest row)
  - `data/splits/soft_labels_{voice,face,av}.parquet` (clip_id + 6 probs)
  - `data/splits/actors_{train,val,test}.json`, `data/splits/manifest_checksum.txt`
- Unit tests with **fake filenames / tiny synthetic CSV** (no real dataset).

## Out of scope
- Any audio/video decoding. Any torch Dataset. Any model. (those are step-02, step-03)

## Input contract
- Filenames from `data/raw/crema_d/AudioWAV/*.wav` and `.../VideoFlash/*.flv`.
- `data/raw/crema_d/finishedResponses.csv` columns incl. `clipName`, `queryType`, `respEmo`.
- respEmo letters map: A=angry, D=disgust, F=fear, H=happy, N=neutral, S=sad.
- Emotion code in filename: ANG,DIS,FEA,HAP,NEU,SAD.

## Output contract
- Manifests: one row per clip that has BOTH audio and video present.
- Soft labels: rows sum to 1.0 (±1e-6); missing condition → mark and fall back to
  hard one-hot from filename emotion (record fallback count).
- Splits: actor sets pairwise disjoint (hard assert). Default 70/10/11 actors by
  seed=42; ratios in config, not hard-coded.

## Acceptance criteria
- [ ] Filename parser unit-tested incl. malformed names (raise actionable error).
- [ ] Split disjointness asserted + tested.
- [ ] Vote aggregation tested on a synthetic CSV; probs sum to 1; smoothing correct.
- [ ] clip↔vote join reports coverage; zero silent drops (log missing).
- [ ] CPU-only, offline. Runs without torch.
- [ ] Reports exact commands + coverage stats.

## Constraints
- GPU/VRAM: none (login node OK for THIS step only — pure CPU/pandas).
- New deps: pandas, pyarrow only if not present; justify in summary.
- Never write outputs into Git (respect `.gitignore`).

## Verification commands
```bash
python -m pytest tests/data/test_filename_parser.py tests/data/test_split.py tests/data/test_votes.py -q
python scripts/build_manifest.py --raw data/raw/crema_d --out data/splits --seed 42
# then print: #clips per split, #actors per split, vote coverage %, fallback count
```

## Notes and decisions
- `clipName` has NO file extension; strip nothing, match against clip_id from filename.
- Some clipNames include `practice`/repeat rows — filter to real 7442 clips only.
- Do NOT invent labels from directory order. Emotion comes from filename token.
- Keep LABELS order canonical (protocol §2). Save mapping alongside outputs.
