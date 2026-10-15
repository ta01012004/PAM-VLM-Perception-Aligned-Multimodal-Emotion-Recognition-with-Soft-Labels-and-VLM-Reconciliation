# Actor-disjoint PAM evidence — 15 October 2026

This package covers all 59 separately trained runs used by the current PAM
study. It contains the complete exported test evidence, including outcomes
that do not favor PAM on recognition. It does not claim matched external
state of the art or broad statistical significance.

## Files

- `runs.csv`: one row per training run, with relative artifact paths and six
  original reported test metrics. Probabilities and scores use fractions/raw
  scores, not percentages.
- `evidence/primary/`: 16 primary outputs; full train/validation/test clip
  manifests and actor lists; original model/loader/label-map inputs; recorded
  configurations; per-run metrics; three pooled/single-split prediction exports.
- `evidence/ce_only/`: ten additional CE-only controls; paired-fold reports;
  both original ten-fold OOF prediction exports. The primary ten-fold outputs
  supply the CE+KL comparison; the word `hard` in historical filenames means
  **smoothed CE-only**, not literal one-hot training.
- `evidence/reviewer/`: all 33 fresh runs, eleven arms and three optimization
  seeds; configurations, model definitions, histories, resource/initialization
  records where available, and original descriptive reports.
- `tools/pam_hard_ablation/code_bundle/`: original primary/control source snapshot.
- `tools/pam_reviewer_ablation/code_bundle/`: deployed source with ablation switches.
  Both source bundles match the collected deployed Python files (42 original,
  43 reviewer files); they are not inferred from the paper or rewritten models.
- `tools/*`: preparation, scheduler drivers and their existing CPU tests.
- `analysis/`: independent six-metric recomputation, identity/actor checks,
  aggregation checks (including all reviewer means/SD and paired contrasts), and regression tests. See [REPRODUCE.md](REPRODUCE.md).
- `figures/`: previously generated descriptive ablation result plots.
- `source_provenance.json`: original and exported SHA256, plus whether a file
  was path-sanitized/reserialized; paths are relative to the private source workspace.
- `checksums.json`: SHA256 of every released package file except this checksum
  file itself. Hash verification detects accidental change; it is not a digital signature.
- [RESULTS.md](RESULTS.md): complete readable summary and interpretation limits.

## Provenance and privacy

Primary suite files and 34 reviewer model JSONs were retrieved read-only from
HPC on 15 October 2026. The other completed controls/ablations came from the
12 October snapshot. All 34 newly retrieved model JSON hashes match the frozen
reviewer input-hash records before path sanitization. Exported metrics are
recomputed, not simulated or estimated. No jobs were submitted or cancelled.

Private paths in structured exports are replaced by `ARTIFACTS_ROOT/`,
`CREMAD_ROOT`, `WEIGHTS_ROOT/`, `SOURCE_ROOT/`, `PRIVATE_HOME/`, or `LOCAL_ROOT/`.
These are **symbolic metadata**, not automatically expanded shell variables.
Numeric values, seeds, actor IDs, split membership and prediction CSVs are
preserved. A source SHA refers to the original bytes, whereas the export SHA
refers to the public bytes; redacted configuration files cannot match their
original byte hashes. The historical private-path input-hash maps are retained
as provenance, not executable portable freeze manifests.

CREMA-D actor/clip IDs and relative audio/video filenames are dataset metadata;
no raw speech, video, individual human-rating records, or model weights are
redistributed. Obtain CREMA-D separately from its maintainers and follow its
license: https://github.com/CheyneyComputerScience/CREMA-D . No new blanket
license is asserted for third-party data or pretrained encoders.

## Evidence boundaries

Five-fold testing uses public outer test identities with nine training actors
held out for validation; four clips are excluded (7,438 total OOF clips).
Ten-fold and 60/20/20 use the recorded actor assignments, not necessarily the
identities of another paper. Ten-fold uses 7,442 total OOF clips.
The holdout uses 55/18/18 actors, 4,496/1,470/1,476 clips.
The fresh 33-run block repeats seeds on this **one** retrospective assignment.
Repeated clips across seeds are not independent new observations.
Checkpoint reload, raw-media/annotation regeneration and end-to-end GPU
training are not performed by the public CPU validator. Checkpoints, raw
annotations/media and pretrained weights are deliberately absent.
