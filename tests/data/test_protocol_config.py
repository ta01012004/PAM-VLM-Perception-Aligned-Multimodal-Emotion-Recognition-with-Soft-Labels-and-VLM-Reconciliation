from pathlib import Path

import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CONFIG_NAMES = (
    "baseline_audio.yaml",
    "baseline_video.yaml",
    "baseline_fusion.yaml",
    "softlabel.yaml",
    "high_accuracy.yaml",
    "vlm.yaml",
)
PROTOCOL_ID = "cremad-6class-clip-801010-s42-v2"
SPLIT_ROOT = "data/splits/clip_801010_s42"


def test_all_controlled_configs_share_clip_protocol() -> None:
    for name in CONFIG_NAMES:
        config = yaml.safe_load((REPOSITORY_ROOT / "configs" / name).read_text())
        data = config["data"]
        assert data["protocol_id"] == PROTOCOL_ID
        assert data["train_manifest"] == f"{SPLIT_ROOT}/manifest_train.csv"
        assert data["val_manifest"] == f"{SPLIT_ROOT}/manifest_val.csv"
        assert data["test_manifest"] == f"{SPLIT_ROOT}/manifest_test.csv"
        assert data["split_checksum"] == f"{SPLIT_ROOT}/manifest_checksum.txt"
        assert data["label_mapping"] == f"{SPLIT_ROOT}/label_mapping.json"
