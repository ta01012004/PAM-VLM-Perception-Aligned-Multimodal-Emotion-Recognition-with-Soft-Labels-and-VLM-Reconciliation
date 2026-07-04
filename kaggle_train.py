# PAM-VLM — Kaggle Training Notebook (Audio-only, WavLM)
# Dataset: ejlok1/cremad trên Kaggle (chỉ cần AudioWAV/)

# ============================================================
# CELL 1 — Install
# ============================================================
# !pip install -q "transformers>=4.41.0,<6.0.0" soundfile scipy pandas pyarrow scikit-learn

# ============================================================
# CELL 2 — Imports & Config
# ============================================================
import os, json, random, logging, time, re
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, confusion_matrix, classification_report

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOGGER = logging.getLogger(__name__)

AUDIO_DIR  = Path("/kaggle/input/datasets/ejlok1/cremad/AudioWAV")
OUTPUT_DIR = Path("/kaggle/working/checkpoints")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LABEL_MAPPING = {"angry":0, "disgust":1, "fear":2, "happy":3, "neutral":4, "sad":5}
NUM_CLASSES   = 6
SEED          = 42
EPOCHS        = 30
BATCH_SIZE    = 8
ACCUM_STEPS   = 4
LR            = 3e-4
WEIGHT_DECAY  = 0.01
PATIENCE      = 6
SAMPLE_RATE   = 16000
AUDIO_MAX_S   = 5.0
AMP           = True

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOGGER.info("Device: %s | GPU: %s", DEVICE,
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")

# ============================================================
# CELL 3 — Seed
# ============================================================
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

seed_everything(SEED)

# ============================================================
# CELL 4 — Audio I/O
# ============================================================
import soundfile as sf
from math import gcd
from scipy.signal import resample_poly

def load_audio(path, sample_rate=SAMPLE_RATE):
    samples, src_rate = sf.read(str(path), dtype="float32", always_2d=True)
    mono = samples.mean(axis=1).astype(np.float32)
    if src_rate != sample_rate:
        cd = gcd(int(src_rate), sample_rate)
        mono = resample_poly(mono, sample_rate // cd, int(src_rate) // cd).astype(np.float32)
    return torch.from_numpy(np.ascontiguousarray(mono))

# ============================================================
# CELL 5 — Build splits từ tên file
# ============================================================
EMO_MAP = {"ANG":"angry", "DIS":"disgust", "FEA":"fear",
           "HAP":"happy", "NEU":"neutral",  "SAD":"sad"}

records = []
for wav in sorted(AUDIO_DIR.glob("*.wav")):
    m = re.match(r"(\d+)_\w+_(\w+)_\w+\.wav", wav.name)
    if not m:
        continue
    emo_code = m.group(2)
    if emo_code not in EMO_MAP:
        continue
    records.append({"clip_id":    wav.stem,
                    "actor_id":   int(m.group(1)),
                    "emotion":    EMO_MAP[emo_code],
                    "audio_path": wav.name})

df_all = pd.DataFrame(records)
LOGGER.info("Total clips: %d | actors: %d", len(df_all), df_all["actor_id"].nunique())

rng    = np.random.default_rng(42)
actors = sorted(df_all["actor_id"].unique())
rng.shuffle(actors)
n       = len(actors)
train_a = set(actors[:int(n*0.70)])
val_a   = set(actors[int(n*0.70):int(n*0.80)])
test_a  = set(actors[int(n*0.80):])

df_train = df_all[df_all["actor_id"].isin(train_a)].copy()
df_val   = df_all[df_all["actor_id"].isin(val_a)].copy()
df_test  = df_all[df_all["actor_id"].isin(test_a)].copy()
LOGGER.info("Split — train:%d | val:%d | test:%d", len(df_train), len(df_val), len(df_test))

# ============================================================
# CELL 6 — Dataset & Dataloader
# ============================================================
AUDIO_MAX_LEN = int(AUDIO_MAX_S * SAMPLE_RATE)

class CREMADatasetAudio(Dataset):
    def __init__(self, df, augment=False):
        self.df      = df.reset_index(drop=True)
        self.augment = augment

    def __len__(self):
        return len(self.df)

    def _pad(self, wav):
        T = wav.shape[0]
        if T >= AUDIO_MAX_LEN:
            return wav[:AUDIO_MAX_LEN], torch.ones(AUDIO_MAX_LEN, dtype=torch.bool)
        pad = AUDIO_MAX_LEN - T
        return (F.pad(wav, (0, pad)),
                torch.cat([torch.ones(T, dtype=torch.bool),
                           torch.zeros(pad, dtype=torch.bool)]))

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        label = int(LABEL_MAPPING[row["emotion"]])
        wav   = load_audio(AUDIO_DIR / row["audio_path"])
        if self.augment and torch.rand(1).item() < 0.5:
            wav = torch.roll(wav, int(torch.randint(0, SAMPLE_RATE//4, (1,)).item()))
        wav, mask = self._pad(wav)
        return {"audio_values": wav,
                "audio_mask":   mask,
                "labels":       torch.tensor(label, dtype=torch.long)}

def collate_audio(batch):
    return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}

train_ds = CREMADatasetAudio(df_train, augment=True)
val_ds   = CREMADatasetAudio(df_val,   augment=False)
test_ds  = CREMADatasetAudio(df_test,  augment=False)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          collate_fn=collate_audio, num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                          collate_fn=collate_audio, num_workers=2, pin_memory=True)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                          collate_fn=collate_audio, num_workers=2, pin_memory=True)

# ============================================================
# CELL 7 — Model
# ============================================================
from transformers import WavLMModel

class AudioOnlyModel(nn.Module):
    def __init__(self, num_classes=6, proj_dim=256):
        super().__init__()
        self.wavlm = WavLMModel.from_pretrained("microsoft/wavlm-base-plus")
        for p in self.wavlm.parameters():
            p.requires_grad_(False)
        self.proj = nn.Sequential(
            nn.Linear(768, proj_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        self.head = nn.Linear(proj_dim, num_classes)

    def forward(self, audio_values, audio_mask):
        out    = self.wavlm(input_values=audio_values, attention_mask=audio_mask.long())
        hidden = out.last_hidden_state                        # [B, T, 768]
        m      = audio_mask[:, :hidden.size(1)].unsqueeze(-1).float()
        pooled = (hidden * m).sum(1) / m.sum(1).clamp(min=1) # [B, 768]
        return self.head(self.proj(pooled))                   # [B, 6]

# ============================================================
# CELL 8 — Loss & Metrics
# ============================================================
def compute_loss(logits, labels):
    return F.cross_entropy(logits, labels)

def compute_metrics(targets, preds):
    t, p = np.array(targets), np.array(preds)
    return {
        "accuracy": float((t == p).mean()),
        "macro_f1": float(f1_score(t, p, average="macro", zero_division=0)),
        "uar":      float(f1_score(t, p, average="macro", zero_division=0)),
    }

# ============================================================
# CELL 9 — Model + Optimizer
# ============================================================
model     = AudioOnlyModel(num_classes=NUM_CLASSES).to(DEVICE)
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total     = sum(p.numel() for p in model.parameters())
LOGGER.info("Params: total=%d trainable=%d (%.1f%%)", total, trainable, 100*trainable/total)

optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=LR, weight_decay=WEIGHT_DECAY,
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
scaler    = torch.amp.GradScaler("cuda", enabled=AMP and DEVICE.type == "cuda")

# ============================================================
# CELL 10 — Train loop
# ============================================================
best_metric, best_epoch, history = float("-inf"), -1, []

for epoch in range(EPOCHS):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    t0 = time.time()
    tr_loss, tr_n, tr_preds, tr_tgts = 0.0, 0, [], []

    for step, batch in enumerate(train_loader):
        bsz   = batch["labels"].shape[0]
        batch = {k: v.to(DEVICE, non_blocking=True) for k, v in batch.items()}

        with torch.autocast("cuda", enabled=AMP and DEVICE.type == "cuda"):
            logits = model(batch["audio_values"], batch["audio_mask"])
            loss   = compute_loss(logits, batch["labels"]) / ACCUM_STEPS

        scaler.scale(loss).backward()
        if (step + 1) % ACCUM_STEPS == 0 or step + 1 == len(train_loader):
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update()
            optimizer.zero_grad(set_to_none=True)

        tr_loss += float(loss.detach()) * ACCUM_STEPS * bsz
        tr_n    += bsz
        tr_preds.extend(logits.detach().argmax(-1).cpu().tolist())
        tr_tgts.extend(batch["labels"].cpu().tolist())

    scheduler.step()
    tr_loss /= tr_n
    tr_met   = compute_metrics(tr_tgts, tr_preds)

    model.eval()
    val_loss, val_n, val_preds, val_tgts = 0.0, 0, [], []
    with torch.inference_mode():
        for batch in val_loader:
            bsz   = batch["labels"].shape[0]
            batch = {k: v.to(DEVICE, non_blocking=True) for k, v in batch.items()}
            with torch.autocast("cuda", enabled=AMP and DEVICE.type == "cuda"):
                logits = model(batch["audio_values"], batch["audio_mask"])
                loss   = compute_loss(logits, batch["labels"])
            val_loss += float(loss.detach()) * bsz; val_n += bsz
            val_preds.extend(logits.argmax(-1).cpu().tolist())
            val_tgts.extend(batch["labels"].cpu().tolist())

    val_loss /= val_n
    val_met   = compute_metrics(val_tgts, val_preds)

    LOGGER.info("Epoch %2d/%d  %.0fs  tr_loss=%.4f  val_loss=%.4f  val_f1=%.4f  val_acc=%.4f",
                epoch+1, EPOCHS, time.time()-t0,
                tr_loss, val_loss, val_met["macro_f1"], val_met["accuracy"])

    history.append({"epoch": epoch, "train_loss": tr_loss, "val_loss": val_loss,
                    "train_metrics": tr_met, "val_metrics": val_met})

    mf1 = val_met["macro_f1"]
    if mf1 > best_metric:
        best_metric, best_epoch = mf1, epoch
        torch.save({"epoch": epoch, "model": model.state_dict(), "best_f1": mf1},
                   OUTPUT_DIR / "best.pt")
        LOGGER.info("  -> best F1=%.4f saved", mf1)

    if epoch - best_epoch >= PATIENCE:
        LOGGER.info("Early stopping (best epoch %d)", best_epoch+1)
        break

LOGGER.info("Done. Best val Macro-F1=%.4f @ epoch %d", best_metric, best_epoch+1)

# ============================================================
# CELL 11 — Test
# ============================================================
ckpt = torch.load(OUTPUT_DIR / "best.pt", map_location=DEVICE)
model.load_state_dict(ckpt["model"])
model.eval()

test_preds, test_tgts = [], []
with torch.inference_mode():
    for batch in test_loader:
        batch = {k: v.to(DEVICE, non_blocking=True) for k, v in batch.items()}
        with torch.autocast("cuda", enabled=AMP and DEVICE.type == "cuda"):
            logits = model(batch["audio_values"], batch["audio_mask"])
        test_preds.extend(logits.argmax(-1).cpu().tolist())
        test_tgts.extend(batch["labels"].cpu().tolist())

met = compute_metrics(test_tgts, test_preds)
print("\n=== TEST RESULTS ===")
print(f"Accuracy : {met['accuracy']:.4f}")
print(f"Macro-F1 : {met['macro_f1']:.4f}")
print(f"UAR      : {met['uar']:.4f}")

EMOTION_NAMES = ["angry","disgust","fear","happy","neutral","sad"]
print("\nClassification Report:")
print(classification_report(test_tgts, test_preds, target_names=EMOTION_NAMES, digits=4))
print("\nConfusion Matrix:")
print(confusion_matrix(test_tgts, test_preds))

json.dump({"best_val_f1": best_metric, "best_epoch": best_epoch,
           "test": met, "history": history},
          open(OUTPUT_DIR/"results.json","w"), indent=2)
LOGGER.info("Results saved.")
