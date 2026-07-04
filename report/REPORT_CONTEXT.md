# PAM-VLM — Context đầy đủ cho báo cáo
# (Codex đọc file này thay cho việc đọc toàn bộ codebase)

---

## 1. Bài toán

**Tên đề tài:** Nâng cao hiệu suất mô hình Audio-Visual Emotion Recognition trên CREMA-D

**Bài toán:** Nhận diện **6 cảm xúc** (angry, disgust, fear, happy, neutral, sad) từ
clip video ngắn có cả giọng nói lẫn khuôn mặt. Nhãn theo thứ tự cố định:
`{"angry":0, "disgust":1, "fear":2, "happy":3, "neutral":4, "sad":5}`

**Dataset:** CREMA-D (Crowd-sourced Emotional Multimodal Actors Dataset)
- 7,442 clip từ **91 diễn viên** (48 nam, 43 nữ), câu kịch bản tiếng Anh
- Gán nhãn bằng crowd-sourcing: mỗi clip được **219,000+ người bình chọn** qua 3 điều kiện:
  - `queryType=1`: chỉ nghe giọng (voice-only) — 73,254 phiếu
  - `queryType=2`: chỉ nhìn mặt (face-only) — 73,192 phiếu
  - `queryType=3`: nghe + nhìn (audio-visual) — 73,242 phiếu
- File audio: 7,442 `.wav` (16kHz mono), file video: 7,443 `.flv`

**Lý do chọn CREMA-D:** có phiếu bầu per-modality (3 điều kiện) — tài sản gần như
chưa ai khai thác cho Label Distribution Learning.

---

## 2. Vấn đề với benchmark hiện tại (QUAN TRỌNG — đây là đóng góp phân tích)

Các bài báo trên CREMA-D báo cáo số rất loạn (69% → 95%) vì **protocol khác nhau**:

| Vấn đề | Chi tiết |
|---|---|
| **Random clip split** | Cùng diễn viên có mặt ở cả train lẫn test → model học "chất giọng người X" chứ không phải cảm xúc → điểm ảo |
| **Augment trước split** | Clip gốc được augment thành nhiều bản, các bản rơi vào cả train và test → bản sao lúc test → điểm vọt |
| **Vượt trần đồng thuận người** | Con người xem audio+video chỉ đồng thuận ~60–75%; một số paper báo cáo 95–100% — vật lý không thể đúng |
| **Bỏ lớp cảm xúc** | Một số báo cáo chỉ 4-5 lớp thay vì 6 → dễ hơn nhưng không so được |

**Bảng so sánh benchmark (dùng trong báo cáo — Bảng 1):**

| Phương pháp | Modality | Protocol | Acc CREMA-D | Ghi chú |
|---|---|---|---|---|
| Hybrid DL + FE (2025) | Audio | Random split (nghi) | 95.10% | 100% trên TESS/EMO-DB → leakage |
| Spectral + Attention (2025) | Audio | Random split | 89.31% | |
| Transformer-CNN (2025) | Audio | Random split, 5 lớp | 90% | bỏ 1 lớp |
| Raw waveform CNN (2023) | Audio | Speaker-indep. ✓ | 69.72% | Mốc trung thực |
| DistilHuBERT (2025) | Audio | Speaker-indep. ✓ | 70.64% | Mốc trung thực |
| Nature hybrid fusion (2026) | A+V | Speaker-indep. ✓ | ~80%+ | Mốc trung thực A-V |
| **PAM-VLM (đề xuất)** | A+V | Actor-indep. ✓ | pending | Mục tiêu ~80-85% |

**Kết luận quan trọng:** Mốc thật trên actor-independent split là **70–85%** (không phải 89–95%).

---

## 3. Phương pháp đề xuất: PAM-VLM

**PAM = Perception-Aligned Multimodal.** Ý tưởng cốt lõi: thay vì dùng nhãn cứng
(một từ "angry"), học từ **phân bố phiếu bầu của con người theo từng modality**
(Label Distribution Learning), phù hợp với cách con người thực sự cảm nhận cảm xúc.

### 3.1 Kiến trúc (3 tầng)

```
TẦNG 1 — ENCODING
audio.wav ──► WavLM-base-plus (frozen, 94M params)
               └─► projector linear (768→256) ──► audio tokens [B, T_a, 256]

video.flv ──► face detect (OpenCV Haar) + crop + align
               └─► EfficientNet-B0 (frozen, 5.3M params, per-frame)
               └─► 2-layer Transformer (hidden=256, heads=4) ──► video tokens [B, T_v, 256]

TẦNG 2 — FUSION (nguồn metric chính)
audio tokens + video tokens
──► Bidirectional Cross-Attention (audio queries video; video queries audio)
──► Gated pooling ──► fusion embedding [B, 256]
──► audio head  ──► audio_logits  [B, 6]   ← supervise với votes_voice
──► video head  ──► video_logits  [B, 6]   ← supervise với votes_face
──► fusion head ──► fusion_logits [B, 6]   ← supervise với votes_av
                    └─► ĐÂY LÀ NGUỒN METRIC CHÍNH (deterministic)

TẦNG 3 — VLM RECONCILER (optional, tắt được)
disagreement = KL(p_audio ‖ p_video)   ← đo mâu thuẫn giữa 2 modality
fusion_embedding + disagreement signal
──► projector ──► Qwen2.5-0.5B-Instruct + LoRA
──► constrained output: "<emotion>angry</emotion>" + lý do ngắn
    (chỉ kích hoạt khi disagreement cao; dùng cho giải thích + zero-shot)
```

### 3.2 Soft-label (Label Distribution Learning) — đóng góp chính

Thay vì cross-entropy với nhãn cứng, dùng **KL divergence** với phân bố phiếu:

```
soft_voice[clip] = normalize(vote_counts_queryType1 + α)   # α=Laplace smoothing
soft_face[clip]  = normalize(vote_counts_queryType2 + α)
soft_av[clip]    = normalize(vote_counts_queryType3 + α)

total_loss = 0.6 × KL(fusion_logits ‖ soft_av)
           + 0.2 × KL(audio_logits  ‖ soft_voice)
           + 0.2 × KL(video_logits  ‖ soft_face)
           + λ × LM_loss  (chỉ khi bật VLM branch)
```

Mỗi hạng loss được log riêng. Mỗi hạng có cờ ablation tắt/bật.

### 3.3 Vai trò VLM (nói thẳng để báo cáo đúng)

VLM **KHÔNG** được kỳ vọng tăng accuracy in-domain.
Vai trò thật của nó:
1. **Hòa giải mâu thuẫn**: khi audio nói "giận" nhưng mặt nói "trung tính", VLM lập luận và chốt có giải thích — điều classifier tuyến tính không làm được.
2. **Zero-shot transfer**: đổi prompt → chạy trên task khác (RAVDESS, sentiment...) mà không train lại. Classifier head khóa cứng 6 lớp, không transfer được.
3. **Explainability**: sinh ra lý do bằng ngôn ngữ tự nhiên.

---

## 4. Dữ liệu đã build (thực tế, không ước tính)

| | Số clip | Số actor |
|---|---|---|
| Train | 5,721 | 70 |
| Validation | 820 | 10 |
| Test | 901 | 11 |
| **Tổng** | **7,442** | **91** |

- Split: actor-independent, seed=42, assert disjoint — **KHÔNG có actor rò rỉ**
- Soft labels: coverage **100%** (không clip nào thiếu phiếu bầu, fallback=0)
- Manifest checksum đã lưu để reproducibility

---

## 5. Giao thức thí nghiệm (đã khóa cứng)

**Metrics chính:** Macro-F1, UAR (= macro recall)
**Metrics phụ:** Accuracy/WAR, per-class P/R/F1, confusion matrix, ECE (calibration)
**Với nhánh VLM:** invalid-format rate, refusal/empty rate, avg token count

**Baselines sẽ chạy (theo thứ tự):**
1. Audio-only (WavLM + linear head)
2. Video-only (EfficientNet + temporal + linear head)
3. Fusion hard-label (CE loss)
4. Fusion soft-label (KL loss) ← ablation từng hạng
5. Fusion + VLM reconciler (LoRA)

**Ablations:**
- Mỗi hạng loss tắt/bật (3 cờ)
- Encoder frozen vs fine-tuned
- VLM branch bật/tắt

**Hardware:** A100 (SLURM cluster, partition `gpu`). Job pam-fusion (JobID 76361)
đang trong hàng đợi. Login node không có GPU — toàn bộ training qua sbatch.

---

## 6. Cấu hình training thực tế

```yaml
epochs: 30
batch_size: 8              # effective batch = 8 × 4 accum = 32
gradient_accumulation: 4
amp: true                  # mixed precision
early_stopping_patience: 6
primary_metric: macro_f1
optimizer: AdamW (lr=3e-4, weight_decay=0.01)
audio_encoder: microsoft/wavlm-base-plus  (FROZEN)
video_encoder: efficientnet_b0            (FROZEN)
num_frames: 16
frame_size: 224
face_detector: opencv_haar (offline, không cần internet)
```

---

## 7. Số params (ước tính)

| Thành phần | Params | Trainable? |
|---|---|---|
| WavLM-base-plus | ~94M | Frozen (0 trainable) |
| EfficientNet-B0 | ~5.3M | Frozen (0 trainable) |
| Audio projector + head | ~0.2M | ✓ |
| Video temporal + head | ~0.5M | ✓ |
| Fusion cross-attn + heads | ~0.4M | ✓ |
| **Classifier total trainable** | **~1.1M** | ✓ |
| Qwen2.5-0.5B (VLM) | ~500M | LoRA only (~2-5M) |

---

## 8. Kỳ vọng kết quả (để viết Expected Results)

| Run | Macro-F1 kỳ vọng | Lý do |
|---|---|---|
| Audio-only | ~65–72% | Ngưỡng speaker-indep. literature |
| Video-only | ~55–65% | Video yếu hơn audio trên CREMA-D |
| Fusion hard-label | ~75–82% | Baseline chính |
| Fusion soft-label | ~77–84% | +calibration, +F1 nhờ nhãn mềm |
| + VLM reconciler | tương đương hoặc +1–2% trên ca mâu thuẫn | Không kỳ vọng tăng tổng |
| Zero-shot cross-task | ~30–50% | Đo generalization, không kỳ vọng cao |

---

## 9. Figures cần vẽ cho báo cáo

| Figure | Nội dung | Trạng thái |
|---|---|---|
| Fig. 1 | Sơ đồ kiến trúc PAM-VLM (3 tầng) | **CẦN VẼ** (TikZ hoặc placeholder) |
| Fig. 2 | Ví dụ soft-label: 1 clip — phân bố phiếu 3 điều kiện | **CẦN VẼ** (bar chart) |
| Fig. 3 | Confusion matrix (sau khi có kết quả) | **PLACEHOLDER** |
| Fig. 4 | Reliability diagram (ECE) | **PLACEHOLDER** |

---

## 10. Các paper cần cite

1. CREMA-D original: Cao et al. (2014), IEEE TAC — dataset paper
2. WavLM: Chen et al. (2022), IEEE JSTSP — audio encoder
3. EfficientNet: Tan & Le (2019), ICML — video encoder backbone
4. Label Distribution Learning: Geng (2016), IEEE TPAMI — lý thuyết soft-label
5. LoRA: Hu et al. (2022), ICLR — fine-tuning LLM
6. Qwen2.5: Qwen Team (2024) — LLM base
7. AVT-CA (2024, arXiv 2407.18552) — SOTA AV-MER để compare
8. Nature hybrid fusion (2026) — speaker-indep. AV-MER để compare
