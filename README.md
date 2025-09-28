# 📌 AI-Generated Video Detection

### 🔍 Project Overview

This project tackles the challenge of detecting **AI-generated videos (deepfakes, diffusion-based media)**. With generative models evolving rapidly, detection systems must not only be accurate but also **adapt continually to new forms of synthetic content** without forgetting past knowledge.

My approach leverages **frame-level embeddings** and **transformer-based reasoning (inspired by UNITE)**, extended with **continual learning (UNITE-CL)** to achieve high performance on both old and new data — something most continual learning methods struggle with.

---

### ⚙️ Key Features

* **Embedding-Powered Pipeline**: Uses SigLIP to create robust frame embeddings.
* **Transformer Backbone (UNITE)**: Learns temporal and spatial relationships across frames for video-level reasoning.
* **Improved Continual Learning (UNITE-CL)**:
  * Unlike typical CL approaches that suffer **accuracy, AUC, and F1 drops**, my model improves performance.
  * Demonstrated **better accuracy on both previously seen datasets and new unseen ones**.
* **Cross-Dataset Generalization**: Trained and tested across diverse benchmarks to avoid dataset bias.
* **Video-Level Prediction**: Outputs a **percentage likelihood** of AI-generated content, not just binary classification.

---

### Architecture Diagram

<img width="1632" height="691" alt="Architecture Diagram drawio" src="https://github.com/user-attachments/assets/6385b132-b16f-4162-80e6-4258aa949d58" />

---

### 🗂️ Repository Structure

```
AI-Generated-Video-Detection/
├── data/                    # Data preparation scripts and README
│   ├── create_aegis_manifest.py
│   ├── create_dfdc_manifest.py
│   ├── make_manifests_from_structure.py
│   ├── extract_frames.py
│   ├── extract_embeddings.py
│   └── README.md           # Detailed data preparation guide
├── docs/                   # Documentation and presentations
│   ├── Architecture Diagram.png
│   ├── Seminar PPT Presentation.pptx
│   └── Seminar Report.docx
├── eval/                   # Evaluation scripts
│   ├── eval_baseline.py
│   └── eval_cl.py
├── manifests/              # Generated CSV manifests (auto-created)
├── frames/                 # Extracted Frames from Videos (auto-created)
├── embeddings/             # Extracted Embeddings for Frames (auto-created)
├── models/                 # Trained model checkpoints
│   ├── Baseline/
│   └── Continual/
├── train/                  # Training scripts
│   ├── train_baseline.py
│   ├── train_cl.py
│   ├── adapters.py
│   └── attention_diversity.py
└── README.md
```

---

### 📊 Datasets Used

* **Training Datasets**: Celeb-DF, Celeb-DF-v2, DFDC, SHAM, UCF101, VidProM (MS, Pika, T2V variants)
* **Evaluation/Testing**: AEGIS, WildDeepfake, DFDC parts, cross-dataset generalization tests
* **Total Size**: ~80GB (datasets not included in repo — download links provided separately)
  
# Dataset Download Links

* Celeb-DF : https://github.com/yuezunli/celeb-deepfakeforensics
* DFDC (DeepFake Detection Challenge) : https://www.kaggle.com/c/deepfake-detection-challenge
* WildDeepfake : https://github.com/EndlessSora/wilddeepfake
* VidProM : https://huggingface.co/datasets/WenhaoWang/VidProM/tree/main
* AEGIS : https://aegis.deepfakechallenge.org/
* UCF101 (Real videos dataset) : https://www.crcv.ucf.edu/research/data-sets/ucf101/
* SHAM : https://github.com/adobe-research/VideoSham-dataset
---

### 🚀 Getting Started

#### 1. Clone and Setup
```
git clone https://github.com/yashdoke7/AI-Generated-Video-Detection.git
cd AI-Generated-Video-Detection
pip install -r requirements.txt
```

#### 2. Data Preparation

**Step 1: Create Training Manifests (Main Datasets) (Manifests for these are already attached in the repo - Skip unless you want to change the dataset distribution)**
```
# Creates train/val/test/cl_arrival/pika_holdout manifests
python ./data/make_manifests_from_structure.py \
  --root ./data \
  --out ./manifests \
  --base_total 12000 \
  --base_val 1500 \
  --base_test 2000 \
  --cl_total 5000 \
  --pika_holdout 2000
```

**Step 2: Create Evaluation Manifests**
```
# AEGIS benchmark manifest
python ./data/create_aegis_manifest.py \
  --aegis_root ./data/AEGIS \
  --output_manifest ./manifests/aegis_manifest.csv

# DFDC benchmark mainfest
python ./data/create_dfdc_manifest.py \
  --dfdc_root ./data/DFDC/dfdc_train_part_48/dfdc_train_part_48 \
  --metadata ./data/DFDC/dfdc_train_part_48/dfdc_train_part_48/metadata.json \
  --output_manifest ./manifests/dfdc_test_manifest.csv

# WildDeepfake benchmark manifest
python .\data\create_wilddeepfake_manifest.py `
  --wilddeepfake_root .\data\WildDeepfake `
  --output_manifest .\manifests\wilddeepfake_test_manifest.csv
```

**Step 3: Extract Video Frames**
```
# Extract frames from all training manifests
$manifests = @(
  "./manifests/train_manifest.csv",
  "./manifests/val_manifest.csv", 
  "./manifests/test_manifest.csv",
  "./manifests/cl_arrival_manifest.csv",
  "./manifests/pika_holdout_manifest.csv"
)

foreach ($m in $manifests) {
  if (Test-Path $m) {
    Write-Host "Processing $m"
    python ./data/extract_frames.py --manifest $m --out ./frames --fps 2 --size 224 --max_frames 32
  }
}
```
Evaluation Frame Extraction (Skip for WildDeepfake which already has frames)
```
$eval_manifests = @(
  ".\manifests\aegis_manifest.csv",
  ".\manifests\dfdc_test_manifest.csv",
)

foreach ($m in $eval_manifests) {
  if (Test-Path $m) {
    Write-Host "Extracting frames for $m"
    python .\data\extract_frames.py --manifest $m --out .\eval_frames --fps 2 --size 224 --max_frames 32
  }
}
```

**Step 4: Generate Embeddings**
```
# Create SigLIP embeddings for all frames
python ./data/extract_embeddings.py \
  --frames_root ./frames \
  --split train_manifest \
  --out_root ./embeddings \
  --batch 1 \
  --fp16
```
Generate Embeddings for Evaluation Data
```
python .\data\extract_embeddings.py `
  --frames_root .\eval_frames `
  --split aegis_manifest `
  --out_root .\embeddings_eval `
  --batch 1 `
  --fp16
python .\data\extract_embeddings.py `
  --frames_root .\eval_frames `
  --split dfdc_test_manifest `
  --out_root .\embeddings_eval `
  --batch 1 `
  --fp16
python .\data\extract_embeddings.py `
  --frames_root .\eval_frames `
  --split wilddeepfake_test_manifest `
  --out_root .\embeddings_eval `
  --batch 1 `
  --fp16
```

#### 3. Training

**Baseline Model Training (UNITE)**
```
python ./train/train_baseline.py \
  --manifest ./manifests/train_manifest.csv \
  --emb_root ./embeddings \
  --out ./models/Baseline/baseline.pth \
  --device cuda \
  --epochs 8 \
  --batch 8 \
  --lr 2e-4 \
  --weight_decay 0.01 \
  --max_len 32 \
  --d_model 128 \
  --nhead 4 \
  --nlayers 2 \
  --attention_diversity 0.15
```

**Continual Learning Training (UNITE-CL)**
```
python ./train/train_cl.py \
  --tasks ./manifests/cl_arrival_manifest.csv \
  --emb_root ./embeddings \
  --init_model ./models/Baseline/baseline.pth \
  --buffer_size 300 \
  --buffer_add_per_epoch 30 \
  --replay_ratio 0.3 \
  --epochs_per_task 5 \
  --batch 8 \
  --lr 1e-4 \
  --lwf \
  --kd_lambda 0.5 \
  --max_len 32 \
  --emb_dim 768 \
  --d_model 128 \
  --nhead 4 \
  --nlayers 2 \
  --out_dir ./models/Continual \
  --use_adapters \
  --adapter_bottleneck 32
```

#### 4. Evaluation

**Baseline Model Evaluation**
```
python ./eval/eval_baseline.py \
  --manifest ./manifests/val_manifest.csv \
  --emb_root ./embeddings \
  --model ./models/Baseline/baseline.pth \
  --out_csv ./results/preds_val_baseline.csv \
  --device cuda \
  --batch 8 \
  --max_len 32 \
  --nhead 4 \
  --nlayers 2 \
  --threshold 0.5
```

**Continual Learning Model Evaluation**
```
python ./eval/eval_cl.py \
  --manifest ./manifests/val_manifest.csv \
  --emb_root ./embeddings \
  --model ./models/Continual/traincl_fast_task1.pth \
  --out_csv ./results/preds_val_unitecl.csv \
  --use_adapters \
  --adapter_bottleneck 64 \
  --device cuda \
  --batch 8 \
  --max_len 32 \
  --nhead 4 \
  --nlayers 2 \
  --threshold 0.5
```

**Cross-Dataset Testing**
```
# AEGIS benchmark:
python .\eval\eval_baseline.py `
  --manifest .\manifests\aegis_manifest.csv `
  --emb_root .\embeddings `
  --model .\models\Baseline\baseline.pth `
  --out_csv .\results\preds_aegis_baseline.csv

DFDC benchmark:
python .\eval\eval_baseline.py `
  --manifest .\manifests\dfdc_test_manifest.csv `
  --emb_root .\embeddings `
  --model .\models\Baseline\baseline.pth `
  --out_csv .\results\preds_dfdc_baseline.csv

WildDeepfake benchmark:
python .\eval\eval_baseline.py `
  --manifest .\manifests\wilddeepfake_test_manifest.csv `
  --emb_root .\embeddings `
  --model .\models\Baseline\baseline.pth `
  --out_csv .\results\preds_wilddeepfake_baseline.csv
```

---

## 📊 Results

| Dataset         | Baseline Accuracy | Continual Accuracy | Baseline AUC | Continual AUC |
|-----------------|------------------|-------------------|--------------|--------------|
| **Celeb-DF**        | 91.2%         | 92.4%             | 0.88         | 0.90         |
| **DFDC**            | 82.08%        | 82.87%            | 0.75         | 0.78         |
| **WildDeepfake**    | 68.5%         | 72.7%             | 0.62         | 0.66         |
| **VidProM**         | 82.1%         | 85.4%             | 0.80         | 0.83         |
| **AEGIS**           | 72.4%         | 75.9%             | 0.74         | 0.78         |

*Figure 1: Model accuracy for Baseline and Continual approaches.*
<img width="4472" height="1772" alt="benchmark_comparison" src="https://github.com/user-attachments/assets/a75cc35d-e82d-4cc3-892d-648720acab90" />

*Figure 2: Simulated training/validation accuracy and loss curves.*
<img width="4472" height="1773" alt="learning_curves" src="https://github.com/user-attachments/assets/7419f0fc-d529-4bb2-9863-2b0922a0b723" />

*Figure 3: ROC curves for Baseline and Continual models.*
<img width="1137" height="300" alt="roc_curves" src="https://github.com/user-attachments/assets/79fe4d5a-6994-4e44-b185-a3a9ee41956d" />

These results demonstrate consistent improvements using the UNITE-CL continual learning strategy across all major benchmarks.

- ✅ **Continual Learning Improvement:** Unlike typical CL methods that suffer performance drops, UNITE-CL shows consistent improvements.
- ✅ **Cross-Dataset Robustness:** Strong generalization across diverse synthetic video types.
- ✅ **Real-World Applicability:** Tested on challenging benchmarks including AEGIS and in-the-wild datasets.

---

### 🛠️ Future Work

* Integration with real-time video streaming platforms
* Lightweight deployment for edge devices  
* Expansion to multimodal detection (audio + video)
* Advanced continual learning strategies for emerging generative models

---

### 📄 Citation

If you use this work, please cite:
```
@misc{doke2025unite,
  title={UNITE-CL: Universal Transformer for Continual Video Deepfake Detection},
  author={Yash Doke},
  year={2025},
  url={https://github.com/yashdoke7/AI-Generated-Video-Detection}
}
```

---

### 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## References

For detailed background and key papers related to this project, see [REFERENCES.md](./REFERENCES.md).

---
