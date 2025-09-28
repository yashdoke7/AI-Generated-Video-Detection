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


<img width="1632" height="711" alt="Arch Diagram_final drawio" src="https://github.com/user-attachments/assets/968b201a-5792-4dad-9885-8cd5060e915b" />


### 🗂️ Repository Structure

```
AI-Generated-Video-Detection/
│── data/          # manifests
│── models/        # UNITE & UNITE-CL implementations
│── train/         # Training scripts
│── eval/          # Evaluation scripts & metrics
│── notebooks/     # Jupyter/Colab notebooks for experimentation
│── results/       # Graphs, metrics, and sample outputs
│── requirements.txt
│── README.md
```

---

### 📊 Datasets Used

* **Training / Pretraining**: OpenVid-1M, VidProM samples, Stable Video Diffusion synthetic data.
* **Evaluation / Benchmarks**: FaceForensics++, DFDC (preview + full), Celeb-DF, DeeperForensics, WildDeepfake, UADFV, AEGIS, SHAM, UCF101.
  *(~80GB total, not included in repo — links provided separately)*

---

### 🚀 Getting Started

1. **Clone the repository**

   ```bash
   git clone https://github.com/<your-username>/AI-Generated-Video-Detection.git
   cd AI-Generated-Video-Detection
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Run demo**

   ```bash
   python demo.py --video sample.mp4
   ```

   Example Output:

   ```
   Prediction: 73% AI-generated
   ```

---

### 📈 Results (To Be Updated)

* ✅ Higher accuracy on both old and new datasets compared to baseline CL methods.
* ✅ Improved AUC and F1-score retention under continual updates.
* ✅ Strong cross-dataset performance showing real-world robustness.
* 📊 Results and plots (loss curves, ROC, confusion matrices).

---

### 🛠️ Future Work

* Add More Synthetic data for the model to train.
* Integrating real-time detection pipelines for streaming platforms.
* Scaling continual learning for unseen generative techniques.
* Exploring lightweight deployment for edge devices.

---

### 📜 License

MIT License

---

This version tells interviewers at a glance: *“This isn’t just another implementation; it solves a known problem (CL accuracy drop) and beats expectations.”*

Do you want me to also craft a **short “Key Innovations” block** (like 3 bullets) you could pin at the top of the README for extra emphasis?
