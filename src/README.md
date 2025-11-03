# 🔊 SignalClassifier-FFT-ML

Classifies synthetic signals by converting them to the **frequency domain** (FFT) and training a simple ML classifier on the largest spectral magnitudes

Includes a **Flask API** for inference and a **Docker** image for deployment

---

## 🧩 Pipeline
1. Generate labeled signals (e.g., 770 Hz vs 785 Hz).  
2. Compute FFT → extract top-K magnitudes.  
3. Train Logistic Regression → report accuracy & confusion matrix.  
4. Visualize FFT examples.  
5. Serve predictions via `/predict` (Flask API).

---

## ▶️ Quick Start (Local)

```bash
python -m venv .venv
source .venv/Scripts/activate   # Windows Git Bash
# source .venv/bin/activate     # macOS/Linux

pip install -r requirements.txt
python src/generate_signals.py
python src/extract_fft_features.py
python src/train_classifier.py
python src/visualize_results.py
