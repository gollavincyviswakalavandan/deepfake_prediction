Use the commands to run the project


cd "Current folder path"

chmod +x run_pipeline.sh

./run_pipeline.sh

----------------------------------------------------------------------------------------------


# 🎥 Uncertainty-Aware DeepFake Video Detection

## 📌 Overview

This project presents an advanced DeepFake video detection system that combines:

- Frame-level CNN classification
- Monte Carlo Dropout for uncertainty estimation
- Temporal suspicious segment detection
- Grad-CAM explainability for manipulated region localization

The system not only detects whether a video is fake, but also:

✔ Identifies suspicious time segments  
✔ Estimates prediction uncertainty  
✔ Visualizes manipulated regions  

This makes it suitable for forensic and research applications.

---

## 🚀 Features

- 🧠 CNN-based DeepFake Detection
- 📊 Monte Carlo Dropout (Uncertainty Estimation)
- 📈 Temporal Probability Graph
- ⚠ Suspicious Frame Detection
- 🔥 Grad-CAM Visualization
- 📁 CSV Report Generation

---

## 🏗 Project Structure


deepfake_project/
│
├── train_model.py
├── split_dataset.py
├── advanced_video_analysis.py
├── gradcam_visualize.py
│
├── model/
│ └── deepfake_model.h5
│
├── dataset/
│ ├── train/
│ ├── val/
│ └── test/
│
├── videos/
│ └── test_video.mp4
│
├── gradcam_output/
├── probability_graph.png
├── uncertainty_graph.png
└── video_analysis_results.csv


---

## 🧪 Methodology

### 1️⃣ Frame-Level Classification
Each video frame is resized to 224x224 and passed through a CNN model trained for binary classification (Real vs Fake).

### 2️⃣ Monte Carlo Dropout
Multiple forward passes (T=10) are performed to compute:
- Mean prediction probability
- Predictive variance (uncertainty)

### 3️⃣ Temporal Suspicious Detection
Frames exceeding a probability threshold (0.7) are marked as suspicious.
Time segments are calculated using FPS.

### 4️⃣ Grad-CAM Explainability
Grad-CAM is used to highlight manipulated facial regions in suspicious frames.

---

## 📊 Outputs

- Probability over time graph
- Uncertainty over time graph
- Suspicious time segment detection
- Grad-CAM heatmap images
- CSV analysis report

---

## ⚙ Installation

```bash
pip install tensorflow opencv-python numpy matplotlib pandas scikit-learn
▶ How To Run
Train Model
python train_model.py
Split Dataset
python split_dataset.py
Video Analysis
python advanced_video_analysis.py
Grad-CAM Visualization
python gradcam_visualize.py
📈 Example Results

ROC-AUC Score

Brier Score

Suspicious segment detection

Visual explanation via Grad-CAM

🧠 Future Improvements

Replace CNN with EfficientNet

Add LSTM for temporal modeling

Integrate Fake News text detection

Deploy as web application

👩‍💻 Author

Nandini Mattey

📄 License

This project is for research and educational purposes.


---

# 🎯 Why This README Is Strong

It shows:

✔ Clear project explanation  
✔ Professional structure  
✔ Methodology breakdown  
✔ Execution steps  
✔ Research direction  

Recruiters love this format.

---

If you want, I can also give:

🔥 Resume project description (2–3 lines)  
🔥 LinkedIn project description  
🔥 Research paper PDF template  
🔥 GitHub portfolio optimization tips  

Cheppu 👑
