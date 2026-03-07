# 🧠 Computer-Aided-Diagnosis-of-Chest-Diseases-from-X-ray-Images-Using-Artifical-Intelligence

### Hybrid Vision Transformer System for Chest X-ray Analysis

An **AI-powered radiology assistant** that analyzes chest X-ray images using a **Hybrid Vision Transformer architecture (DeiT + Swin Transformer)** to detect multiple thoracic pathologies.

The system performs:

* AI-based **disease prediction**
* **Explainable AI heatmaps**
* **Clinical risk assessment**
* **Automated PDF radiology reports**
* **AI patient guidance**

⚠️ **Disclaimer:** This project is an AI research tool and **not a certified medical diagnostic system**. All predictions require clinical validation by qualified healthcare professionals.

---

# 📌 Features

## 🔬 AI Disease Detection

Detects **15 chest diseases** using hybrid deep learning.

Example conditions:

* Atelectasis
* Cardiomegaly
* Effusion
* Pneumonia
* Pneumothorax
* Tuberculosis
* Edema
* Consolidation
* Nodule
* Mass
* Infiltration
* COVID-19
* NORMAL

---

## 🧠 Hybrid Transformer Architecture

The system uses **two Vision Transformers simultaneously**.

| Model          | Role                          |
| -------------- | ----------------------------- |
| **DeiT-Small** | Global feature extraction     |
| **Swin-Tiny**  | Hierarchical spatial learning |

Predictions are fused using:

```
Final Probability =
0.5 × DeiT + 0.5 × Swin
```

This hybrid architecture improves robustness and prediction stability.

---

## 🔥 Explainable AI Heatmaps

The system generates **attention heatmaps** showing where the AI model focuses in the X-ray image.

Techniques used:

* Gradient-based saliency maps
* Vision Transformer attention

Output example:

```
Original X-ray
     +
AI Attention Map
```

---

## 📊 Medical Visualization

Automatically generates clinical graphs:

* Disease probability chart
* Normal vs disease distribution
* Risk level visualization

Graphs are saved in:

```
static/graphs/
```

---

## 📄 Automated Radiology Report

The system generates a **professional 2-page PDF report** containing:

* Patient information
* AI predictions
* Risk assessment
* Radiology interpretation
* Heatmap visualization
* AI lifestyle guidance
* Clinical recommendations

Reports are stored in:

```
static/reports/
```

---

## 🤖 AI Medical Guidance

Optional AI guidance powered by **OpenRouter API**.

Provides:

* Patient-friendly explanation
* Diet suggestions
* Lifestyle recommendations
* Warning signs

Fallback explanations are used if the API is unavailable.

---

# 🏗️ System Architecture

```
             X-ray Image
                  │
                  ▼
         Image Preprocessing
                  │
                  ▼
        ┌───────────────────┐
        │   DeiT-Small      │
        └───────────────────┘
                  │
                  ▼
        ┌───────────────────┐
        │   Swin-Tiny       │
        └───────────────────┘
                  │
                  ▼
       Probability Fusion (50/50)
                  │
                  ▼
           Final Prediction
                  │
                  ▼
        Explainable AI Heatmap
                  │
                  ▼
       Graphs + PDF Report
```

---

# 🧰 Tech Stack

## Backend

* Python
* Flask

## Deep Learning

* PyTorch
* timm (PyTorch Image Models)

## AI Models

* DeiT-Small Vision Transformer
* Swin-Tiny Transformer

## Image Processing

* OpenCV
* Pillow
* NumPy

## Visualization

* Matplotlib

## PDF Generation

* ReportLab

## AI Assistant

* OpenRouter API

## Environment

* python-dotenv

---

# 📁 Project Structure

```
X-ray/
│
├── app.py
├── models.py
├── services.py
├── utils.py
│
├── models/
│   ├── deit_small_15_classes.pth
│   └── swin_ultra_fast_15_classes.pth
│
├── static/
│   ├── heatmaps/
│   ├── graphs/
│   └── reports/
│
├── uploads/
│
├── templates/
│   └── index.html
│
├── requirements.txt
└── README.md
```

---

# ⚙️ Installation

## 1️⃣ Clone Repository

```bash
git clone https://github.com/yourusername/ai-radiology-assistant.git
cd ai-radiology-assistant
```

---

## 2️⃣ Create Virtual Environment

```
python -m venv venv
```

Activate:

Windows

```
venv\Scripts\activate
```

Linux / Mac

```
source venv/bin/activate
```

---

## 3️⃣ Install Dependencies

```
pip install -r requirements.txt
```

---

# 🔑 Environment Variables

Create `.env` file.

Example:

```
OPENROUTER_API_KEY=your_api_key_here
MAX_UPLOAD_SIZE=16777216
MODEL_MODE=hybrid
```

---

# 🚀 Run the Application

Start the Flask server:

```
python app.py
```

Open browser:

```
http://127.0.0.1:5000
```

---

# 📊 Training the Models

Two models are trained separately:

### DeiT Training

```
train_deit.py
```

### Swin Training

```
train_swin.py
```

Dataset requirements:

* Image size: **224 × 224**
* Classes: **15**
* Structure:

```
dataset/
 ├── train/
 └── val/
```

---

# 📈 Expected Performance

| Model         | Accuracy |
| ------------- | -------- |
| DeiT-Small    | ~90-94%  |
| Swin-Tiny     | ~92-96%  |
| Hybrid Fusion | ~94-97%  |

---

# 🔌 API Endpoints

| Endpoint          | Purpose             |
| ----------------- | ------------------- |
| `/`               | Web interface       |
| `/analyze`        | Run AI inference    |
| `/download`       | Download PDF report |
| `/api/symptoms`   | Symptom options     |
| `/api/model-info` | Model details       |
| `/health`         | System health       |

---

# ⚠️ Limitations

* Not FDA / CE certified
* Not intended for clinical decision making
* Requires radiologist confirmation
* May miss subtle findings

---

# 👨‍⚕️ Author

**AI Radiology Research Project**

Lead Radiologist (simulated)

Dr. Alexander V. Sterling, MD, PhD

---

# 📜 License

MIT License

---

# ⭐ Acknowledgements

Libraries and tools used:

* PyTorch
* timm
* Flask
* OpenCV
* ReportLab
* OpenRouter API

---

# 🚀 Future Improvements

* DICOM support
* PACS integration
* Mobile interface
* Multi-modal clinical AI
* Faster inference using TensorRT

---

**AI Radiology Assistant — Hybrid Vision Transformer System**
