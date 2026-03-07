# ⚙️ Setup Instructions

## AI Radiology Assistant – Hybrid Transformer System

These instructions will help you **install, configure, and run the AI X-ray analysis system locally**.

---

# 1️⃣ Prerequisites

Make sure your system has the following installed:

### Required Software

* **Python 3.9 – 3.11**
* **Git**
* **pip**

Check installation:

```bash
python --version
pip --version
```

---

# 2️⃣ Download the Project

If using Git:

```bash
git clone https://github.com/yourusername/xray-ai-system.git
cd xray-ai-system
```

Or download the ZIP and extract it.

Example folder:

```
X-ray/
```

---

# 3️⃣ Create Virtual Environment

Navigate to the project folder.

```bash
cd X-ray
```

Create environment:

```bash
python -m venv venv
```

---

# 4️⃣ Activate Virtual Environment

### Windows

```bash
venv\Scripts\activate
```

### Linux / Mac

```bash
source venv/bin/activate
```

After activation you should see:

```
(venv)
```

---

# 5️⃣ Install Dependencies

Install all required libraries:

```bash
pip install -r requirements.txt
```

This installs:

* Flask
* PyTorch
* timm
* OpenCV
* ReportLab
* Matplotlib
* OpenRouter API
* NumPy
* Pillow

---

# 6️⃣ Download Trained Models

Place trained model files inside the **models folder**.

Example:

```
X-ray/
│
├── models/
│   ├── deit_small_15_classes.pth
│   └── swin_ultra_fast_15_classes.pth
```

If the folder does not exist:

```
mkdir models
```

---

# 7️⃣ Create `.env` File

Create a `.env` file in the project root.

Example:

```
X-ray/.env
```

Add configuration:

```
OPENROUTER_API_KEY=your_openrouter_api_key
MAX_UPLOAD_SIZE=16777216
MODEL_MODE=hybrid
```

Explanation:

| Variable           | Purpose              |
| ------------------ | -------------------- |
| OPENROUTER_API_KEY | AI guidance API      |
| MAX_UPLOAD_SIZE    | Max image size       |
| MODEL_MODE         | hybrid / deit / swin |

---

# 8️⃣ Project Folder Structure

Your project should look like this:

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
├── uploads/
├── static/
│   ├── heatmaps/
│   ├── graphs/
│   └── reports/
│
├── templates/
│   └── index.html
│
├── requirements.txt
├── README.md
└── .env
```

---

# 9️⃣ Run the Application

Start the Flask server.

```bash
python app.py
```

You should see:

```
Hybrid Transformer Engine Activated
Flask running on http://127.0.0.1:5000
```

---

# 🔟 Open the Web Interface

Open browser:

```
http://127.0.0.1:5000
```

You can now:

* Upload X-ray images
* Run AI analysis
* View heatmaps
* Generate PDF reports

---

# 🧪 Testing the System

Upload a **chest X-ray image**.

The system will:

1️⃣ preprocess image
2️⃣ run hybrid AI model
3️⃣ generate predictions
4️⃣ create heatmap
5️⃣ generate graphs
6️⃣ produce PDF report

---

# 📄 Generated Files

The system automatically creates:

| Folder           | Purpose               |
| ---------------- | --------------------- |
| uploads/         | uploaded images       |
| static/heatmaps/ | AI attention maps     |
| static/graphs/   | probability graphs    |
| static/reports/  | generated PDF reports |

---

# ⚡ Performance Tips

For faster inference:

### Enable GPU

Install CUDA PyTorch:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

---

### Recommended Hardware

| Component | Recommended |
| --------- | ----------- |
| RAM       | 16GB        |
| GPU       | NVIDIA RTX  |
| CPU       | 6+ cores    |

---

# ❗ Troubleshooting

### Error: Model Not Found

Ensure models exist:

```
models/deit_small_15_classes.pth
models/swin_ultra_fast_15_classes.pth
```

---

### Error: Module Not Found

Install dependencies again:

```
pip install -r requirements.txt
```

---

### Error: OpenRouter API

Check `.env` file.

---

# 🚀 System Ready

Once running you will have a complete:

```
AI Radiology Hybrid Transformer System
```

with:

* Hybrid Vision Transformer
* Explainable AI heatmaps
* Clinical risk analysis
* Automated radiology reports

---

**AI Radiology Assistant – Hybrid Transformer System**
