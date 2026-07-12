# 🛡️ SEAD-AI — Social Engineering Attack Detection System

![Python](https://img.shields.io/badge/Python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.3-orange)
![BERT](https://img.shields.io/badge/BERT-bert--base--uncased-green)
![Flask](https://img.shields.io/badge/Flask-3.0-lightgrey)
![Accuracy](https://img.shields.io/badge/Accuracy-99%25-brightgreen)
![License](https://img.shields.io/badge/License-MIT-yellow)

> An AI-powered real-time detection system that identifies
> phishing and social engineering attacks using
> BERT + Psychological Influence Analysis.

---

## 🎯 What is SEAD-AI?

Social engineering attacks are responsible for **90% of all
cyber breaches** worldwide. Traditional rule-based systems
fail against modern AI-generated phishing attacks.

**SEAD-AI** solves this by combining:
- 🧠 **Fine-tuned BERT** — detects language manipulation patterns
- ⚡ **PsychoGuard Engine** — detects Cialdini's 6 Influence Principles
- 🔍 **Explainable AI** — tells users exactly WHY a message is suspicious
- 📊 **Real-time scoring** — combined threat score in milliseconds

---

## 🆕 Novel Contribution — PsychoGuard

The first known system to encode **Cialdini's 6 Principles
of Influence** as machine learning features:

| Principle | What It Detects |
|-----------|----------------|
| Authority | Fake CEO/IRS/Bank impersonation |
| Urgency | "Act now or account deleted" |
| Scarcity | "Only 3 spots left!" |
| Social Proof | "Millions of users trust us" |
| Reciprocity | Fake gifts to create obligation |
| Liking | False rapport and flattery |

---

## 🚀 Live Demo

> Coming soon on Render.com

---

## 📸 Screenshots

### Main Analyzer
Paste any message → Get instant threat analysis

### Results Dashboard
- Threat Score meter (0-100%)
- Psychology triggers detected
- Explainable AI reasons
- Recommended action

---

## 🏗️ System Architecture
---

## 📊 Performance Results

| Metric | Sample Data | Full Dataset |
|--------|-------------|--------------|
| Accuracy | 83.33% | **99.05%** |
| Precision | 75.00% | **99.05%** |
| Recall | 100.00% | **99.05%** |
| F1 Score | 85.71% | **99.05%** |
| Dataset Size | 30 rows | 23,467 rows |

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| AI Model | BERT (bert-base-uncased) |
| Framework | PyTorch + HuggingFace |
| Backend | Flask (Python) |
| Database | SQLite |
| Frontend | HTML + CSS + JavaScript |
| GPU | NVIDIA CUDA |

---

## 📁 Project Structure
---

## ⚡ Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/sead-ai.git
cd sead-ai
```

### 2. Create virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download datasets
```bash
python datasets/download_datasets.py
```

### 5. Preprocess data
```bash
python preprocessing/clean.py
```

### 6. Train BERT model
```bash
python models/train_bert.py
```

### 7. Run the app
```bash
python app.py
```

Open: **http://localhost:5000**

---

## 🐳 Docker Deployment

```bash
docker-compose up --build
```

Open: **http://localhost:5000**

---

## 🧪 Run Tests

```bash
python tests/test_all.py
```

Expected:

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | /analyze | Analyze a message |
| GET | /history | Scan history |
| GET | /stats | Dashboard stats |
| GET | /search?q= | Search history |
| GET | /health | System health |

### Example API Call
```python
import requests

response = requests.post("http://localhost:5000/analyze",
    json={"text": "URGENT: Your account is suspended!"})

print(response.json())
# {
#   "threat_score": 78,
#   "risk_level": "High Risk",
#   "reasons": ["Urgency tactics detected", ...]
# }
```

---

## 📚 Datasets Used

| Dataset | Size | Type |
|---------|------|------|
| CEAS 2008 | 32,204 rows | Phishing |
| Enron Emails | 15,000 rows | Legitimate |
| Nazario Corpus | 5,000 rows | Phishing |
| Nigerian Fraud | 5,000 rows | Phishing |
| SpamAssassin | 6,000 rows | Mixed |

**Total: 23,467 training samples**

---

## 📖 Research References

1. Fakhouri et al. — AI-Driven Solutions for Social Engineering (IEEE, 2024)
2. Wang et al. — Adversarial AI Detection (IJST, 2025)
3. Cialdini, R. — Influence: The Psychology of Persuasion (1984)
4. Thorne — Cognitive Vulnerabilities in the Age of LLMs (TAJET, 2025)

---

## 👨‍💻 Author

**B.Tech Cybersecurity Research Project**
Built as part of IBM Skills project — 2025

---

## 📄 License

MIT License — free to use and modify.