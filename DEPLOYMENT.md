# SEAD-AI — Deployment Guide (Phase 13)

## Option 1 — Run Locally (Recommended for Demo)

### Step 1 — Setup
```powershell
cd D:\SEAD_AI_Phase13\sead_ai
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### Step 2 — Prepare Data & Train Model
```powershell
python datasets/download_datasets.py
python preprocessing/clean.py
python models/train_bert.py
```

### Step 3 — Run App
```powershell
python app.py
```

Open browser: **http://localhost:5000**

---

## Option 2 — Docker Deployment

### Requirements
- Install Docker Desktop from docker.com

### Run
```bash
# Build and start
docker-compose up --build

# Run in background
docker-compose up -d

# Stop
docker-compose down
```

Open browser: **http://localhost:5000**

---

## Option 3 — Deploy to Render (Free Cloud)

1. Push project to GitHub
2. Go to render.com → New → Web Service
3. Connect your GitHub repo
4. Render auto-detects render.yaml
5. Click Deploy

**Your app will be live at:**
`https://sead-ai.onrender.com`

---

## Option 4 — Deploy to Heroku

```bash
# Install Heroku CLI, then:
heroku login
heroku create sead-ai-app
git push heroku main
heroku open
```

---

## Production Checklist

| Item | Status |
|------|--------|
| BERT model trained | Run `python models/train_bert.py` |
| Dataset preprocessed | Run `python preprocessing/clean.py` |
| Tests passing | Run `python tests/test_all.py` |
| Flask debug OFF | Set `FLASK_DEBUG=False` in .env |
| Secret key set | Change SECRET_KEY in .env |
| Database initialized | Auto-done on startup |

---

## Folder Structure (Final)
```
sead_ai/
├── app.py                ← Flask entry point
├── config.py             ← All settings
├── wsgi.py               ← Production WSGI
├── Procfile              ← Heroku/Render config
├── Dockerfile            ← Docker config
├── docker-compose.yml    ← Docker Compose
├── requirements.txt      ← Dependencies
├── DEPLOYMENT.md         ← This file
│
├── datasets/             ← Phase 2
├── preprocessing/        ← Phase 3
├── models/               ← Phase 4
├── predictor/            ← Phase 5, 7, 8
├── psychology/           ← Phase 6
├── database/             ← Phase 9
├── templates/            ← Phase 11
├── static/               ← Phase 11
├── tests/                ← Phase 12
└── utils/                ← Phase 1
```
