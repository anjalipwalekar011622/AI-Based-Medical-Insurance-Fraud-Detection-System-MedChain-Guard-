# 🛡️ MedChain Guard — AI Fraud Detection Prototype

Hackathon-ready. Backend + Frontend. One core feature. Demo-ready in under 10 minutes.

---

##  Quick Start (2 steps)

### Step 1 — Start the Backend
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

### Step 2 — Open the Frontend
```bash
# Just open the HTML file in your browser
open frontend/index.html
# OR serve it:
cd frontend && python -m http.server 3000
# Then visit http://localhost:3000
```

---

##  Demo Flow (for judges)

1. Click **🟢 Normal Claim** → submit → shows ✅ LEGITIMATE, low risk score
2. Click **🔴 Fraudulent** → submit → shows ⛔ FRAUD DETECTED with red alert + risk flags
3. Click **🟡 Borderline** → submit → shows ⚠️ SUSPICIOUS with yellow warning

---

##  Architecture

```
Browser (HTML/CSS/JS)
       │
       │ POST /predict
       ▼
FastAPI Backend (Python)
  ├── Isolation Forest (sklearn)
  ├── Rule-based flag engine
  └── Risk score 0–100
```

##  Features Used for Detection
| Feature | Why it matters |
|---|---|
| Claim Amount | Unusually high = suspicious |
| Treatment Frequency | Repeated visits = possible fraud |
| No. of Procedures | Too many = billing fraud |
| Days Since Last Claim | Very recent = duplicate claim |
| Provider Claim Count | High volume = fraudulent provider |

##  Key Files
- `backend/main.py` — FastAPI server + ML model
- `frontend/index.html` — Single-file UI (no build needed)

---

##  Talking Points for Presentation
- **AI Model**: Isolation Forest detects anomalies without labeled training data
- **Rule Engine**: Domain-specific rules add interpretability
- **Blockchain concept**: Each verified claim gets a tamper-proof hash (simulated)
- **Scalability**: Can plug in XGBoost or real blockchain (Hyperledger) for production

---

*Built for hackathon demo purposes.*
