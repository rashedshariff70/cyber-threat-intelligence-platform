# 🛡️ CyberGuard — Intelligent Threat Intelligence Platform

## Project Title
**CyberGuard: Intelligent Threat Intelligence Platform**  
*AI-powered URL Phishing Detection · Network Attack Classification · LLM Security Assistant*

---

## Setup Instructions

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Create your `.env` file
```
GROQ_API_KEY=gsk_XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
```

### 3. Place your trained model files in the project root
- `model.pkl` — URL phishing Random Forest model
- `lgbm_model.pkl` — LightGBM attack classifier
- `lgbm_scaler.pkl` — StandardScaler for cyber features
- `lgbm_label_encoder.pkl` — LabelEncoder for attack classes

### 4. Run the app
```bash
python app.py
```

### 5. Open in browser
```
http://localhost:5000
```

---

## Project Structure
```
cyberguard/
├── app.py                     # Main Flask application
├── requirements.txt
├── .env                       # GROQ_API_KEY
├── model.pkl                  # URL phishing model
├── lgbm_model.pkl             # LightGBM cyber attack model
├── lgbm_scaler.pkl
├── lgbm_label_encoder.pkl
└── templates/
    ├── base.html              # Shared navbar + layout
    ├── home.html              # Landing page + quick scan
    ├── about.html             # Project overview
    ├── models.html            # Confusion matrix + charts
    ├── predict_url.html       # URL phishing prediction
    ├── predict_cyber.html     # Cyber attack prediction
    └── predict_chatbot.html   # LLM Q&A chatbot
```

---

## Module Summaries

### Module 1 — URL Phishing Detection (Random Forest)
- **Accuracy**: 99.97%
- **Features**: 14 (entropy, URL length, suspicious keywords, brand impersonation, etc.)
- **Dataset**: Custom phishing URL dataset (~48,820 samples, balanced)

### Module 2 — Cyber Attack Detection (LightGBM)
- **Accuracy**: 99.16%
- **Classes**: Backdoor, DoS, Generic, Probe, Worm
- **Dataset**: UNSW-NB15 (~340,000 samples)
- **Best vs**: XGBoost (98.78%), Random Forest (98.25%), Decision Tree (97.67%)

### Module 3 — AI Assistant (Groq Llama 3.3 70B)
- Real-time Q&A about the project, models, and cybersecurity concepts
- Context-aware multi-turn conversations
- Pre-loaded suggested questions for oral exam practice
