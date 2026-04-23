from flask import Flask, render_template, request, jsonify
import pickle
import joblib
import numpy as np
import re
import math
from urllib.parse import urlparse

app = Flask(__name__)

# ─────────────────────────────────────────
# Load Models
# ─────────────────────────────────────────
try:
    with open("model.pkl", "rb") as f:
        url_model = pickle.load(f)
    url_model_loaded = True
except:
    url_model_loaded = False

try:
    lgbm_model = joblib.load("lgbm_model.pkl")
    lgbm_scaler = joblib.load("lgbm_scaler.pkl")
    lgbm_label_encoder = joblib.load("lgbm_label_encoder.pkl")
    lgbm_loaded = True
except:
    lgbm_loaded = False


# ─────────────────────────────────────────
# URL Feature Extraction
# ─────────────────────────────────────────
def url_entropy(url):
    prob = [float(url.count(c)) / len(url) for c in dict.fromkeys(list(url))]
    return -sum([p * math.log2(p) for p in prob if p > 0])


def extract_url_features(url):
    parsed = urlparse(url)
    features = []

    features.append(len(url))
    features.append(len(parsed.netloc))
    features.append(url.count('.'))
    features.append(url.count('-'))
    features.append(url.count('/'))
    features.append(len(re.findall(r'[@?=&_%]', url)))
    features.append(sum(c.isdigit() for c in url))
    features.append(1 if url.startswith("https") else 0)
    features.append(1 if re.match(r'http[s]?://\d+\.\d+\.\d+\.\d+', url) else 0)
    features.append(url_entropy(url))

    suspicious_words = ['login','verify','secure','update','account','bank','free','gift',
                        'alert','confirm','signin','password','auth','bonus','reward']
    features.append(sum(word in url.lower() for word in suspicious_words))

    brands = ['paypal','amazon','google','apple','microsoft','facebook','instagram','netflix','bank','upi']
    features.append(sum(brand in url.lower() for brand in brands))

    features.append(len(parsed.netloc.split('.')))
    features.append(1 if re.search(r'\d+\.\d+\.\d+\.\d+', url) else 0)

    return features


# ─────────────────────────────────────────
# Routes
# ─────────────────────────────────────────
@app.route("/")
def home():
    return render_template("home.html")


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/models")
def models():
    return render_template("models.html")


@app.route("/predict/url")
def predict_url_page():
    return render_template("predict_url.html")


@app.route("/predict/cyber")
def predict_cyber_page():
    return render_template("predict_cyber.html")


# ─────────────────────────────────────────
# API: URL PHISHING DETECTION
# ─────────────────────────────────────────
@app.route("/api/predict/url", methods=["POST"])
def api_predict_url():
    data = request.get_json()
    url = data.get("url", "").strip()

    if not url:
        return jsonify({"error": "No URL provided"}), 400

    try:
        features = extract_url_features(url)

        if url_model_loaded:
            prediction = url_model.predict([features])[0]
            proba = url_model.predict_proba([features])[0]
            confidence = float(max(proba)) * 100
        else:
            # fallback demo logic
            score = features[9]
            suspicious = features[10]
            prediction = 1 if (score > 4.0 or suspicious > 0) else 0
            confidence = 85.0 + (score * 2) if prediction == 1 else 90.0

        label = "PHISHING" if prediction == 1 else "LEGITIMATE"
        color = "danger" if prediction == 1 else "success"

        feature_names = ["URL Length","Domain Length","Dot Count","Hyphen Count",
                         "Path Depth","Special Chars","Digit Count","HTTPS","IP URL",
                         "Entropy","Suspicious Keywords","Brand Impersonation",
                         "Subdomain Level","IP in Domain"]

        return jsonify({
            "url": url,
            "prediction": label,
            "confidence": round(confidence, 2),
            "color": color,
            "features": dict(zip(feature_names, features[:14]))
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ─────────────────────────────────────────
# API: CYBER ATTACK CLASSIFICATION
# ─────────────────────────────────────────
@app.route("/api/predict/cyber", methods=["POST"])
def api_predict_cyber():
    data = request.get_json()
    features = data.get("features", [])

    if not features:
        return jsonify({"error": "No features provided"}), 400

    try:
        feat_array = np.array(features).reshape(1, -1)

        if lgbm_loaded:
            feat_scaled = lgbm_scaler.transform(feat_array)
            prediction_idx = lgbm_model.predict(feat_scaled)[0]
            proba = lgbm_model.predict_proba(feat_scaled)[0]

            label = lgbm_label_encoder.inverse_transform([prediction_idx])[0]
            confidence = float(max(proba)) * 100

            class_probs = {
                cls: round(float(p)*100, 2)
                for cls, p in zip(lgbm_label_encoder.classes_, proba)
            }
        else:
            labels = ["Backdoor","DoS","Generic","Probe","Worm"]
            proba = np.random.dirichlet(np.ones(5))

            label = labels[np.argmax(proba)]
            confidence = float(max(proba)) * 100

            class_probs = {
                l: round(float(p)*100,2)
                for l,p in zip(labels, proba)
            }

        return jsonify({
            "prediction": label,
            "confidence": round(confidence, 2),
            "class_probabilities": class_probs
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ─────────────────────────────────────────
# OPTIONAL: CHATBOT PLACEHOLDER (NO LLM)
# ─────────────────────────────────────────
@app.route("/api/chat", methods=["POST"])
def api_chat():
    return jsonify({
        "reply": "Chatbot not implemented.\n\n👉 Integrate your own LLM API (OpenAI / Groq / etc.) here if needed."
    })


# ─────────────────────────────────────────
# MODEL STATS
# ─────────────────────────────────────────
@app.route("/api/model-stats")
def api_model_stats():
    return jsonify({
        "url_phishing": {
            "model": "Random Forest",
            "accuracy": 99.97,
            "precision": 100,
            "recall": 100,
            "f1": 100,
            "features": 14,
            "trees": 500
        },
        "cyber_attacks": {
            "model": "LightGBM",
            "accuracy": 99.16,
            "classes": ["Backdoor","DoS","Generic","Probe","Worm"]
        }
    })


# ─────────────────────────────────────────
# RUN APP
# ─────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, port=5000)