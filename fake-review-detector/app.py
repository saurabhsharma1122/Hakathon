from flask import Flask, render_template, request, jsonify
from transformers import pipeline
import re

app = Flask(__name__)

print("Loading AI model... please wait")
classifier = pipeline("zero-shot-classification", model="cross-encoder/nli-distilroberta-base")
print("Model loaded!")


def analyze_review(text):
    candidate_labels = ["genuine review", "fake review"]
    result = classifier(text, candidate_labels)

    scores = dict(zip(result["labels"], result["scores"]))
    fake_score = scores.get("fake review", 0)
    genuine_score = scores.get("genuine review", 0)

    flags = []

    # Heuristic checks
    if text.count("!") > 3:
        flags.append("Too many exclamation marks")
        fake_score += 0.08

    if len(text.split()) < 6:
        flags.append("Review is suspiciously short")
        fake_score += 0.1

    caps = [w for w in text.split() if w.isupper() and len(w) > 2]
    if len(caps) > 2:
        flags.append("Excessive use of CAPS")
        fake_score += 0.06

    generic_phrases = ["best product", "highly recommend", "five stars", "love it", "amazing product", "must buy"]
    found = [p for p in generic_phrases if p.lower() in text.lower()]
    if len(found) >= 2:
        flags.append("Contains multiple generic phrases")
        fake_score += 0.07

    if re.search(r"(.)\1{4,}", text):
        flags.append("Repeated characters detected")
        fake_score += 0.05

    # Normalize
    total = fake_score + genuine_score
    fake_pct = min(fake_score / total, 1.0) if total > 0 else 0.5

    verdict = "FAKE" if fake_pct >= 0.5 else "GENUINE"
    confidence = fake_pct if verdict == "FAKE" else (1 - fake_pct)

    return {
        "verdict": verdict,
        "confidence": round(confidence * 100, 1),
        "fake_score": round(fake_pct * 100, 1),
        "genuine_score": round((1 - fake_pct) * 100, 1),
        "flags": flags
    }


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json()
    review = data.get("review", "").strip()

    if len(review) < 5:
        return jsonify({"error": "Review is too short to analyze."}), 400

    result = analyze_review(review)
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True)