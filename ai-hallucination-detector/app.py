from flask import Flask, render_template, request, jsonify
from transformers import pipeline
import re

app = Flask(__name__)

print("Loading AI model... please wait")
classifier = pipeline("zero-shot-classification", model="cross-encoder/nli-distilroberta-base")
print("Model loaded!")


def analyze_hallucination(text):
    candidate_labels = ["factual and accurate response", "hallucinated or fabricated response"]
    result = classifier(text, candidate_labels)

    scores = dict(zip(result["labels"], result["scores"]))
    hallucination_score = scores.get("hallucinated or fabricated response", 0)
    factual_score = scores.get("factual and accurate response", 0)

    flags = []

    # Heuristic checks
    vague_citations = ["studies show", "research says", "experts say", "according to sources",
                       "it is known that", "scientists believe", "many experts", "some researchers"]
    found_vague = [p for p in vague_citations if p.lower() in text.lower()]
    if found_vague:
        flags.append(f"Vague citation detected: \"{found_vague[0]}\"")
        hallucination_score += 0.08

    overconfident = ["absolutely", "definitely", "always", "never", "100%", "guaranteed",
                     "proven fact", "undeniable", "without doubt"]
    found_over = [p for p in overconfident if p.lower() in text.lower()]
    if len(found_over) >= 2:
        flags.append("Overconfident language detected")
        hallucination_score += 0.07

    # Fake-sounding statistics
    if re.search(r'\b\d+(\.\d+)?%', text) and any(v in text.lower() for v in vague_citations):
        flags.append("Unverified statistic with vague source")
        hallucination_score += 0.08

    # Very specific but unverifiable numbers
    if re.search(r'\b(19|20)\d{2}\b', text) and re.search(r'\b\d+\b', text):
        if any(v in text.lower() for v in vague_citations):
            flags.append("Specific dates/numbers with no verifiable source")
            hallucination_score += 0.05

    # Contradictory phrases
    contradictions = [("always", "never"), ("increase", "decrease"), ("hot", "cold")]
    for a, b in contradictions:
        if a in text.lower() and b in text.lower():
            flags.append(f"Possibly contradictory statements detected")
            hallucination_score += 0.06
            break

    if len(text.split()) < 10:
        flags.append("Response is too short to verify properly")
        hallucination_score += 0.05

    # Normalize
    total = hallucination_score + factual_score
    hallucination_pct = min(hallucination_score / total, 1.0) if total > 0 else 0.5

    verdict = "HALLUCINATED" if hallucination_pct >= 0.5 else "LIKELY FACTUAL"
    confidence = hallucination_pct if verdict == "HALLUCINATED" else (1 - hallucination_pct)

    return {
        "verdict": verdict,
        "confidence": round(confidence * 100, 1),
        "hallucination_score": round(hallucination_pct * 100, 1),
        "factual_score": round((1 - hallucination_pct) * 100, 1),
        "flags": flags
    }


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json()
    text = data.get("text", "").strip()

    if len(text) < 10:
        return jsonify({"error": "Response is too short to analyze."}), 400

    result = analyze_hallucination(text)
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, port=5001)
