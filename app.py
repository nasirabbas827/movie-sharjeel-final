from flask import Flask, render_template, request
import joblib
import numpy as np
import re
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# -------------------------------
# Load saved models and artifacts
# -------------------------------
logreg = joblib.load("tfidf_logreg_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")

lstm_model = load_model("lstm_model.h5")
tokenizer = joblib.load("lstm_tokenizer.pkl")

MAX_LEN = 200  # Must match LSTM training

app = Flask(__name__)

# -------------------------------
# Helper function to convert binary prediction to label
# -------------------------------
def decode_sentiment(pred, classes=["negative", "positive"]):
    """Convert 0/1 prediction to label"""
    if isinstance(pred, (np.ndarray, list)):
        pred = int(pred[0])
    return classes[pred]

# -------------------------------
# Text preprocessing
# -------------------------------
def clean_text(text):
    """Clean user input before prediction"""
    if not isinstance(text, str):
        return ""
    text = re.sub(r'<[^>]+>', ' ', text)             # Remove HTML
    text = re.sub(r'http\S+|www\S+', ' ', text)      # Remove URLs
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)         # Remove non-alphabets
    text = re.sub(r'\s+', ' ', text).strip()         # Extra spaces
    return text.lower()

# -------------------------------
# Flask Routes
# -------------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    review_text = ""
    model_choice = "logreg"

    if request.method == "POST":
        review_text = request.form["review"]
        model_choice = request.form["model"]

        # Preprocess input
        review_cleaned = clean_text(review_text)
        print(review_cleaned)

        if model_choice == "logreg":
            # TF-IDF + Logistic Regression
            X_input = tfidf.transform([review_cleaned])
            pred = logreg.predict(X_input)[0]
            pred_label = decode_sentiment(pred)
            prediction = f"Predicted Sentiment (Logistic Regression): {pred_label.capitalize()}"

        elif model_choice == "lstm":
            # Tokenize + Pad
            seq = tokenizer.texts_to_sequences([review_cleaned])
            X_input = pad_sequences(seq, maxlen=MAX_LEN, padding="post")
            pred_prob = lstm_model.predict(X_input, verbose=0)
            pred = int(pred_prob.flatten()[0] > 0.5)  # binary threshold
            pred_label = decode_sentiment(pred)
            prediction = f"Predicted Sentiment (LSTM): {pred_label.capitalize()}"

    return render_template(
        "index.html",
        prediction=prediction,
        review_text=review_text,
        model_choice=model_choice
    )

if __name__ == "__main__":
    app.run(debug=True)
