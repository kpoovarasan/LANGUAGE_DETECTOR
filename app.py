from flask import Flask, request, render_template
import pickle
import nltk
from nltk.tokenize import word_tokenize

# Download tokenizer (only once)
nltk.download('punkt')

# Load model and vectorizer
with open("language_detector_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("language_vectorizer.pkl", "rb") as vec_file:
    vectorizer = pickle.load(vec_file)

# Initialize Flask app
app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        text = request.form["text"]

        # Tokenize and rejoin
        tokens = word_tokenize(text)
        cleaned_text = ' '.join(tokens)

        # Vectorize and predict
        X = vectorizer.transform([cleaned_text])
        
        # Language code to full name
        lang_map = {
            'en': 'English', 'fr': 'French', 'de': 'German', 'es': 'Spanish', 'it': 'Italian',
            'pt': 'Portuguese', 'nl': 'Dutch', 'ru': 'Russian', 'zh': 'Chinese',
            'ja': 'Japanese', 'ar': 'Arabic', 'hi': 'Hindi', 'sw': 'Swahili',
            'tr': 'Turkish', 'ur': 'Urdu', 'vi': 'Vietnamese', 'pl': 'Polish',
            'el': 'Greek', 'bg': 'Bulgarian', 'th': 'Thai'
        }
        pred_code = model.predict(X)[0]
        prediction = lang_map.get(pred_code, pred_code)  # fallback to code if not found



        return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
