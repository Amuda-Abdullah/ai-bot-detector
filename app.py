from flask import Flask, render_template, request, jsonify, redirect
from flask_cors import CORS
import pickle
import re
import string
import os
import requests
from dotenv import load_dotenv

load_dotenv()  # load the .env file

app = Flask(__name__)
CORS(app)

API_KEY = os.getenv("API_KEY")
ZEROGPT_URL = os.getenv("ZEROGPT_URL")

@app.route('/', methods=['GET', 'POST'])
def home():
    result = None
    input_text = ''
    percentage = 0

    if request.method == 'POST':
        input_text = request.form.get('input_text', '')

        try:
            with open("text_model.pkl", "rb") as model_file:
                loaded_model = pickle.load(model_file)
            with open("model.pkl", "rb") as vectorizer_file:
                loaded_vectorizer = pickle.load(vectorizer_file)

            text_tfidf = loaded_vectorizer.transform([clean_text(input_text)])
            prediction = loaded_model.predict(text_tfidf)
            result = "Bot-generated" if prediction[0] == 1 else "Human-written"
            percentage = loaded_model.predict_proba(text_tfidf)[0][1] * 100
        except Exception as e:
            result = f"Error: {str(e)}"

    return render_template('landing.html', result=result, input_text=input_text, percentage=percentage)

@app.route('/whatsapp_redirect')
def whatsapp_redirect():
    whatsapp_number = "2347012546884"
    message = "Hello, I need assistance with the Bot Text Detector!"
    whatsapp_url = f"https://wa.me/{whatsapp_number}?text={message.replace(' ', '%20')}"
    return redirect(whatsapp_url)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json 
        if not data or "input_text" not in data:
            return jsonify({"error": "Missing 'input_text' field"}), 400

        response = requests.post(
            ZEROGPT_URL,
            headers={
                "Content-Type": "application/json",
                "ApiKey": API_KEY
            },
            json={"input_text": data["input_text"]}
        )

        return jsonify(response.json()), response.status_code

    except Exception as e:
        return jsonify({"error": str(e)}), 500

def clean_text(text):
    text = text.lower()
    text = re.sub(f"[{string.punctuation}]", "", text)
    text = re.sub("\d+", "", text)
    return text

if __name__ == '__main__':
    os.makedirs('static', exist_ok=True)
    app.run(debug=True, port=8004)
