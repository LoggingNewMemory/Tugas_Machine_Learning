import sys
import os
import re
import joblib
import pandas as pd
from urllib.parse import urlparse
from scipy.sparse import hstack
from flask import Flask, request, jsonify, render_template
import requests
from huggingface_hub import InferenceClient

app = Flask(__name__)

# Load Models
MODEL_DIR = './models'
scaler_path = f"{MODEL_DIR}/scaler_phishing.joblib"
tfidf_path = f"{MODEL_DIR}/tfidf_phishing.joblib"
rf_path = f"{MODEL_DIR}/RandomForest_Phishing.joblib"

print("\n" + "="*50)
print("Loading ML models...")
scaler = joblib.load(scaler_path)
tfidf = joblib.load(tfidf_path)
best_model = joblib.load(rf_path)
print("✅ Random Forest model loaded!")

# Set up Hugging Face Inference Client
# It will automatically pick up the HF_TOKEN environment variable if set in Space Secrets
hf_client = InferenceClient(api_key=os.environ.get("HF_TOKEN"))

def tokenizer_url(url):
    url_str = str(url)
    url_str = re.sub(r'^https?://', '', url_str)
    url_str = re.sub(r'^www\.', '', url_str)
    tokens = re.split(r'[./\-_?&=]', url_str)
    return " ".join([t for t in tokens if t])

def get_largest_line_len(url):
    try:
        r = requests.get(url, timeout=3)
        if r.status_code == 200 and r.text:
            lines = r.text.splitlines()
            if lines:
                return max(len(line) for line in lines)
    except:
        pass
    return 3066

def extract_features(url):
    parsed_url = url
    if not parsed_url.startswith(('http://', 'https://')):
        parsed_url = 'http://' + parsed_url
        
    url_len = len(url)
    try:
        domain_len = len(urlparse(parsed_url).netloc)
    except:
        domain_len = 0
        
    largest_line_len = get_largest_line_len(parsed_url)
    
    numeric_df = pd.DataFrame([[url_len, domain_len, largest_line_len]], columns=['URLLength', 'DomainLength', 'LargestLineLength'])
    X_num = scaler.transform(numeric_df)
    
    url_clean = tokenizer_url(url)
    X_text = tfidf.transform([url_clean])
    
    X_final = hstack([X_num, X_text])
    return X_final

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    url = data.get('url', '')
    if not url:
        return jsonify({"error": "No URL provided"}), 400
        
    try:
        X_features = extract_features(url)
        prediction = best_model.predict(X_features)[0]
        ml_result = "Phishing 🚨" if prediction == 0 else "Legitimate ✅"
    except Exception as e:
        ml_result = f"Error: {e}"
        
    try:
        # Check if HF_TOKEN is configured
        if not os.environ.get("HF_TOKEN"):
            llm_result = "Please configure the HF_TOKEN secret in your Space settings to enable Qwen analysis."
        else:
            response = hf_client.chat_completion(
                model="Qwen/Qwen2.5-3B-Instruct",
                messages=[
                    {"role": "system", "content": "You are a cybersecurity expert. Analyze the given URL for signs of phishing. Keep it brief."},
                    {"role": "user", "content": f"Analyze this URL: {url}"}
                ],
                max_tokens=200
            )
            llm_result = response.choices[0].message.content
    except Exception as e:
        llm_result = f"Could not reach Hugging Face Serverless API: {str(e)}"
        
    return jsonify({
        "ml_prediction": ml_result,
        "llm_analysis": llm_result
    })

if __name__ == '__main__':
    # Run on HuggingFace Spaces default port
    print("\n🚀 Starting Flask server on port 7860...")
    app.run(port=7860, host="0.0.0.0")
