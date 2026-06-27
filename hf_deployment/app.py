import sys
import os
import subprocess
import re
import joblib
import pandas as pd
from urllib.parse import urlparse
from scipy.sparse import hstack
from flask import Flask, request, jsonify, render_template
from openai import OpenAI
import requests

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

# OpenAI Client for Qwen
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="empty"
)

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
    # Fallback to Legitimate median (3066) to prevent false-positive phishing flags on short/offline URLs
    return 3066

def extract_features(url):
    # Ensure URL has a scheme for correct parsing
    parsed_url = url
    if not parsed_url.startswith(('http://', 'https://')):
        parsed_url = 'http://' + parsed_url
        
    url_len = len(url) # Length of original input
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
        # In this dataset, 0 is Phishing and 1 is Legitimate
        ml_result = "Phishing 🚨" if prediction == 0 else "Legitimate ✅"
        
    except Exception as e:
        ml_result = f"Error: {e}"
        
    try:
        response = client.chat.completions.create(
            model="Qwen/Qwen2.5-3B-Instruct",
            messages=[
                {"role": "system", "content": "You are a cybersecurity expert. Analyze the given URL for signs of phishing. Keep it brief."},
                {"role": "user", "content": f"Analyze this URL: {url}"}
            ]
        )
        llm_result = response.choices[0].message.content
    except Exception as e:
        llm_result = "Could not reach Qwen server. Please ensure the Qwen model is running on port 8000."
        
    return jsonify({
        "ml_prediction": ml_result,
        "llm_analysis": llm_result
    })

if __name__ == '__main__':
    import time
    import urllib.request
    import atexit

    print("\n" + "="*50)
    print("Checking for local Qwen server...")
    
    qwen_running = False
    try:
        with urllib.request.urlopen("http://localhost:8000/v1/models", timeout=2):
            pass
        qwen_running = True
        print("✅ Qwen server is already running!")
    except Exception:
        pass

    vllm_process = None
    vllm_log = None
    if not qwen_running:
        print("Starting Qwen model server in the background using vLLM...")
        env = os.environ.copy()
        env["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
        
        vllm_log = open("qwen_server.log", "w")
        
        vllm_process = subprocess.Popen([
            sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--model", "Qwen/Qwen2.5-3B-Instruct",
            "--served-model-name", "Qwen/Qwen2.5-3B-Instruct",
            "--max-model-len", "4096",
            "--port", "8000"
        ], stdout=vllm_log, stderr=subprocess.STDOUT, env=env)

        print("Waiting for Qwen server to be ready (this may take a minute or two)...")
        for _ in range(300): # Wait up to 10 minutes
            try:
                with urllib.request.urlopen("http://localhost:8000/v1/models", timeout=2):
                    pass
                qwen_running = True
                print("✅ Qwen server is up and running!")
                break
            except Exception:
                time.sleep(2)
                
        if not qwen_running:
            print("❌ Failed to start Qwen server. Check qwen_server.log for details.")
            vllm_process.terminate()

    if vllm_process:
        def cleanup():
            print("\nShutting down Qwen server...")
            vllm_process.terminate()
            vllm_process.wait()
            if vllm_log:
                vllm_log.close()
        atexit.register(cleanup)

    # Run on HuggingFace Spaces default port
    app.run(port=7860, host="0.0.0.0")
