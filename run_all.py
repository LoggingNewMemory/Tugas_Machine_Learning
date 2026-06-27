# %%
# @title
# ============================================================
# Install Dependencies
# ============================================================
import sys
import os
import subprocess

VENV_DIR = ".venv"

def ensure_venv_and_deps():
    # Check if running inside a virtual environment
    in_venv = sys.prefix != sys.base_prefix
    
    if not in_venv:
        print(f"🚀 Not in a virtual environment. Setting up '{VENV_DIR}'...")
        if not os.path.exists(VENV_DIR):
            subprocess.run([sys.executable, "-m", "venv", VENV_DIR], check=True)
            
        # Path to venv python
        if os.name == 'nt':
            venv_python = os.path.join(VENV_DIR, "Scripts", "python.exe")
        else:
            venv_python = os.path.join(VENV_DIR, "bin", "python")
            
        # Restart script inside venv
        print("🔄 Restarting script inside the virtual environment...")
        os.execv(venv_python, [venv_python] + sys.argv)

    # From here on, we are inside the venv
    import importlib
    deps = {
        'pandas': 'pandas',
        'numpy': 'numpy',
        'scipy': 'scipy',
        'sklearn': 'scikit-learn',
        'joblib': 'joblib',
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn',
        'gdown': 'gdown',
        'openai': 'openai'
    }
    for mod, pkg in deps.items():
        try:
            importlib.import_module(mod)
        except ImportError:
            print(f"📦 Installing {pkg}...")
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg], check=True)

ensure_venv_and_deps()

# ============================================================
# Download Dataset dari lokal
# ============================================================
import os
import subprocess

if not os.path.exists("PhiUSIIL_Phishing_URL_Dataset.csv"):
    print("Downloading Dataset...")
    subprocess.run([sys.executable, "-m", "gdown", "https://drive.google.com/uc?id=1o3HngJ003MgcYGuu2QC80n20zz7Dd9VB&export=download", "-O", "PhiUSIIL_Phishing_URL_Dataset.csv"])

import pandas as pd

def list_features(df: pd.DataFrame, filename: str = "dataset") -> None:
    """Menampilkan informasi lengkap tentang setiap feature (kolom)."""

    total_rows, total_cols = df.shape

    print("=" * 60)
    print(f"  INFORMASI DATASET: {filename}")
    print("=" * 60)
    print(f"  Jumlah baris   : {total_rows}")
    print(f"  Jumlah kolom   : {total_cols}")
    print("=" * 60)

    print(f"\n{'No':<5} {'Nama Feature':<30} {'Tipe Data':<15} {'Missing':<10} {'Unik'}")
    print("-" * 75)

    for i, col in enumerate(df.columns, start=1):
        dtype    = str(df[col].dtype)
        missing  = df[col].isnull().sum()
        unique   = df[col].nunique()
        print(f"{i:<5} {col:<30} {dtype:<15} {missing:<10} {unique}")

    print("\n" + "=" * 60)
    print("  DAFTAR NAMA FEATURE (LIST)")
    print("=" * 60)
    features = list(df.columns)
    print(features)

    print("\n" + "=" * 60)
    print("  PREVIEW DATA (5 BARIS PERTAMA)")
    print("=" * 60)
    print(df.head().to_string(index=False))

filename = 'PhiUSIIL_Phishing_URL_Dataset.csv'
df = pd.read_csv(filename)
list_features(df, filename)



# %%
# @title
# ==========================================
# CEK MODEL DI LOKAL (LOAD ATAU TRAIN)
# ==========================================
import os
import joblib

# Lokasi folder untuk menyimpan model lokal
MODEL_DIR = './models'

if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

scaler_path = f"{MODEL_DIR}/scaler_phishing.joblib"
tfidf_path = f"{MODEL_DIR}/tfidf_phishing.joblib"
rf_path = f"{MODEL_DIR}/RandomForest_Phishing.joblib"

skip_training = False

if os.path.exists(scaler_path) and os.path.exists(tfidf_path) and os.path.exists(rf_path):
    print("✅ Model ditemukan di lokal! Langsung diload tanpa training ulang...")
    scaler = joblib.load(scaler_path)
    tfidf = joblib.load(tfidf_path)
    best_model = joblib.load(rf_path)
    skip_training = True
else:
    print("❌ Model belum lengkap di lokal. Memulai proses Training...")
    skip_training = False


# %%
# @title
if not skip_training:
    import pandas as pd
    import numpy as np
    import time
    import re
    from scipy.sparse import hstack
    
    # Scikit-learn Modules
    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.feature_extraction.text import TfidfVectorizer
    
    # --- TAMBAHAN IMPORT METRIK ---
    from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, classification_report
    
    # Models
    from sklearn.ensemble import RandomForestClassifier
    
    # ---------------------------------------------------------
    # 1. LOAD DATA
    # ---------------------------------------------------------
    filename = 'PhiUSIIL_Phishing_URL_Dataset.csv'
    
    try:
        df = pd.read_csv(filename)
        df = df.drop_duplicates()
        print(f"Data dimuat. Total baris unik: {df.shape[0]}")
    except Exception as e:
        print(f"Error: {e}")
        exit()
    
    # ---------------------------------------------------------
    # 2. FEATURE ENGINEERING (STATISTIK + TEKS)
    # ---------------------------------------------------------
    numeric_features = ['URLLength', 'DomainLength', 'LargestLineLength']
    available_numeric = [f for f in numeric_features if f in df.columns]
    X_numeric = df[available_numeric]
    
    scaler = StandardScaler()
    X_numeric_scaled = scaler.fit_transform(X_numeric)
    
    print("Sedang memproses TF-IDF pada URL...")
    def tokenizer_url(url):
        url_str = str(url)
        url_str = re.sub(r'^https?://', '', url_str)
        url_str = re.sub(r'^www\.', '', url_str)
        tokens = re.split(r'[./\-_?&=]', url_str)
        return " ".join([t for t in tokens if t])
    
    df['URL_clean'] = df['URL'].apply(tokenizer_url)
    
    tfidf = TfidfVectorizer(max_features=5000, lowercase=True)
    X_text = tfidf.fit_transform(df['URL_clean'])
    
    # C. Gabung Fitur (Dan mengambil 1 fitur target, yaitu 'label')
    X_final = hstack([X_numeric_scaled, X_text])
    y = df['label']
    print(f"Total Fitur setelah digabung: {X_final.shape[1]}")
    
    # =========================================================
    # --- KODE TAMBAHAN: RINGKASAN PENGGUNAAN FITUR DATASET ---
    # =========================================================
    print("\n--- RINGKASAN PENGGUNAAN FITUR DARI DATASET ---")
    semua_kolom = [col for col in df.columns if col != 'URL_clean']
    fitur_numerik_dipakai = available_numeric
    fitur_teks_dipakai = ['URL']
    fitur_target = ['label']
    semua_fitur_dipakai = fitur_numerik_dipakai + fitur_teks_dipakai + fitur_target
    fitur_tidak_dipakai = [col for col in semua_kolom if col not in semua_fitur_dipakai]
    
    print(f"Total Kolom Asli Dataset: {len(semua_kolom)} -> {semua_kolom}")
    print(f"\n✅ FITUR YANG DIPAKAI (Total: {len(semua_fitur_dipakai)}):")
    print(f"   - Numerik : {fitur_numerik_dipakai}")
    print(f"   - Teks    : {fitur_teks_dipakai} (Dikonversi menjadi {X_text.shape[1]} dimensi dengan TF-IDF)")
    print(f"   - Target  : {fitur_target}")
    print(f"\n❌ FITUR YANG DIBUANG/TIDAK DIPAKAI (Total: {len(fitur_tidak_dipakai)}):")
    print(f"   - {fitur_tidak_dipakai}")
    print("=" * 55 + "\n")
    
    # ---------------------------------------------------------
    # 3. SPLIT DATA
    # ---------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.2, random_state=42, stratify=y)
    
    # ---------------------------------------------------------
    # 4. HYPERPARAMETER TUNING UNTUK BANYAK MODEL
    # ---------------------------------------------------------
    print("\n--- MULAI HYPERPARAMETER TUNING MASSAL ---")
    print("Mohon tunggu, komputer sedang mencari settingan terbaik untuk tiap model...")
    
    model_dasar = {
        "Random Forest": RandomForestClassifier(random_state=42, n_jobs=-1)
    }
    
    param_grids = {
        "Random Forest": {
            'n_estimators': [50, 100],
            'max_depth': [None, 20]
        },
        "XGBoost": {
            'n_estimators': [50, 100],
            'learning_rate': [0.01, 0.1],
            'max_depth': [3, 5]
        }
    }
    
    tuned_models = {}
    
    for nama_model in model_dasar.keys():
        print(f"\n⚙️ Sedang Tuning: {nama_model} ...")
        grid_search = GridSearchCV(
            estimator=model_dasar[nama_model],
            param_grid=param_grids[nama_model],
            cv=5,
            scoring='accuracy',
            n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        tuned_models[nama_model] = grid_search.best_estimator_
        print(f"  ✅ Selesai! Settingan Terbaik: {grid_search.best_params_}")
        print(f"  📊 Akurasi (Validasi): {grid_search.best_score_ * 100:.4f}%") # Format 4 desimal
    
    # ---------------------------------------------------------
    # 5. EVALUASI AKHIR (Akurasi, Recall, Precision, F1)
    # ---------------------------------------------------------
    results = []
    print("\n" + "="*55)
    print("--- EVALUASI AKHIR SEMUA MODEL PADA DATA UJIAN ---")
    print("="*55)
    
    for name, model in tuned_models.items():
        print(f"\n[Evaluasi] {name}:")
    
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
    
        # Menghitung Metrik Dasar (Macro Average untuk menyeimbangkan kelas 0 dan 1)
        acc_train = accuracy_score(y_train, y_pred_train)
        acc_test = accuracy_score(y_test, y_pred_test)
    
        rec_train = recall_score(y_train, y_pred_train, average='macro')
        rec_test = recall_score(y_test, y_pred_test, average='macro')
    
        prec_train = precision_score(y_train, y_pred_train, average='macro')
        prec_test = precision_score(y_test, y_pred_test, average='macro')
    
        f1_train = f1_score(y_train, y_pred_train, average='macro')
        f1_test = f1_score(y_test, y_pred_test, average='macro')
    
        # Menampilkan Laporan Klasifikasi Rinci (Mencegah pembulatan ke 1.00)
        print(f"Laporan Klasifikasi untuk {name}:")
        print(classification_report(y_test, y_pred_test, digits=4))
    
    # Menyimpan hasil untuk ditabelkan
        results.append({
            'Model': name,
            'Test Acc': acc_test,
            'Test Precision': prec_test,
            'Test Recall': rec_test,
            'Test F1-Score': f1_test,
            'Train Acc': acc_train,
            'Train Precision': prec_train, # Tambahan
            'Train Recall': rec_train,     # <-- INI YANG SEMPAT HILANG
            'Train F1-Score': f1_train
        })
    
    # Tampilkan Hasil dalam Tabel
    df_results = pd.DataFrame(results)
    # Setting pandas untuk menampilkan 4 angka di belakang koma secara konsisten
    pd.options.display.float_format = '{:,.4f}'.format
    
    print("\n--- TABEL PERBANDINGAN FINAL METRIK ---")
    print(df_results.to_string(index=False)) # to_string(index=False) agar tabel lebih rapi
    
    # Visualisasi
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Visualisasi Test Metrics untuk memudahkan perbandingan di Paper
    df_melt = df_results.melt(
        id_vars="Model",
        value_vars=["Test Acc", "Test Precision", "Test Recall", "Test F1-Score"],
        var_name="Metrik",
        value_name="Skor"
    )
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Skor', y='Model', hue='Metrik', data=df_melt, palette='Set2')
    # Remove strict xlim so that metrics below 0.99 can be seen
    min_score = df_melt['Skor'].min()
    plt.xlim(max(0.0, min_score - 0.05), 1.0000)
    plt.title('Perbandingan Detail Performa Model pada Data Ujian')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()
    print("\n--- MENYIMPAN MODEL ---")
    import joblib
    joblib.dump(scaler, scaler_path)
    joblib.dump(tfidf, tfidf_path)
    joblib.dump(tuned_models['Random Forest'], rf_path)
    print(f"Model berhasil disimpan ke {MODEL_DIR}!")


# %%
# @title
if not skip_training:
    # Import library tambahan yang dibutuhkan
    from sklearn.metrics import confusion_matrix
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    
    print("--- MENGHITUNG CONFUSION MATRIX ---")
    
    # Siapkan list untuk menyimpan data tabel
    cm_data = []
    
    # Siapkan canvas / figure untuk visualisasi berdampingan
    # Jumlah kolom disesuaikan dengan jumlah model yang ada di 'tuned_models'
    num_models = len(tuned_models)
    fig, axes = plt.subplots(1, num_models, figsize=(5 * num_models, 5))
    
    # Jika hanya 1 model, axes tidak berbentuk array, jadi kita bungkus ke dalam list
    if num_models == 1:
        axes = [axes]
    
    # Loop untuk tiap model
    for i, (name, model) in enumerate(tuned_models.items()):
        # 1. Lakukan Prediksi
        y_pred_test = model.predict(X_test)
    
        # 2. Buat Confusion Matrix
        cm = confusion_matrix(y_test, y_pred_test)
    
        # 3. Ekstrak nilai untuk Tabel (Asumsi label binary: 0 dan 1)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            cm_data.append({
                'Model': name,
                'True Negative (TN)': tn,
                'False Positive (FP)': fp,
                'False Negative (FN)': fn,
                'True Positive (TP)': tp
            })
        else:
            # Jika bukan binary (multiclass)
            cm_data.append({
                'Model': name,
                'Catatan': 'Format multiclass, lihat visualisasi.'
            })
    
        # 4. Plotting Heatmap di subplot yang sesuai
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i], cbar=False, annot_kws={"size": 14})
        axes[i].set_title(f'{name}', fontsize=14, fontweight='bold')
        axes[i].set_xlabel('Predicted Label', fontsize=12)
        axes[i].set_ylabel('True Label', fontsize=12)
    
    # Tampilkan Visualisasi
    plt.tight_layout()
    plt.show()
    
    # Tampilkan Tabel
    print("\n--- TABEL RINCIAN CONFUSION MATRIX ---")
    df_cm = pd.DataFrame(cm_data)
    print(df_cm) # 'print' menampilkan tabel di console lokal

# %%
# @title
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Memastikan variabel 'df' sudah didefinisikan
try:
    df
except NameError:
    # Load dataset jika variabel df belum ada di memori
    filename = 'PhiUSIIL_Phishing_URL_Dataset.csv'
    df = pd.read_csv(filename)
    print(f"Data dimuat ulang dari {filename}")

# Memilih kolom numerik saja untuk korelasi
numeric_df = df.select_dtypes(include=['float64', 'int64'])

# Menghitung matriks korelasi
corr_matrix = numeric_df.corr()

# Visualisasi dengan Heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix - Phishing URL Dataset')
plt.show()

# %%

from openai import OpenAI

# Connected to your local laptop!
LAPTOP_API_URL = "http://localhost:8000/v1"

client = OpenAI(
    base_url=LAPTOP_API_URL,
    api_key="empty"
)

def analyze_phishing_url(url):
    print(f"Sending URL to local Qwen model: {url}")
    response = client.chat.completions.create(
        model="Qwen/Qwen2.5-3B-Instruct",
        messages=[
            {"role": "system", "content": "You are a cybersecurity expert. Analyze the given URL for signs of phishing. Keep it brief."},
            {"role": "user", "content": f"Analyze this URL: {url}"}
        ]
    )
    return response.choices[0].message.content

# Start local Qwen server automatically
import subprocess
import time
import urllib.request
import os
import sys

print("\n" + "="*50)
print("Starting local Qwen model server in the background...")
vllm_python = os.path.join("qwen_venv", "bin", "python") if os.name != 'nt' else os.path.join("qwen_venv", "Scripts", "python.exe")

if not os.path.exists(vllm_python):
    print(f"❌ Error: Could not find python in qwen_venv. Please start the server manually.")
else:
    # Set environment variable inside the process
    env = os.environ.copy()
    env["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    
    vllm_process = subprocess.Popen([
        vllm_python, "-m", "vllm.entrypoints.openai.api_server",
        "--model", "./Qwen2.5-3B-Instruct",
        "--served-model-name", "Qwen/Qwen2.5-3B-Instruct",
        "--max-model-len", "4096",
        "--port", "8000"
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env)

    print("Waiting for Qwen server to be ready (this may take a minute or two)...")
    server_ready = False
    for _ in range(120): # Wait up to 4 minutes
        try:
            with urllib.request.urlopen("http://localhost:8000/v1/models", timeout=2):
                pass
            server_ready = True
            print("✅ Qwen server is up and running!")
            break
        except Exception:
            time.sleep(2)
            
    if server_ready:
        try:
            # Test the 5th URL from your dataset:
            sample_url = df.iloc[4]['URL']
            print("\n" + analyze_phishing_url(sample_url))
        finally:
            print("\nShutting down Qwen server...")
            vllm_process.terminate()
            vllm_process.wait()
    else:
        print("❌ Failed to start Qwen server or it took too long.")
        vllm_process.terminate()
