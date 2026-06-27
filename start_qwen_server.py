import subprocess
import sys
import os

print("\n" + "="*60)
print("Starting Local Qwen Model Server (vLLM)")
print("="*60 + "\n")

# Disable FlashInfer sampler to prevent missing NVCC errors
os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"

vllm_python = os.path.join("qwen_venv", "bin", "python") if os.name != 'nt' else os.path.join("qwen_venv", "Scripts", "python.exe")

if not os.path.exists(vllm_python):
    print("❌ Error: 'qwen_venv' not found. Please ensure vllm is installed.")
    sys.exit(1)

print("Loading Qwen2.5-3B-Instruct into memory (this may take a minute)...")
try:
    # Start vLLM pointing to the Qwen model using the venv
    subprocess.run([
        vllm_python, "-m", "vllm.entrypoints.openai.api_server", 
        "--model", "./Qwen2.5-3B-Instruct", 
        "--served-model-name", "Qwen/Qwen2.5-3B-Instruct",
        "--max-model-len", "4096",
        "--port", "8000"
    ])
except KeyboardInterrupt:
    print("\nShutting down Qwen server...")
