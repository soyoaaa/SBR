# Safety Anchor: Defending Harmful Fine-tuning via Geometric Bottlenecks

Official implementation of the paper **"Safety Anchor: Defending Harmful Fine-tuning via Geometric Bottlenecks"**.

## 🚀 Reproduction Methods

### Method 1: Standard Reproduction (Based on this Repository)

**1. Environment Setup**
Please create a Python 3.10.16 environment and install the specific dependencies:

```bash
conda create -n sbr python=3.10.16
conda activate sbr

# Install dependencies (Ensure torch 2.7 compatible CUDA version)
pip install torch==2.7.0+cu118
pip install "unsloth==2025.6.2" "unsloth_zoo==2025.6.1"
pip install transformers==4.51.3 trl==0.15.2 accelerate==1.6.0
```

### 2. Configuration

* **Model Setup**:
    1.  Download your base model, e.g., Llama3-8B.
    2.  Open `HFT_SBR.py` and `HFT_base.py`.
    3.  Replace the `MODEL_PATH` variable with the actual local path to your model.

* **Data Setup**:
    We provide the necessary datasets in the `data/` folder.
    * *Note*: If you use a different data format, you must modify the **template construction** logic in the code to match your data.

### 3. Execution (Harmful Fine-tuning)

We provide scripts for both undefended and defended fine-tuning.

* **Option A: Baseline (Without Defense)**
    Directly run the following command to perform harmful fine-tuning without defense:
    ```bash
    python HFT_base.py
    ```

* **Option B: SBR Defense**
    Run the following command to perform fine-tuning with Safety Bottleneck Regularization (SBR):
    ```bash
    python HFT_SBR.py
    ```

### 4. Evaluation

For rapid testing, we randomly sampled **100 instances** from the full 1000-sample dataset in `test_unsafe.py`. Results on this subset are statistically consistent with the full dataset. We recommend using **LLM-as-a-Judge**.

1.  Open `test_unsafe.py`.
2.  Replace `"your_api_key"` with your actual API key.
3.  Run the evaluation:
    ```bash
    python test_unsafe.py
    ```

---

## Method 2: AI-Assisted Reproduction (Flexible)

To avoid complex environment configurations (e.g., specific `unsloth` or `torch` versions), you can leverage AI tools (like Gemini) to reproduce our work using your preferred setup.

**Steps:**
1.  Upload this repository link and our paper PDF to the AI.
2.  Ask the AI to understand the logic and rewrite the reproduction script.
3.  Further rewrite the reproduction script tailored for your local environment and your own data templates.
4. **PS**: You must verify that the **data processing and tokenization templates** in the AI-generated code match your specific dataset format.

---