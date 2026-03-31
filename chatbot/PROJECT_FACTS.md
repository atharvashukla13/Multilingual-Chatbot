# 📋 Multilingual Ayurvedic RAG Chatbot — Project Facts

> This document tracks all key facts about the project—datasets, preprocessing, training, and architecture.

---

## 🏗️ Architecture Overview

| Component | Technology |
|-----------|-----------|
| **UI** | Streamlit (`app.py`) |
| **Base Model** | Google `mt5-small` (multilingual T5, 300M params) |
| **Fine-Tuning** | QLoRA (4-bit quantization + LoRA adapters) |
| **Retrieval** | FAISS index over 8,729 Ayurvedic passages |
| **Embeddings** | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` |
| **Translation** | `deep-translator` (Google Translate API) |
| **Pipeline** | RAG (Retrieve → Augment → Generate) |

### How It Works
1. User enters a query (English or Hindi)
2. Query is translated to Hindi (if English)
3. FAISS retrieves top-k relevant passages from the knowledge base
4. Retrieved passages are fed as context to the mT5 model
5. Model generates an answer in Hindi
6. Answer is translated back to user's language

### Fallback Mechanism
- If no fine-tuned LoRA adapter is found, the system uses the **base mT5-small** model
- If the base model produces garbage output (e.g., `extra_id` tokens), the system returns the **top retrieved passage** as the answer
- This makes the chatbot functional even without fine-tuning (acts as a smart search engine)

---

## 📚 Datasets Used

### 1. Hindi Orthopedic Q&A (`hindi_dataset.csv`)
- **Source:** Pre-existing CSV in `data/raw/`
- **Size:** 2,183 patient records → **4,364 Q&A pairs**
- **Format:** Clinical records with `symptoms`, `treatment`, `Diagnosis`, `Patient History`
- **Preprocessing:** Converted symptoms→question, treatment→answer, with diagnosis context
- **Example Q:** `बाएं कूल्हे का अस्वास्कुलर नेक्रोसिस के लक्षण 'कूल्हे में दर्द...' हैं। इसका उपचार क्या है?`
- **Language:** Hindi

### 2. BhashaBench Ayurvedic (`bhashbench_ayur_hindi.json`)
- **Source:** Hugging Face — `MBZUAI-Paris/BhashaBench` (Ayurveda subset)
- **Size:** 5,615 pairs
- **Format:** MCQ with `question`, `option_a`-`option_d`, `correct_answer` (A/B/C/D)
- **Preprocessing:** Extracted correct option text as answer, added topic/domain context
- **Language:** Hindi
- **Also available:** English version (`bhashbench_ayur_english.json`)

### 3. HiMed-Trad Benchmark (`himed_trad_bench.json`)
- **Source:** Hugging Face — `SajjadAyoubi/HiMed` (Traditional Medicine benchmark split)
- **Size:** 6,010 pairs
- **Format:** MCQ/QA/Dialogue with `question`, `answer`, `cot` (chain-of-thought explanation)
- **Preprocessing:** Used CoT explanations as answers (richer, more explanatory)
- **Language:** Hindi

### 4. HiMed-Trad Corpus (`himed_trad_corpus.json`)
- **Source:** Hugging Face — `SajjadAyoubi/HiMed` (Traditional Medicine corpus split)
- **Size:** 286,657 total entries → **14,995 sampled**
- **Breakdown:** MCQ: 96,484 | QA: 93,859 | Dialogue: 96,314
- **Sampling:** Balanced across MCQ/QA/Dialogue types (5K each)
- **Preprocessing:** CoT for MCQ, direct answers for QA, dialogue responses for Dialogue
- **Language:** Hindi

### 5. Ashtanga Hridayam Classical Text (`ashtanga.txt`)
- **Source:** Downloaded textbook — "अष्टांगहृदयम्" (Ashtanga Hridayam) by Vagbhata
- **Size:** 2,183 passages → 2,183 Q&A pairs
- **Processing Pipeline:**
  1. Raw text file (`dataset/ashtanga.txt`) parsed into passages
  2. Stored as `data/processed/classical_passages.json` with chapter/section metadata
  3. Passages indexed in FAISS for retrieval (part of the 8,729 passage knowledge base)
  4. Converted to Q&A format for fine-tuning (question = topic prompt, answer = passage text)
- **Language:** Hindi (Sanskrit-Hindi commentary)

---

## 🔧 Preprocessing Pipeline

### Script: `data/preprocess_all.py`

**Input:** 5 datasets from `dataset/` and `data/` directories  
**Output:** `data/processed/train.json`, `val.json`, `test.json`

### Final Dataset Statistics

| Split | Samples |
|-------|---------|
| **Train** | 28,191 |
| **Val** | 3,317 |
| **Test** | 1,659 |
| **Total** | **33,167** |

### Source Distribution

| Source | Count |
|--------|-------|
| HiMed Corpus (sampled) | 14,995 |
| HiMed Bench | 6,010 |
| BhashaBench Ayur | 5,615 |
| Hindi Dataset (Orthopedic) | 4,364 |
| Classical Text (Ashtanga Hridayam) | 2,183 |

### Data Format (each JSON entry)
```json
{
  "question_hi": "प्रश्न हिंदी में",
  "answer_hi": "उत्तर हिंदी में",
  "source": "dataset_name"
}
```

### Split Ratio
- Train: 85% | Val: 10% | Test: 5%
- Random seed: 42 (reproducible)

---

## 🎯 Fine-Tuning Configuration

### Method: QLoRA (Quantized Low-Rank Adaptation)

| Parameter | Value |
|-----------|-------|
| **Base Model** | `google/mt5-small` |
| **Quantization** | 4-bit (NF4 via bitsandbytes) |
| **LoRA Rank (r)** | 16 |
| **LoRA Alpha** | 32 |
| **LoRA Dropout** | 0.05 |
| **Target Modules** | `q`, `v` (attention layers) |
| **Learning Rate** | 3e-4 |
| **Epochs** | 3 |
| **Batch Size** | 4 (with gradient accumulation = 4 → effective 16) |
| **Max Input Length** | 512 tokens |
| **Max Target Length** | 256 tokens |
| **Optimizer** | paged_adamw_8bit |
| **Scheduler** | cosine |
| **Warmup** | 10% of steps |
| **FP16** | Enabled (on Colab GPU) |

### Training Platform
- **Google Colab** (free T4 GPU)
- **Notebook:** `Fine_Tune_mT5_Colab.ipynb`
- **Estimated Time:** ~2-3 hours on T4

### Output
- LoRA adapters saved to: `chatbot/models/mt5_ayurvedic_lora/`
- Files: `adapter_model.safetensors`, `adapter_config.json`

---

## 📁 Key File Locations

```
chatbot/
├── app.py                          # Streamlit UI
├── config.py                       # All hyperparameters & paths
├── requirements.txt                # Python dependencies
├── PROJECT_FACTS.md                # ← This file
├── dataset/
│   ├── ashtanga.txt                # Ashtanga Hridayam book (raw)
│   ├── bhashbench_ayur_hindi.json  # BhashaBench MCQs
│   ├── bhashbench_ayur_english.json
│   ├── himed_trad_bench.json       # HiMed benchmark
│   ├── himed_trad_corpus.json      # HiMed corpus (286K)
│   └── hindi_dataset.csv           # Orthopedic Q&A
├── data/
│   ├── preprocess_all.py           # Unified preprocessing script
│   ├── raw/
│   │   └── hindi_dataset.csv
│   └── processed/
│       ├── train.json              # 24,482 samples
│       ├── val.json                # 2,880 samples
│       ├── test.json               # 1,441 samples
│       ├── classical_passages.json # Ashtanga Hridayam passages
│       └── ayurvedic_kb.index      # FAISS index (8,729 passages)
├── models/
│   ├── inference.py                # Generator with fallback logic
│   ├── fine_tune_mt5.py            # Training script (local)
│   └── mt5_ayurvedic_lora/         # LoRA adapters (after training)
├── rag/
│   ├── pipeline.py                 # RAG orchestrator
│   ├── retriever.py                # FAISS retrieval
│   └── translator.py              # deep-translator wrapper
└── Fine_Tune_mT5_Colab.ipynb       # Colab training notebook
```

---

## 🔄 Dependencies

Key packages (from `requirements.txt`):
- `transformers` — Model loading & inference
- `peft` — LoRA adapter support
- `sentence-transformers` — Embedding model
- `faiss-cpu` — Vector similarity search
- `deep-translator` — Translation (replaced `googletrans`)
- `streamlit` — Web UI
- `datasets` — Hugging Face dataset loading
- `bitsandbytes` — 4-bit quantization (Colab only)

### Known Dependency Notes
- `googletrans` was replaced with `deep-translator` due to `httpx` version conflict with the `datasets` library
- `bitsandbytes` is only needed for fine-tuning on Colab, not for local inference

---

## 📝 Change Log

| Date | Change |
|------|--------|
| 2026-03-24 | Initial setup: retriever, translator, basic inference |
| 2026-03-24 | Downloaded datasets: BhashaBench, HiMed, hindi_dataset |
| 2026-03-24 | Built FAISS index (8,729 passages) |
| 2026-03-24 | Fixed googletrans → deep-translator |
| 2026-03-24 | Added fallback: base model returns top passage if output is garbage |
| 2026-03-24 | Fixed UI: passages display immediately with response |
| 2026-03-28 | Created comprehensive preprocessing (`preprocess_all.py`) |
| 2026-03-28 | Merged all 5 datasets → 28,803 Q&A pairs (train: 24,482) |
| 2026-03-28 | Created Colab fine-tuning notebook |
