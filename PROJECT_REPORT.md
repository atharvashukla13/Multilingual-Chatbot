# 🌿 Bilingual Ayurvedic Health Advisor Chatbot

## Project Overview

A **Hindi-primary, bilingual (Hindi + English)** chatbot that answers Ayurvedic health queries using a **Retrieval-Augmented Generation (RAG)** pipeline. The system retrieves relevant passages from a curated Ayurvedic knowledge base and generates contextual responses using a fine-tuned multilingual language model.

---

## Architecture

```
User Query (Hindi / English)
        │
        ▼
┌─────────────────────┐
│  Language Detection  │  ← langdetect library
│  (Hindi / English)   │
└────────┬────────────┘
         │ if English
         ▼
┌─────────────────────┐
│  EN → HI Translation│  ← googletrans (dev) / IndicTrans2 (prod)
└────────┬────────────┘
         │ Hindi query
         ▼
┌─────────────────────┐
│  FAISS Retriever     │  ← SentenceTransformer + FAISS Index
│  Top-K passages      │     (paraphrase-multilingual-MiniLM-L12-v2)
└────────┬────────────┘
         │ query + passages
         ▼
┌─────────────────────┐
│  mT5-small Generator │  ← QLoRA fine-tuned on Ayurvedic Q&A
│  (Hindi response)    │
└────────┬────────────┘
         │ if English user
         ▼
┌─────────────────────┐
│  HI → EN Translation│
└────────┬────────────┘
         ▼
    Final Response
```

---

## Technology Stack & Why We Chose Each

### 1. Base Model — **mT5-small** (300M params)

| Aspect | Decision | Rationale |
|--------|----------|-----------|
| Model | `google/mt5-small` | Multilingual T5, trained on 101 languages including **Hindi and Sanskrit**. Seq2seq architecture is ideal for Q&A generation. |
| Why not GPT/LLaMA? | Too large | GPT-2 is English-only; LLaMA-7B needs 14GB+ VRAM. mT5-small fits in 4GB VRAM with quantization. |
| Quantization | **QLoRA (4-bit NF4)** | Reduces memory from ~1.2GB to ~300MB. Enables training on **RTX 3050 (4GB VRAM)** or Colab T4. |
| Fine-tuning | **LoRA adapters** (rank=16, alpha=32) | Only ~2M trainable params instead of 300M. Adapter files are ~20MB — easy to share and iterate. |

### 2. Embedding Model — **paraphrase-multilingual-MiniLM-L12-v2**

| Aspect | Decision | Rationale |
|--------|----------|-----------|
| Model | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | Supports **50+ languages** including Hindi. Produces 384-dim embeddings. |
| Why not OpenAI embeddings? | Offline + free | No API key needed, runs locally. Crucial for a student project with limited resources. |
| Index | **FAISS (IndexFlatIP)** | Facebook's vector similarity search. Inner Product on normalized embeddings = cosine similarity. Fast, CPU-compatible. |

### 3. Translation — **langdetect + googletrans**

| Aspect | Decision | Rationale |
|--------|----------|-----------|
| Language Detection | `langdetect` | Lightweight, deterministic (seeded), supports Hindi detection out-of-the-box. |
| Translation | `googletrans 4.0.0-rc.1` | Free, no API key needed. Good quality for development. Can swap to **IndicTrans2** for production accuracy. |
| Strategy | **Hindi-primary pipeline** | All retrieval and generation happens in Hindi. English users' queries are translated to Hindi first, responses translated back. This keeps the core model focused on one language. |

### 4. Web Interface — **Streamlit**

| Aspect | Decision | Rationale |
|--------|----------|-----------|
| Framework | Streamlit | Rapid prototyping, built-in chat UI components, cached model loading. Perfect for demos. |
| Features | Chat history, passage sidebar, language auto-detect | Transparency — users can see which passages the answer came from. |

---

## Data Sources

### Datasets Used

| # | Source | Type | Size | Language | Status |
|---|--------|------|------|----------|--------|
| 1 | **hindi_dataset.csv** | Clinical Ayurvedic Q&A (symptoms, diagnosis, treatment) | ~2MB | Hindi | ✅ Integrated |
| 2 | **BhashaBench-Ayur** (`bharatgenai/BhashaBench-Ayur`) | 14,963 exam-based MCQs across 15+ Ayurvedic disciplines | ~15K samples | Hindi + English | ✅ Downloader ready |
| 3 | **Ashtanga Hridayam** (`ashtanga.txt`) | Classical Ayurvedic text — Hindi commentary by Dr. Brahmanand Tripathi | 22,430 lines → **2,183 passages** | Hindi | ✅ Processed |
| 4 | **Ayurveda Text Q&A** (`Macromrit/ayurveda-text-based-qanda`) | Q&A from classical texts | Variable | English + Sanskrit | ✅ Downloader ready |
| 5 | **HiMed-Trad** (`FreedomIntelligence/HiMed`) | Hindi traditional Indian medicine corpus + benchmark | ~286K corpus + 6K bench | Hindi | ✅ Downloader ready |

### Why These Specific Sources?

- **hindi_dataset.csv** — Real clinical data with symptom-diagnosis-treatment triples. Generates multiple Q&A styles per record (symptom→diagnosis, symptom→treatment, history→full analysis).
- **BhashaBench-Ayur** — The most rigorous Ayurvedic evaluation benchmark. Exam-quality questions validated by domain experts.
- **Ashtanga Hridayam** — One of the *three great treatises* (Brihat Trayi) of Ayurveda. Our copy is a clean, well-structured Hindi commentary covering doshas, rasas, diet, seasonal routines, treatments, and panchakarma in detail.
- **HiMed-Trad** — Massive Hindi corpus specifically covering traditional Indian medicine including Ayurveda, Siddha, and Unani systems.

### Classical Text Processing

We built a custom pipeline (`process_classical_text.py`) to handle `ashtanga.txt`:

1. **Skip front-matter** (publisher info, table of contents — first 900 lines)
2. **Clean OCR artifacts** (stray symbols, page numbers, headers)
3. **Detect chapter/section boundaries** (patterns like `अध्यायः`, `वर्गः`, topic headers with `---`)
4. **Chunk into ~500-char passages** at sentence boundaries (Hindi `।` and `॥`)
5. **Output**: 2,183 clean passages with chapter and section metadata

---

## Codebase Structure

```
chatbot/
├── config.py                          # All hyperparameters & paths (single source of truth)
├── requirements.txt                   # 12 dependencies
├── app.py                             # Streamlit web interface
│
├── data/
│   ├── download_data.py               # Download 3 HuggingFace datasets
│   ├── preprocess.py                  # Clean, translate EN→HI, format, split train/val/test
│   ├── build_kb.py                    # Build FAISS index from all passages
│   └── process_classical_text.py      # Process ashtanga.txt into chunked passages
│
├── dataset/
│   ├── hindi_dataset.csv              # Local clinical Ayurvedic data
│   └── ashtanga.txt                   # Ashtanga Hridayam full text (Hindi)
│
├── models/
│   ├── fine_tune_mt5.py               # QLoRA fine-tuning (Colab + 3050 compatible)
│   └── inference.py                   # Load model & generate responses
│
└── rag/
    ├── retriever.py                   # SentenceTransformer + FAISS retrieval
    ├── translator.py                  # Language detection + EN↔HI translation
    └── pipeline.py                    # Full RAG orchestrator
```

---

## Key Implementation Details

### Data Preprocessing (`preprocess.py`)

- Ingests **both** HuggingFace datasets and local `hindi_dataset.csv`
- Normalizes Devanagari text (Unicode normalization)
- Translates English Q&As to Hindi using `googletrans`
- Generates **multiple Q&A styles** from clinical data:
  - *Symptom → Diagnosis*
  - *Symptom → Treatment + Timespan*
  - *Patient History → Diagnosis + Treatment*
- Splits into train/val/test (80/10/10)
- Builds a knowledge base JSON with `passage_hi` for FAISS indexing

### Knowledge Base Building (`build_kb.py`)

- Merges Q&A passages + classical text passages (Ashtanga Hridayam)
- Embeds all passages using `paraphrase-multilingual-MiniLM-L12-v2`
- Builds FAISS `IndexFlatIP` (cosine similarity on normalized vectors)
- Saves index + metadata for runtime retrieval

### Fine-Tuning (`fine_tune_mt5.py`)

- Input format: `"प्रश्न: {Hindi question} संदर्भ: {context passages}"`
- Target: `"{Hindi answer}"`
- 4-bit quantization + LoRA (only ~2M trainable params)
- Designed to run on:
  - **RTX 3050** (4GB VRAM) — with gradient accumulation
  - **Google Colab T4** (16GB VRAM) — comfortable fit

### RAG Pipeline (`pipeline.py`)

The `AyurvedicRAG` class orchestrates the full flow:

```python
def answer(self, user_query):
    # 1. Detect language (Hindi or English)
    # 2. If English → translate to Hindi
    # 3. Retrieve top-K passages from FAISS
    # 4. Feed query + passages to mT5
    # 5. Generate Hindi response
    # 6. If English user → translate back to English
    # 7. Return response + metadata
```

---

## Hardware Requirements

| Environment | GPU | VRAM | Notes |
|-------------|-----|------|-------|
| Local (user's setup) | RTX 3050 | 4GB | QLoRA makes fine-tuning possible |
| Google Colab (free) | T4 | 16GB | More comfortable for training |
| Inference only | CPU | — | Works without GPU (slower) |

---

## How to Run

```bash
# 1. Install dependencies
pip install -r chatbot/requirements.txt

# 2. Set HuggingFace token (for gated datasets)
export HF_TOKEN=your_token_here    # Linux/Mac
$env:HF_TOKEN = "your_token"       # PowerShell

# 3. Download datasets
python chatbot/data/download_data.py

# 4. Preprocess & build knowledge base
python chatbot/data/preprocess.py
python chatbot/data/process_classical_text.py
python chatbot/data/build_kb.py

# 5. Fine-tune model (or use on Colab)
python chatbot/models/fine_tune_mt5.py

# 6. Launch chatbot
streamlit run chatbot/app.py
```

---

## Current Status

| Phase | Description | Status |
|-------|-------------|--------|
| Project Setup | Config, requirements, folder structure | ✅ Complete |
| Data Download | BhashaBench-Ayur, Ayurveda Q&A, HiMed | ✅ Scripts ready |
| Data Preprocessing | Clean, translate, format, split | ✅ Complete |
| Classical Text Integration | Ashtanga Hridayam → 2,183 passages | ✅ Complete |
| FAISS Knowledge Base | Build vector index | ✅ Complete |
| Model Fine-Tuning | mT5-small + QLoRA | ✅ Script ready |
| RAG Pipeline | Retrieve + Generate + Translate | ✅ Complete |
| Web Interface | Streamlit chat UI | ✅ Complete |
| End-to-End Testing | Full pipeline test | 🔄 Pending |

---

## Next Steps

1. **Download all datasets** (once internet is available)
2. **Run preprocessing** to build the full knowledge base
3. **Fine-tune mT5-small** on Colab with the combined dataset
4. **Evaluate** using BhashaBench-Ayur benchmark questions
5. **Iterate** on response quality based on test results
