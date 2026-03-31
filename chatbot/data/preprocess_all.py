"""
Comprehensive Preprocessing Script — Merges ALL datasets into train/val/test.

Datasets processed:
  1. hindi_dataset.csv          — Orthopedic Q&A (~6.5K)
  2. bhashbench_ayur_hindi.json — Ayurvedic MCQ exam (~5.6K) -> RAG KB only
  3. himed_trad_bench.json      — Traditional medicine MCQ with CoT (~6K) -> Split between generative and RAG
  4. himed_trad_corpus.json     — Large corpus MCQ/QA/Dialogue (~286K, sampled to 20K) -> Split between generative and RAG
  5. classical_passages.json    — Ashtanga Hridayam passages (converted to Q&A)

Output:
  data/processed/train.json
  data/processed/val.json
  data/processed/test.json
  data/processed/knowledge_base.json
"""

import os
import json
import csv
import random
import re

random.seed(42)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
DATA_DIR = os.path.join(BASE_DIR, "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
RAW_DIR = os.path.join(DATA_DIR, "raw")

os.makedirs(PROCESSED_DIR, exist_ok=True)

# Global Knowledge Base accumulator for RAG
rag_kb = []


def load_json(path):
    """Load a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def add_to_rag(question, answer, category, source):
    """Add a Q&A pair to the RAG Knowledge Base."""
    if question and answer:
        passage = f"{question}\n{answer}"
        rag_kb.append({
            "question_hi": question,
            "answer_hi": answer,
            "passage_hi": passage,
            "category": category,
            "source": source
        })


def process_hindi_dataset():
    """Process hindi_dataset.csv — clinical records with symptoms/treatment."""
    csv_path = os.path.join(RAW_DIR, "hindi_dataset.csv")
    if not os.path.exists(csv_path):
        csv_path = os.path.join(DATASET_DIR, "hindi_dataset.csv")
    if not os.path.exists(csv_path):
        print("  ⚠️  hindi_dataset.csv not found, skipping")
        return []

    pairs = []
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            symptoms = row.get("symptoms", "").strip()
            treatment = row.get("treatment", "").strip()
            diagnosis = row.get("Diagnosis", "").strip()
            history = row.get("Patient History", "").strip()

            if not symptoms or not treatment or len(treatment) < 5:
                continue

            # Create Q&A pair: symptoms → treatment
            if diagnosis:
                question = f"{diagnosis} के लक्षण '{symptoms}' हैं। इसका उपचार क्या है?"
                answer = f"{treatment}। निदान: {diagnosis}।"
            else:
                question = f"इन लक्षणों का उपचार बताइए: {symptoms}"
                answer = treatment

            item = {
                "question_hi": question,
                "answer_hi": answer,
                "source": "hindi_dataset"
            }
            pairs.append(item)
            add_to_rag(question, answer, diagnosis if diagnosis else "general", "hindi_dataset")

            # Also create a diagnosis-based Q&A if we have history
            if diagnosis and history and len(history) > 10:
                q2 = f"{diagnosis} क्या है और इसका इलाज कैसे होता है?"
                a2 = f"{diagnosis} का उपचार: {treatment}। रोगी इतिहास: {history}"
                pairs.append({
                    "question_hi": q2,
                    "answer_hi": a2,
                    "source": "hindi_dataset"
                })
                add_to_rag(q2, a2, diagnosis, "hindi_dataset")

    print(f"  ✅ hindi_dataset.csv: {len(pairs)} generative pairs added")
    return pairs


def process_bhashbench():
    """Process BhashaBench Ayur Hindi — MCQ. Routes ENTIRELY to RAG KB, zero to generative."""
    path = os.path.join(DATASET_DIR, "bhashbench_ayur_hindi.json")
    if not os.path.exists(path):
        print("  ⚠️  bhashbench_ayur_hindi.json not found, skipping")
        return []

    data = load_json(path)
    count = 0
    option_map = {"A": "option_a", "B": "option_b", "C": "option_c", "D": "option_d"}

    for item in data:
        question = item.get("question", "").strip()
        correct = item.get("correct_answer", "").strip().upper()
        option_key = option_map.get(correct)

        if not question or not option_key:
            continue

        answer_text = item.get(option_key, "").strip()
        if not answer_text:
            continue

        topic = item.get("topic", "general")
        domain = item.get("subject_domain", "")
        
        answer = answer_text
        if topic and domain:
            answer = f"{answer_text}। यह {topic} ({domain}) से संबंधित है।"
        elif topic:
            answer = f"{answer_text}। ({topic})"

        # ONLY add to RAG KB
        add_to_rag(question, answer, topic, "bhashbench_ayur")
        count += 1

    print(f"  ✅ bhashbench_ayur_hindi.json: {count} pairs sent to RAG KB (0 to generative)")
    return []  # Return empty list for generative fine-tuning


def process_himed_bench():
    """Process HiMed-trad bench — Filter MCQs to RAG, QA/Dialogue to Generative."""
    path = os.path.join(DATASET_DIR, "himed_trad_bench.json")
    if not os.path.exists(path):
        print("  ⚠️  himed_trad_bench.json not found, skipping")
        return []

    data = load_json(path)
    pairs = []
    mcq_count = 0

    for item in data:
        question_raw = item.get("question", "").strip()
        answer_letter = item.get("answer", "").strip()
        cot = item.get("cot", "").strip()
        q_type = item.get("type", "MCQ")
        subject = item.get("subject", "general")

        if not question_raw:
            continue

        if q_type == "MCQ":
            lines = question_raw.split("\n")
            question = lines[0].strip()

            answer_text = ""
            for line in lines[1:]:
                line = line.strip()
                if line.startswith(f"{answer_letter}.") or line.startswith(f"{answer_letter} "):
                    answer_text = re.sub(r'^[A-E][\.\s]+', '', line).strip()
                    break

            if cot and len(cot) > 10:
                final_answer = cot
            elif answer_text:
                final_answer = answer_text
            else:
                continue

            # Route MCQ to RAG KB only
            add_to_rag(question, final_answer, subject, "himed_bench_mcq")
            mcq_count += 1
            continue

        elif q_type == "QA":
            question = question_raw
            final_answer = item.get("answer", "").strip()
            if cot and len(cot) > len(final_answer):
                final_answer = cot

        elif q_type == "Dialogue":
            lines = question_raw.split("\n")
            last_user_q = ""
            for line in lines:
                if line.strip().startswith("User:"):
                    last_user_q = line.replace("User:", "").strip()
            question = last_user_q if last_user_q else question_raw[:200]
            final_answer = item.get("answer", "").strip()
        else:
            continue

        if question and final_answer and len(final_answer) > 5:
            pairs.append({
                "question_hi": question,
                "answer_hi": final_answer,
                "source": "himed_bench"
            })
            add_to_rag(question, final_answer, subject, "himed_bench")

    print(f"  ✅ himed_trad_bench.json: {len(pairs)} generative pairs added | {mcq_count} sent to RAG KB")
    return pairs


def process_himed_corpus(max_qa=10000, max_dialogue=10000):
    """Process HiMed-trad corpus — MCQs to RAG KB, QA/Dialogue to Generative (increased max)."""
    path = os.path.join(DATASET_DIR, "himed_trad_corpus.json")
    if not os.path.exists(path):
        print("  ⚠️  himed_trad_corpus.json not found, skipping")
        return []

    print("  🔄 Loading HiMed corpus (large file, may take a minute)...")
    data = load_json(path)
    print(f"  📊 Total corpus entries: {len(data)}")

    mcq_items = []
    qa_items = []
    dialogue_items = []

    for item in data:
        q_type = item.get("type", "MCQ")
        if q_type == "MCQ":
            mcq_items.append(item)
        elif q_type == "QA":
            qa_items.append(item)
        elif q_type == "Dialogue":
            dialogue_items.append(item)

    print(f"     Found -> MCQ: {len(mcq_items)}, QA: {len(qa_items)}, Dialogue: {len(dialogue_items)}")

    # Sample QA and Dialogue for Generative Data
    qa_sample = min(len(qa_items), max_qa)
    dialogue_sample = min(len(dialogue_items), max_dialogue)
    
    sampled_generative = []
    sampled_generative.extend(random.sample(qa_items, qa_sample) if qa_sample < len(qa_items) else qa_items)
    sampled_generative.extend(random.sample(dialogue_items, dialogue_sample) if dialogue_sample < len(dialogue_items) else dialogue_items)

    pairs = []
    for item in sampled_generative:
        question_raw = item.get("question", "").strip()
        answer_raw = item.get("answer", "").strip()
        cot = item.get("cot", "").strip()
        q_type = item.get("type", "MCQ")
        subject = item.get("subject", "general")

        if not question_raw:
            continue

        if q_type == "QA":
            question = question_raw
            final_answer = cot if (cot and len(cot) > len(answer_raw)) else answer_raw

        elif q_type == "Dialogue":
            lines = question_raw.split("\n")
            last_user_q = ""
            for line in lines:
                if line.strip().startswith("User:"):
                    last_user_q = line.replace("User:", "").strip()
            question = last_user_q if last_user_q else question_raw[:200]
            final_answer = answer_raw
        else:
            continue

        if question and final_answer and len(final_answer) > 5:
            pairs.append({
                "question_hi": question,
                "answer_hi": final_answer,
                "source": "himed_corpus"
            })
            add_to_rag(question, final_answer, subject, "himed_corpus")
            
    # Add ALL MCQ items to RAG KB (or a large sample if too huge, let's take up to 20,000 for KB to keep embedding reasonable)
    mcq_sample_size = min(len(mcq_items), 20000)
    sampled_mcqs = random.sample(mcq_items, mcq_sample_size) if mcq_sample_size < len(mcq_items) else mcq_items
    
    for item in sampled_mcqs:
        question_raw = item.get("question", "").strip()
        answer_raw = item.get("answer", "").strip()
        cot = item.get("cot", "").strip()
        subject = item.get("subject", "general")
        
        lines = question_raw.split("\n")
        question = lines[0].strip()
        final_answer = cot if (cot and len(cot) > 10) else answer_raw
        
        if question and final_answer and len(final_answer) > 5:
            add_to_rag(question, final_answer, subject, "himed_corpus_mcq")

    print(f"  ✅ himed_trad_corpus.json: {len(pairs)} generative pairs added | {mcq_sample_size} sent to RAG KB")
    return pairs


def process_classical_passages():
    """Convert Ashtanga Hridayam passages into Q&A pairs."""
    path = os.path.join(PROCESSED_DIR, "classical_passages.json")
    if not os.path.exists(path):
        print("  ⚠️  classical_passages.json not found, skipping")
        return []

    data = load_json(path)
    pairs = []

    for passage in data:
        text = passage.get("passage_hi", "").strip()
        category = passage.get("category", "").strip()

        if not text or len(text) < 30:
            continue

        if category:
            question = f"{category} के बारे में बताइए।"
        else:
            first_line = text.split("।")[0].strip()[:60]
            question = f"{first_line} के बारे में विस्तार से बताइए।"

        pairs.append({
            "question_hi": question,
            "answer_hi": text[:500],
            "source": "classical_text"
        })

    print(f"  ✅ classical_passages.json: {len(pairs)} pairs")
    return pairs


def split_data(all_pairs, train_ratio=0.85, val_ratio=0.10, test_ratio=0.05):
    """Split into train/val/test."""
    random.shuffle(all_pairs)
    n = len(all_pairs)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train = all_pairs[:train_end]
    val = all_pairs[train_end:val_end]
    test = all_pairs[val_end:]

    return train, val, test


def main():
    print("=" * 60)
    print("🔧 Preprocessing ALL datasets for mT5 fine-tuning")
    print("=" * 60)

    all_pairs = []

    # 1. Hindi Dataset (orthopedic Q&A)
    print("\n📖 [1/5] Hindi Dataset (Orthopedic Q&A)...")
    all_pairs.extend(process_hindi_dataset())

    # 2. BhashaBench Ayur (exam MCQs)
    print("\n📖 [2/5] BhashaBench Ayur (Exam MCQs)...")
    all_pairs.extend(process_bhashbench())

    # 3. HiMed-trad bench (benchmark MCQs with explanations)
    print("\n📖 [3/5] HiMed-trad Bench (MCQ with CoT)...")
    all_pairs.extend(process_himed_bench())

    # 4. HiMed-trad corpus (large corpus)
    print("\n📖 [4/5] HiMed-trad Corpus...")
    all_pairs.extend(process_himed_corpus(max_qa=10000, max_dialogue=10000))

    # 5. Classical passages (Ashtanga Hridayam)
    print("\n📖 [5/5] Classical Passages (Ashtanga Hridayam)...")
    all_pairs.extend(process_classical_passages())

    # Summary
    print(f"\n{'=' * 60}")
    print(f"📊 TOTAL generative pairs collected: {len(all_pairs)}")
    print(f"📚 TOTAL RAG Knowledge Base passages collected: {len(rag_kb)}")

    # Splitting Generative Data
    train, val, test = split_data(all_pairs)
    print(f"\n📦 Split: Train={len(train)} | Val={len(val)} | Test={len(test)}")

    # Save Generative Data
    for name, data in [("train.json", train), ("val.json", val), ("test.json", test)]:
        path = os.path.join(PROCESSED_DIR, name)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"   💾 Saved Generative: {path}")

    # Save RAG Knowledge Base
    kb_path = os.path.join(PROCESSED_DIR, "knowledge_base.json")
    with open(kb_path, "w", encoding="utf-8") as f:
        json.dump(rag_kb, f, ensure_ascii=False, indent=2)
    print(f"   💾 Saved RAG Knowledge Base: {kb_path} ({len(rag_kb)} entries)")

    print(f"\n{'=' * 60}")
    print(f"✅ Preprocessing complete!")
    print(f"   Generative models: Upload train.json and val.json to Colab")
    print(f"   RAG framework: Use knowledge_base.json to build the FAISS index")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
