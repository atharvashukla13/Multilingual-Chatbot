"""
Dataset Validation Script
- Checks for empty/garbage entries
- Verifies BhashaBench MCQ extraction against source
- Shows length distributions
- Prints random samples for manual inspection
"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import json
import os
import random

random.seed(42)

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED = os.path.join(BASE, "data", "processed")
DATASET = os.path.join(BASE, "dataset")


def load(name):
    with open(os.path.join(PROCESSED, name), "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================
# 1. BASIC INTEGRITY CHECKS
# ============================================================
print("=" * 70)
print("1. BASIC INTEGRITY CHECKS")
print("=" * 70)

train = load("train.json")
val = load("val.json")
test = load("test.json")
all_data = train + val + test

print(f"   Train: {len(train)} | Val: {len(val)} | Test: {len(test)} | Total: {len(all_data)}")

# Check for missing fields
missing_q = [i for i, d in enumerate(all_data) if not d.get("question_hi", "").strip()]
missing_a = [i for i, d in enumerate(all_data) if not d.get("answer_hi", "").strip()]
print(f"   Missing question: {len(missing_q)}")
print(f"   Missing answer:   {len(missing_a)}")

# Check for very short answers (< 5 chars = likely garbage)
short_a = [d for d in all_data if len(d.get("answer_hi", "")) < 10]
print(f"   Very short answers (<10 chars): {len(short_a)}")
if short_a:
    print("   Examples of short answers:")
    for s in short_a[:3]:
        print(f"     Q: {s['question_hi'][:60]}...")
        print(f"     A: {s['answer_hi']}")
        print()

# Check for extra_id tokens (mT5 garbage)
garbage = [d for d in all_data if "extra_id" in d.get("answer_hi", "") or "extra_id" in d.get("question_hi", "")]
print(f"   Entries with 'extra_id' tokens: {len(garbage)}")

# Check for entries that are just options letters (A, B, C, D)
option_only = [d for d in all_data if d.get("answer_hi", "").strip() in ["A", "B", "C", "D", "E"]]
print(f"   Answers that are just option letters: {len(option_only)}")

# ============================================================
# 2. LENGTH DISTRIBUTION
# ============================================================
print(f"\n{'=' * 70}")
print("2. LENGTH DISTRIBUTION (characters)")
print("=" * 70)

q_lens = [len(d["question_hi"]) for d in all_data]
a_lens = [len(d["answer_hi"]) for d in all_data]

print(f"   Questions: min={min(q_lens)}, max={max(q_lens)}, avg={sum(q_lens)//len(q_lens)}")
print(f"   Answers:   min={min(a_lens)}, max={max(a_lens)}, avg={sum(a_lens)//len(a_lens)}")

# Distribution buckets
print("\n   Answer length distribution:")
buckets = [(0, 20), (20, 50), (50, 100), (100, 200), (200, 500), (500, 1000), (1000, 5000), (5000, 99999)]
for lo, hi in buckets:
    count = sum(1 for l in a_lens if lo <= l < hi)
    pct = count * 100 // len(a_lens)
    bar = "█" * (pct // 2)
    print(f"   {lo:>5}-{hi:>5}: {count:>5} ({pct:>2}%) {bar}")

# ============================================================
# 3. SOURCE DISTRIBUTION
# ============================================================
print(f"\n{'=' * 70}")
print("3. SOURCE DISTRIBUTION")
print("=" * 70)

sources = {}
for d in all_data:
    src = d.get("source", "unknown")
    sources[src] = sources.get(src, 0) + 1

for src, cnt in sorted(sources.items(), key=lambda x: -x[1]):
    pct = cnt * 100 // len(all_data)
    print(f"   {src:>20}: {cnt:>6} ({pct}%)")

# ============================================================
# 4. BHASHBENCH MCQ VERIFICATION
# ============================================================
print(f"\n{'=' * 70}")
print("4. BHASHBENCH MCQ VERIFICATION (cross-check with source)")
print("=" * 70)

bb_path = os.path.join(DATASET, "bhashbench_ayur_hindi.json")
if os.path.exists(bb_path):
    with open(bb_path, "r", encoding="utf-8") as f:
        bb_data = json.load(f)

    option_map = {"A": "option_a", "B": "option_b", "C": "option_c", "D": "option_d"}
    correct = 0
    wrong = 0
    
    bb_train = [d for d in all_data if d.get("source") == "bhashbench_ayur"]
    
    # Verify 100 random samples
    samples = random.sample(bb_train, min(100, len(bb_train)))
    for sample in samples:
        q = sample["question_hi"]
        a = sample["answer_hi"]
        
        # Find matching question in source
        match = None
        for item in bb_data:
            if item.get("question", "").strip() == q.strip():
                match = item
                break
        
        if match:
            correct_key = option_map.get(match["correct_answer"].strip().upper())
            correct_text = match.get(correct_key, "")
            if correct_text and correct_text in a:
                correct += 1
            else:
                wrong += 1
                if wrong <= 2:  # Show first 2 mismatches
                    print(f"   ⚠️  MISMATCH:")
                    print(f"     Q: {q[:80]}...")
                    print(f"     Expected: {correct_text[:80]}")
                    print(f"     Got: {a[:80]}")
        
    print(f"   Verified {correct + wrong} BhashaBench entries: {correct} correct, {wrong} mismatches")
    print(f"   Accuracy: {correct * 100 // (correct + wrong)}%")

# ============================================================
# 5. HIMED COT QUALITY CHECK
# ============================================================
print(f"\n{'=' * 70}")
print("5. HIMED CHAIN-OF-THOUGHT QUALITY CHECK")
print("=" * 70)

himed = [d for d in all_data if d.get("source") in ("himed_bench", "himed_corpus")]
print(f"   Total HiMed entries: {len(himed)}")

# Check that answers are substantial (CoT should be longer than simple option letters)
substantial = sum(1 for d in himed if len(d["answer_hi"]) > 30)
print(f"   Substantial answers (>30 chars): {substantial} ({substantial * 100 // len(himed)}%)")

# ============================================================
# 6. CLASSICAL TEXT QUALITY CHECK
# ============================================================
print(f"\n{'=' * 70}")
print("6. CLASSICAL TEXT (ASHTANGA HRIDAYAM) QUALITY CHECK")
print("=" * 70)

classical = [d for d in all_data if d.get("source") == "classical_text"]
print(f"   Total passages: {len(classical)}")

# Check for Hindi content (should contain Devanagari characters)
devanagari = sum(1 for d in classical if any('\u0900' <= c <= '\u097F' for c in d["answer_hi"]))
print(f"   Contain Devanagari script: {devanagari} ({devanagari * 100 // max(1, len(classical))}%)")

# ============================================================
# 7. RANDOM SAMPLES (5 per source)
# ============================================================
print(f"\n{'=' * 70}")
print("7. RANDOM SAMPLES (3 per source for manual inspection)")
print("=" * 70)

for src in sorted(sources.keys()):
    src_data = [d for d in all_data if d.get("source") == src]
    samples = random.sample(src_data, min(3, len(src_data)))
    print(f"\n   --- {src.upper()} ---")
    for i, s in enumerate(samples):
        q = s["question_hi"][:120]
        a = s["answer_hi"][:200]
        print(f"   [{i+1}] Q: {q}")
        print(f"       A: {a}")
        print()

# ============================================================
# 8. DUPLICATE CHECK
# ============================================================
print(f"{'=' * 70}")
print("8. DUPLICATE CHECK")
print("=" * 70)

questions = [d["question_hi"] for d in all_data]
unique_q = set(questions)
print(f"   Total questions: {len(questions)}")
print(f"   Unique questions: {len(unique_q)}")
print(f"   Duplicates: {len(questions) - len(unique_q)}")

print(f"\n{'=' * 70}")
print("✅ VALIDATION COMPLETE")
print("=" * 70)
