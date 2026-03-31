import json
import random
import os

def balance_dataset(filepath):
    print(f"Balancing {filepath}...")
    if not os.path.exists(filepath):
        print(f"Error: {filepath} not found.")
        return
        
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    mcq_data = [d for d in data if 'से संबंधित है' in d.get('answer_hi', '')]
    conv_data = [d for d in data if 'से संबंधित है' not in d.get('answer_hi', '')]
    
    print(f"  Original: {len(conv_data)} conversational, {len(mcq_data)} MCQ")
    
    random.seed(42)
    target_mcq_count = min(len(mcq_data), int(len(conv_data) * 0.3))
    
    if target_mcq_count > 0:
        mcq_sampled = random.sample(mcq_data, target_mcq_count)
    else:
        mcq_sampled = []
        
    balanced_data = conv_data + mcq_sampled
    random.shuffle(balanced_data)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(balanced_data, f, ensure_ascii=False, indent=2)
        
    print(f"  Success! New size: {len(balanced_data)} samples.")

if __name__ == "__main__":
    train_path = r"c:\Users\Shobhit\Downloads\LABS\Deep\Multilingual chatbot with Aryuvedic Domain\chatbot\data\processed\train.json"
    val_path = r"c:\Users\Shobhit\Downloads\LABS\Deep\Multilingual chatbot with Aryuvedic Domain\chatbot\data\processed\val.json"
    
    balance_dataset(train_path)
    balance_dataset(val_path)
    print("Done balancing datasets!")
