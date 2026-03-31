"""
Inference module — loads fine-tuned mT5 model and generates responses.
"""

import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    BASE_MODEL_NAME, FINE_TUNED_MODEL_DIR,
    NUM_BEAMS, MAX_GENERATE_LENGTH, REPETITION_PENALTY, LENGTH_PENALTY
)


class AyurvedicGenerator:
    """Load fine-tuned mT5 model and generate Hindi Ayurvedic responses."""

    def __init__(self):
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading generator on: {self.device}")

        # Check if LoRA adapters exist
        adapter_config = os.path.join(FINE_TUNED_MODEL_DIR, "adapter_config.json")
        has_finetuned = os.path.exists(adapter_config)

        if has_finetuned:
            # Load tokenizer from fine-tuned dir
            self.tokenizer = AutoTokenizer.from_pretrained(FINE_TUNED_MODEL_DIR)

            # Load base model + merge LoRA adapters
            print(f"Loading base model: {BASE_MODEL_NAME}")
            from peft import PeftModel

            if torch.cuda.is_available():
                from transformers import BitsAndBytesConfig
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                )
                base_model = AutoModelForSeq2SeqLM.from_pretrained(
                    BASE_MODEL_NAME,
                    quantization_config=bnb_config,
                    device_map="auto",
                )
            else:
                base_model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL_NAME)
                base_model = base_model.to(self.device)

            print(f"Loading LoRA adapters: {FINE_TUNED_MODEL_DIR}")
            self.model = PeftModel.from_pretrained(base_model, FINE_TUNED_MODEL_DIR)
        else:
            # No fine-tuned model yet — use base model directly
            print(f"No fine-tuned model found at {FINE_TUNED_MODEL_DIR}")
            print(f"Using base model: {BASE_MODEL_NAME}")
            self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL_NAME)
            self.model = self.model.to(self.device)

        self.model.eval()
        self.is_finetuned = has_finetuned
        print("Generator loaded!")

    def generate(self, query_hi, context_passages=None):
        """
        Generate a Hindi response given a Hindi query and optional context passages.
        
        Args:
            query_hi: Hindi question string
            context_passages: List of Hindi passages from retrieval (optional)
            
        Returns:
            Hindi response string
        """
        # Build input text
        if context_passages:
            context = " ".join(context_passages[:3])  # Use top 3 passages
            input_text = f"प्रश्न: {query_hi} संदर्भ: {context}"
        else:
            input_text = f"प्रश्न: {query_hi}"

        # Tokenize
        inputs = self.tokenizer(
            input_text,
            max_length=512,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=MAX_GENERATE_LENGTH,
                num_beams=NUM_BEAMS,
                repetition_penalty=REPETITION_PENALTY,
                length_penalty=LENGTH_PENALTY,
                early_stopping=True,
            )

        # Decode
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # If base model produces garbage (extra_id tokens), use retrieval fallback
        if not self.is_finetuned and (not response.strip() or "extra_id" in response or len(response.strip()) < 10):
            if context_passages:
                # Return the most relevant passage as the answer
                return context_passages[0]
            return response

        return response


if __name__ == "__main__":
    # Quick test
    generator = AyurvedicGenerator()
    
    test_queries = [
        "अश्वगंधा के फायदे क्या हैं?",
        "वात दोष को कैसे संतुलित करें?",
        "त्रिफला क्या है?"
    ]
    
    for query in test_queries:
        print(f"\n❓ {query}")
        response = generator.generate(query)
        print(f"💬 {response}")
