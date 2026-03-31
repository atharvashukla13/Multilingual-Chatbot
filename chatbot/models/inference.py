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

        model_dir = FINE_TUNED_MODEL_DIR

        # Auto-fix Google Drive single-shard safetensors bug
        broken_name = os.path.join(model_dir, "model-001.safetensors")
        fixed_name = os.path.join(model_dir, "model.safetensors")
        if os.path.exists(broken_name) and not os.path.exists(fixed_name):
            os.rename(broken_name, fixed_name)

        has_local_config = os.path.exists(os.path.join(model_dir, "config.json"))
        local_weight_files = (
            "model.safetensors",
            "model.safetensors.index.json",
            "pytorch_model.bin",
            "pytorch_model.bin.index.json",
        )
        has_local_weights = (
            os.path.exists(model_dir)
            and any(os.path.exists(os.path.join(model_dir, name)) for name in local_weight_files)
        )
        has_local_slow_tokenizer = os.path.exists(os.path.join(model_dir, "spiece.model"))

        self.is_finetuned = has_local_config and has_local_weights

        if self.is_finetuned:
            print(f"Loading local full model from: {model_dir}")

            if not has_local_slow_tokenizer:
                raise FileNotFoundError(
                    f"Missing spiece.model in {model_dir}. Copy the full inference tokenizer files first."
                )

            self.tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                model_dir,
                device_map="auto" if torch.cuda.is_available() else None
            )
            if not torch.cuda.is_available():
                self.model = self.model.to(self.device)
        else:
            print(f"Local full model not found at {model_dir}")
            print(f"Falling back to base model: {BASE_MODEL_NAME}")
            self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, use_fast=False)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL_NAME)
            self.model = self.model.to(self.device)

        self.model.eval()
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
            input_text = f"प्रश्न: इन तथ्यों के आधार पर जानकारी दें: '{context}'। {query_hi}"
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
        response = response.replace("<extra_id_0>", "").strip()

        # If base model produces garbage (extra_id tokens), use retrieval fallback
        if not self.is_finetuned and len(response) < 10:
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
