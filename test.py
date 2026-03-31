from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

path = r"c:\Users\Shobhit\Downloads\LABS\Deep\Multilingual chatbot with Aryuvedic Domain\chatbot\models\mt5_ayurvedic_lora"

tok = AutoTokenizer.from_pretrained(path, use_fast=False)
mdl = AutoModelForSeq2SeqLM.from_pretrained(path)

q = "प्रश्न: अश्वगंधा के फायदे क्या हैं?"
inputs = tok(q, return_tensors="pt")
out = mdl.generate(**inputs, max_length=64, num_beams=4)
print(tok.decode(out[0], skip_special_tokens=True))
