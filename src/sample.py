import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# Hugging Face token (either set as environment variable or paste directly)
HF_TOKEN = os.getenv("HF_TOKEN") or "your_huggingface_token_here"

model_name = "google/gemma-2b"

# Configure 4-bit quantization (fits in 16GB RAM)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_TOKEN)

# Load model with quantization
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    quantization_config=bnb_config,
    token=HF_TOKEN
)

# Example prompt
prompt = "Write me a poem about Machine Learning."
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# Generate with decoding parameters to avoid repetition
outputs = model.generate(
    **inputs,
    max_new_tokens=200,
    do_sample=True,            # enable sampling
    top_p=0.9,                 # nucleus sampling
    temperature=0.7,           # randomness control
    repetition_penalty=1.2     # discourage loops
)

# Print result
print(tokenizer.decode(outputs[0], skip_special_tokens=True))