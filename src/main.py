import os
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_model(model_name):
    print(f"Loading model: {model_name} ...")
    # Authenticate using your token (set in your environment or via login)
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        token=True,
        device_map="auto",
        low_cpu_mem_usage=True
    )
    if torch.cuda.is_available():
        print("CUDA is available; model has been dispatched automatically.")
    else:
        print("CUDA is not available. Running on CPU.")
    return tokenizer, model

def generate_response(tokenizer, model, prompt, max_length=100):
    inputs = tokenizer(prompt, return_tensors="pt")
    # Move inputs to GPU if available
    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
    
    print("Generation started...")
    start_time = time.time()
    outputs = model.generate(**inputs, max_length=max_length, do_sample=True, temperature=0.7)
    end_time = time.time()
    print("Generation finished.")
    print(f"Generation took {end_time - start_time:.2f} seconds.")
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

def main():
    # Use the official model identifier from Hugging Face.
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
    tokenizer, model = load_model(model_name)

    print("Welcome to Reasoning Playground!")
    print("Type your prompt and press Enter (type 'exit' to quit).")

    while True:
        user_input = input(">> ")
        if user_input.strip().lower() in ["exit", "quit"]:
            print("Goodbye!")
            break
        response = generate_response(tokenizer, model, user_input)
        print("\nResponse:")
        print(response)
        print("-" * 50)

if __name__ == "__main__":
    main()
