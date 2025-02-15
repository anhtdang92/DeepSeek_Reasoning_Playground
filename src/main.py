import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_model(model_name):
    print(f"Loading model: {model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=True)
    if torch.cuda.is_available():
        model = model.to("cuda")
        print("Model moved to CUDA.")
    else:
        print("CUDA is not available. Running on CPU.")
    return tokenizer, model

def generate_response(tokenizer, model, prompt, max_length=100):
    inputs = tokenizer(prompt, return_tensors="pt")
    # If using GPU, move inputs to CUDA as well.
    if torch.cuda.is_available():
        inputs = {key: value.to("cuda") for key, value in inputs.items()}
    outputs = model.generate(**inputs, max_length=max_length, do_sample=True, temperature=0.7)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

def main():
    # Update this with the actual Hugging Face model ID for DeepSeek-R1-Distill-Qwen-32B
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