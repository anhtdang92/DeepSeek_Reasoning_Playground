import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_model(model_name):
    print(f"Loading model: {model_name} ...")
    
    # If the model is private, you need to be logged in or pass use_auth_token=True
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, 
        use_auth_token=True,
        trust_remote_code=True  # if required by the model
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        use_auth_token=True,
        trust_remote_code=True,   # if required
        device_map="auto",
        low_cpu_mem_usage=True,   # or load_in_8bit=True if needed
    )
    
    if torch.cuda.is_available():
        print("CUDA is available; model has been dispatched automatically.")
    else:
        print("CUDA is not available. Running on CPU.")
    
    return tokenizer, model

def generate_response(tokenizer, model, prompt, max_length=50, temperature=0.7):
    inputs = tokenizer(prompt, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
    
    print("Generation started...")
    start_time = time.time()
    outputs = model.generate(
        **inputs,
        max_length=max_length,
        do_sample=True,
        temperature=temperature,
    )
    end_time = time.time()
    print("Generation finished.")
    print(f"Generation took {end_time - start_time:.2f} seconds.")
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def main():
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-8B"
    tokenizer, model = load_model(model_name)

    print("Welcome to Reasoning Playground!")
    print("Type your prompt and press Enter (type 'exit' to quit).")

    while True:
        user_input = input(">> ")
        if user_input.strip().lower() in ["exit", "quit"]:
            print("Goodbye!")
            break
        
        response = generate_response(tokenizer, model, user_input, max_length=50)
        print("\nResponse:")
        print(response)
        print("-" * 50)

if __name__ == "__main__":
    main()