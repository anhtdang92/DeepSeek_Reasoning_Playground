import os
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_model(model_name):
    print(f"Loading model: {model_name} ...")
    # If the model is private, ensure you're authenticated (token=True picks up your stored token)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, 
        token=True,
        trust_remote_code=True  # if required by the model
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        token=True,
        trust_remote_code=True,   # if required
        device_map="auto",
        low_cpu_mem_usage=True   # or load_in_8bit=True if needed
    )
    
    # Set the model to evaluation mode to disable dropout and gradient computations
    model.eval()

    if torch.cuda.is_available():
        print("CUDA is available; model has been dispatched automatically.")
    else:
        print("CUDA is not available. Running on CPU.")
    
    return tokenizer, model

def generate_response(tokenizer, model, prompt, max_new_tokens=50, temperature=0.7):
    inputs = tokenizer(prompt, return_tensors="pt")
    # Move inputs to GPU if available
    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
    
    print("Generation started...")
    start_time = time.time()
    # Disable gradient computation for inference
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
        )
    end_time = time.time()
    print("Generation finished.")
    print(f"Generation took {end_time - start_time:.2f} seconds.")
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def main():
    # Use the 7B model identifier for faster inference on consumer hardware.
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    tokenizer, model = load_model(model_name)

    print("Welcome to Reasoning Playground!")
    print("Type your prompt and press Enter (type 'exit' to quit).")
    print("Type 'reason' to use a default reasoning prompt for testing.")
    print("For a simple chat, try typing 'hello'.\n")

    # Default reasoning prompt that encourages step-by-step reasoning
    reasoning_prompt = (
        "Please solve the following problem step by step:\n"
        "A train travels 300 miles in 5 hours and then 400 miles in 8 hours. "
        "What is its average speed for the entire journey? "
        "Present your step-by-step reasoning and provide your final answer within \\boxed{...}."
    )
    
    # Simple chat prefix to encourage succinct responses
    simple_chat_prefix = "You are a friendly chatbot. Respond in a short and simple manner."
    
    while True:
        user_input = input(">> ")
        if user_input.strip().lower() in ["exit", "quit"]:
            print("Goodbye!")
            break
        elif user_input.strip().lower() == "reason":
            prompt = reasoning_prompt
            max_new_tokens = 500  # allow longer output for reasoning
        elif user_input.strip().lower() == "hello":
            prompt = f"{simple_chat_prefix}\n{user_input}"
            max_new_tokens = 50
        else:
            prompt = user_input
            max_new_tokens = 50
        
        response = generate_response(tokenizer, model, prompt, max_new_tokens=max_new_tokens)
        print("\nResponse:")
        print(response)
        print("-" * 50)

if __name__ == "__main__":
    main()