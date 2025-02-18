import os
import time
import torch
import datetime
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer

# Configure structured logging
logging.basicConfig(
    filename="conversation_log.txt",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

def log_conversation(user_input, response):
    logging.info("User: %s", user_input)
    logging.info("Response: %s", response)
    logging.info("-" * 50)

def load_model(model_name):
    print(f"Loading model: {model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        device_map="auto",
        low_cpu_mem_usage=True
    )
    model.eval()
    if torch.cuda.is_available():
        print("CUDA is available; model dispatched automatically.")
    else:
        print("Running on CPU.")
    return tokenizer, model

def generate_response(tokenizer, model, prompt, max_new_tokens=450, temperature=0.5):
    inputs = tokenizer(prompt, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
    
    print("Generation started...")
    start_time = time.time()
    try:
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=0.95,
                eos_token_id=tokenizer.eos_token_id
            )
    except Exception as e:
        print("Error during generation:", e)
        return ""
    end_time = time.time()
    print(f"Generation finished in {end_time - start_time:.2f} seconds.")
    
    prompt_length = inputs["input_ids"].shape[-1]
    generated_tokens = outputs[0][prompt_length:]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return response

def chain_generate(tokenizer, model, initial_prompt, max_new_tokens=512, temperature=0.5, final_marker="\\boxed{", max_iterations=100, context_token_limit=300):
    """
    Iteratively generate output until the final marker is found.
    Uses a dynamic sliding window (by token count) to minimize redundancy.
    """
    complete_response = ""
    context = initial_prompt.strip()
    
    for i in range(max_iterations):
        print(f"Iteration {i+1} of chaining...")
        # Create a dynamic context: take tokens from the complete_response within a limit
        recent_lines = complete_response.splitlines()
        # Here we simply take the last few lines; in a more advanced version you might re-tokenize and trim by token count.
        recent_context = "\n".join(recent_lines[-3:]) if complete_response else ""
        prompt = f"{context}\n{recent_context}\n[Continue your reasoning concisely without repeating previous steps.]"
        partial_response = generate_response(tokenizer, model, prompt, max_new_tokens, temperature)
        if not partial_response.strip():
            print("No additional text generated; breaking out of the loop.")
            break

        # Trim any repeated text from the new output
        if complete_response and partial_response.startswith(complete_response):
            new_text = partial_response[len(complete_response):]
        else:
            new_text = partial_response
        
        complete_response += new_text.strip() + "\n"
        # Check if we've reached the final marker
        if final_marker in complete_response:
            print("Final marker found; stopping iteration.")
            break

    return complete_response.strip()

def main():
    model_name = os.getenv("MODEL_NAME", "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
    tokenizer, model = load_model(model_name)
    
    print("Welcome to Reasoning Playground!")
    print("Type your prompt and press Enter (type 'exit' to quit).")
    print("Type 'reason' to use a default reasoning prompt for testing.")
    print("For a simple chat, try typing 'hello'.\n")
    
    reasoning_prompt = (
        "Please solve the following problem step by step. "
        "Do not repeat any previously provided steps. "
        "Ensure your answer is complete and concise. "
        "Provide your final answer within \\boxed{...}.\n\n"
        "**Problem:**\n"
        "A train travels 300 miles in 5 hours and then 400 miles in 8 hours. "
        "What is its average speed for the entire journey?"
    )
    
    simple_chat_prefix = "You are a friendly chatbot. Respond in a short and simple manner."
    
    while True:
        user_input = input(">> ").strip()
        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break
        elif user_input.lower() == "reason":
            prompt = reasoning_prompt
            response = chain_generate(tokenizer, model, prompt, max_new_tokens=512, temperature=0.5)
        elif user_input.lower() == "hello":
            prompt = f"{simple_chat_prefix}\n{user_input}"
            response = generate_response(tokenizer, model, prompt, max_new_tokens=450, temperature=0.5)
        else:
            prompt = user_input
            response = generate_response(tokenizer, model, prompt, max_new_tokens=450, temperature=0.5)
        
        print("\nResponse:")
        print(response)
        print("-" * 50)
        log_conversation(user_input, response)

if __name__ == "__main__":
    main()