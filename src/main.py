import os
import time
import torch
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

def chain_generate_optimized(tokenizer, model, initial_prompt, max_new_tokens=512, temperature=0.5, final_marker="\\boxed{", max_iterations=100, context_token_limit=300):
    """
    Optimized iterative generation that:
      - Uses a sliding window to keep only the most recent context tokens.
      - Passes cached past_key_values so that previously computed tokens are not re‑processed.
      - Measures the time for each iteration and prints performance statistics.
    """
    complete_response = ""
    context = initial_prompt.strip()
    past_key_values = None
    iteration_times = []
    overall_start = time.perf_counter()

    for i in range(max_iterations):
        print(f"Iteration {i+1} of chaining...")
        # Build a recent context from the complete response using a sliding window.
        if complete_response:
            tokenized = tokenizer(complete_response, return_tensors="pt")
            token_ids = tokenized["input_ids"][0]
            # Keep only the last context_token_limit tokens.
            token_ids = token_ids[-context_token_limit:]
            recent_context = tokenizer.decode(token_ids, skip_special_tokens=True)
        else:
            recent_context = ""
        
        # Compose the prompt with the initial context and the recent response.
        prompt = f"{context}\n{recent_context}\n[Continue your reasoning concisely without repeating previous steps.]"
        prompt_inputs = tokenizer(prompt, return_tensors="pt")
        if torch.cuda.is_available():
            prompt_inputs = {k: v.to("cuda") for k, v in prompt_inputs.items()}

        print("Generation started...")
        iter_start = time.perf_counter()
        with torch.no_grad():
            # Use return_dict_in_generate to extract past_key_values for caching.
            outputs = model.generate(
                **prompt_inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=0.95,
                eos_token_id=tokenizer.eos_token_id,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict_in_generate=True
            )
        iter_elapsed = time.perf_counter() - iter_start
        iteration_times.append(iter_elapsed)
        print(f"Iteration {i+1} finished in {iter_elapsed:.2f} seconds.")
        
        # Update the cached past_key_values.
        past_key_values = outputs.past_key_values
        
        # Extract the newly generated tokens by skipping the prompt tokens.
        prompt_length = prompt_inputs["input_ids"].shape[-1]
        full_sequence = outputs.sequences[0]
        new_tokens = full_sequence[prompt_length:]
        new_text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        
        if not new_text:
            print("No additional text generated; breaking out of the loop.")
            break
        
        complete_response += new_text + "\n"
        if final_marker in complete_response:
            print("Final marker found; stopping iteration.")
            break

    overall_elapsed = time.perf_counter() - overall_start
    avg_iter = sum(iteration_times) / len(iteration_times) if iteration_times else 0
    print("\n--- Performance Summary ---")
    print(f"Total iterations: {len(iteration_times)}")
    print(f"Average time per iteration: {avg_iter:.2f} seconds")
    print(f"Total generation time: {overall_elapsed:.2f} seconds")
    return complete_response.strip()

def main():
    # Set your model name here (default or via the MODEL_NAME env variable)
    model_name = os.getenv("MODEL_NAME", "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
    tokenizer, model = load_model(model_name)
    
    print("Welcome to the optimized Reasoning Playground!")
    print("Type your prompt and press Enter (type 'exit' to quit).")
    print("Type 'reason' to use a default reasoning prompt for testing.")
    print("Type 'hello' for a simple chat example.\n")
    
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
            response = chain_generate_optimized(tokenizer, model, prompt, max_new_tokens=512, temperature=0.5)
        elif user_input.lower() == "hello":
            prompt = f"{simple_chat_prefix}\n{user_input}"
            inputs = tokenizer(prompt, return_tensors="pt")
            if torch.cuda.is_available():
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=450,
                    do_sample=True,
                    temperature=0.5,
                    top_p=0.95,
                    eos_token_id=tokenizer.eos_token_id
                )
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        else:
            prompt = user_input
            inputs = tokenizer(prompt, return_tensors="pt")
            if torch.cuda.is_available():
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=450,
                    do_sample=True,
                    temperature=0.5,
                    top_p=0.95,
                    eos_token_id=tokenizer.eos_token_id
                )
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        print("\nResponse:")
        print(response)
        print("-" * 50)
        log_conversation(user_input, response)

if __name__ == "__main__":
    main()
