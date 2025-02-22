import os
import time
import torch
import logging
import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer

# Custom formatter with microsecond precision.
class PreciseFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):
        dt = datetime.datetime.fromtimestamp(record.created)
        s = dt.strftime(datefmt) if datefmt else dt.strftime("%Y-%m-%d %H:%M:%S")
        return f"{s}.{dt.microsecond:06d}"

# Set up logging.
handler = logging.FileHandler("conversation_log.txt")
handler.setLevel(logging.INFO)
precise_formatter = PreciseFormatter("%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
handler.setFormatter(precise_formatter)
logging.getLogger().handlers = [handler]
logging.getLogger().setLevel(logging.INFO)

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

def chain_generate_optimized(tokenizer, model, initial_prompt, max_new_tokens=512, temperature=0.6,
                             final_marker="\\boxed{", max_iterations=100, context_token_limit=300):
    """
    Optimized iterative generation that:
      - Tokenizes the initial prompt once.
      - Maintains accumulated token IDs (avoiding repeated tokenization).
      - Uses a sliding window of the most recent tokens.
      - Caches past_key_values to reduce recomputation.
      - Measures and prints per-iteration timing.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Tokenize the prompt once.
    complete_tokens = tokenizer.encode(initial_prompt, return_tensors="pt").to(device)
    past_key_values = None
    iteration_times = []
    overall_start = time.perf_counter()

    for i in range(max_iterations):
        print(f"Iteration {i+1} of chaining...")
        # Use a sliding window for input tokens.
        input_tokens = complete_tokens[:, -context_token_limit:] if complete_tokens.shape[-1] > context_token_limit else complete_tokens

        print("Generation started...")
        iter_start = time.perf_counter()
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_tokens,
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
        print(f"Iteration {i+1} finished in {iter_elapsed:.6f} seconds.")

        past_key_values = outputs.past_key_values

        # Get new tokens (excluding the input prompt).
        prompt_length = input_tokens.shape[-1]
        new_tokens = outputs.sequences[:, prompt_length:]
        if new_tokens.shape[-1] == 0:
            print("No additional tokens generated; stopping.")
            break

        complete_tokens = torch.cat([complete_tokens, new_tokens], dim=-1)
        new_text = tokenizer.decode(new_tokens[0], skip_special_tokens=True).strip()
        print(f"New text: {new_text}")

        if not new_text:
            print("Empty output; stopping.")
            break

        if final_marker in new_text:
            print("Final marker found; stopping iteration.")
            break

    overall_elapsed = time.perf_counter() - overall_start
    avg_iter = sum(iteration_times) / len(iteration_times) if iteration_times else 0
    print("\n--- Performance Summary ---")
    print(f"Total iterations: {len(iteration_times)}")
    print(f"Average time per iteration: {avg_iter:.6f} seconds")
    print(f"Total generation time: {overall_elapsed:.6f} seconds")
    
    complete_text = tokenizer.decode(complete_tokens[0], skip_special_tokens=True)
    return complete_text.strip()

def main():
    # Use the Llama-based distilled model.
    model_name = os.getenv("MODEL_NAME", "deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
    tokenizer, model = load_model(model_name)
    
    print("Welcome to the optimized Reasoning Playground!")
    print("Type your prompt and press Enter (type 'exit' to quit).")
    print("Type 'reason' to use a default reasoning prompt for testing.")
    print("Type 'hello' for a simple chat example.\n")
    
    reasoning_prompt = (
        "Please solve the following problem step by step. "
        "Do not repeat any previous steps. "
        "Ensure your answer is complete and concise. "
        "Provide your final answer within \\boxed{...}.\n\n"
        "**Problem:**\n"
        "A train travels 300 miles in 5 hours and then 400 miles in 8 hours. "
        "What is its average speed for the entire journey?"
    )
    
    simple_chat_prefix = "You are a friendly chatbot. Respond briefly and simply."
    
    while True:
        user_input = input(">> ").strip()
        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break
        elif user_input.lower() == "reason":
            # Enforce model to start with a reasoning marker.
            prompt = "<think>\n" + reasoning_prompt
            response = chain_generate_optimized(tokenizer, model, prompt, max_new_tokens=512, temperature=0.6)
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
                    temperature=0.6,
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
                    temperature=0.6,
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
