# DeepSeek_Reasoning_Playground

This project is an experimental playground to test and compare the reasoning capabilities of DeepSeek models against popular OpenAI models. The goal is to evaluate the performance, response quality, and efficiency of the **deepseek-ai/DeepSeek-R1-Distill-Qwen-32B** model when running on consumer-grade hardware.

## Project Scope

- **Experimentation:**  
  Test the capabilities of **deepseek-ai/DeepSeek-R1-Distill-Qwen-32B** by generating responses for various prompts, especially those that challenge reasoning, math, and coding abilities.

- **Comparison:**  
  Benchmark its performance against OpenAI models (such as GPT-4) in terms of response accuracy, generation speed, and overall effectiveness in handling complex tasks.

- **Evaluation Metrics:**  
  Measure tokens per second, reasoning quality, response diversity, and other qualitative differences that can guide future improvements or adaptations.

## Hardware Setup

This project will be executed on the following hardware configuration:

- **GPU:** NVIDIA RTX 4090  
- **Memory:** 64 GB RAM  
- **CPU:** AMD Ryzen 9 5950X  

This configuration represents a high-end consumer-level PC, providing a practical environment for running advanced AI models locally.

## Getting Started

### 1. Clone the Repository:
```bash
git clone https://github.com/anhtdang92/DeepSeek_Reasoning_Playground.git
cd DeepSeek_Reasoning_Playground
```

### 2. Set Up the Virtual Environment:

#### Create the Virtual Environment:
```bash
python -m venv venv
```

#### Activate the Virtual Environment:

- **On Windows (PowerShell):**
  ```powershell
  .venv\Scripts\Activate.ps1
  ```

- **On macOS/Linux:**
  ```bash
  source venv/bin/activate
  ```

### 3. Install Dependencies:
```bash
pip install -r requirements.txt
```

### 4. Run the Main Script:
```bash
python src/main.py
```

## Next Steps

- **Model Evaluation:**  
  Experiment with different prompt types and compare the model's responses with those from OpenAI models.

- **Performance Analysis:**  
  Monitor tokens per second and resource usage to evaluate efficiency.

- **Further Enhancements:**  
  Consider integrating additional testing frameworks or visualization tools to better compare model performance.

## License

This project is licensed under the terms of the MIT License. See the [LICENSE](LICENSE) file for details.