import time
import torch
from vllm import LLM, SamplingParams
from datasets import load_dataset
from evaluate import load
from huggingface_hub import login

def check_cuda():
    if torch.cuda.is_available():
        print("CUDA is available!")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        return True
    else:
        print("CUDA is not available. Using CPU.")
        return False

def load_model(model_name):
    # Login to Hugging Face
    # login()
    
    if check_cuda():
        return LLM(model=model_name, trust_remote_code=True, gpu_memory_utilization=0.8)
    else:
        return LLM(model=model_name, trust_remote_code=True, cpu_only=True)

def run_inference(llm, prompt, max_tokens=100):
    sampling_params = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=max_tokens)
    outputs = llm.generate([prompt], sampling_params)
    return outputs[0].outputs[0].text

def load_benchmark_data(dataset_name, split="test", num_samples=100):
    dataset = load_dataset(dataset_name, split=split)
    return dataset.select(range(min(num_samples, len(dataset))))

def run_benchmark(llm, dataset, metric_name="rouge"):
    metric = load(metric_name)
    total_time = 0
    results = []

    for item in dataset:
        prompt = item["prompt"]
        reference = item["target"]
        
        start_time = time.time()
        generated = run_inference(llm, prompt)
        end_time = time.time()
        
        total_time += end_time - start_time
        results.append(metric.compute(predictions=[generated], references=[reference]))
    
    avg_time = total_time / len(dataset)
    avg_score = sum(result[metric_name] for result in results) / len(results)
    
    return {
        "average_time": avg_time,
        f"average_{metric_name}_score": avg_score
    }

def main():
    model_name = "meta-llama/Llama-2-7b-hf"
    llm = load_model(model_name)
    
    dataset = load_benchmark_data("cnn_dailymail", "test", num_samples=10)
    
    benchmark_results = run_benchmark(llm, dataset)
    
    print(f"Benchmark results for {model_name}:")
    print(f"Average inference time: {benchmark_results['average_time']:.4f} seconds")
    print(f"Average ROUGE score: {benchmark_results['average_rouge_score']:.4f}")

if __name__ == "__main__":
    main()