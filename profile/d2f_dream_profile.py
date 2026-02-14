"""
D2F Dream Model Profiling Example

This example demonstrates how to profile the performance
of Dream model with D2F decoding strategy using PyTorch Profiler.
"""
import os
import time
from pathlib import Path
import torch
from torch.profiler import profile, record_function, ProfilerActivity
from diffulex import Diffulex, SamplingParams
from transformers import AutoTokenizer


def main():
    model_path = "/data1/ckpts/Dream-org/Dream-v0-Base-7B"
    lora_path = "/data1/ckpts/SJTU-Deng-Lab/D2F_Dream_Base_7B_Lora"
    
    output_dir = Path("log/profiles")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading model...")
    model_load_start = time.time()
    llm = Diffulex(
        model_path,
        lora_path=lora_path,
        use_lora=True,
        model_name="dream",
        enforce_eager=True,
        tensor_parallel_size=1,
        data_parallel_size=1,
        gpu_memory_utilization=0.25,
        max_model_len=2048,
        decoding_strategy="d2f",
        mask_token_id=151666,
        diffusion_block_size=32,
        accept_threshold=0.95,
        complete_threshold=0.9,
        add_new_block_threshold=0.1,
    )
    model_load_time = time.time() - model_load_start
    print(f"Model loaded in {model_load_time:.2f} seconds")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    sampling_params = SamplingParams(temperature=0.0, max_tokens=256)
    
    prompts = [
        "What is 2+2?",
        "Explain quantum computing in simple terms.",
        "Write a Python function to calculate factorial.",
    ]
    
    print(f"\nStarting inference profiling with PyTorch Profiler...")
    
    # PyTorch Profiler
    trace_file = output_dir / "d2f_dream_trace.json"
    with_stack = os.getenv("DIFFULEX_PROFILE_WITH_STACK", "True").lower() == "true"
    
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=with_stack,
    ) as prof:
        with record_function("d2f_inference"):
            inference_start = time.time()
            outputs = llm.generate(prompts, sampling_params)
            inference_time = time.time() - inference_start
    
    # Export trace file
    prof.export_chrome_trace(str(trace_file))
    print(f"PyTorch Profiler trace saved to: {trace_file}")
    print(f"View trace at: https://ui.perfetto.dev/")
    
    # Calculate metrics
    total_tokens = sum(len(o.get('token_ids', [])) for o in outputs)
    num_outputs = len(outputs)
    avg_diff_steps = sum(o.get('n_diff_steps', 0) for o in outputs) / num_outputs if outputs else 0
    throughput = total_tokens / inference_time if inference_time > 0 else 0
    
    # Print summary
    print("\n" + "=" * 80)
    print("Profiling Summary")
    print("=" * 80)
    print(f"Model Loading Time: {model_load_time:.2f} seconds")
    print(f"Inference Time: {inference_time:.2f} seconds")
    print(f"Total Duration: {model_load_time + inference_time:.2f} seconds")
    print(f"\nInference Metrics:")
    print(f"  Number of Prompts: {num_outputs}")
    print(f"  Total Tokens: {total_tokens}")
    print(f"  Average Throughput: {throughput:.2f} tokens/sec")
    print(f"  Average Diffusion Steps: {avg_diff_steps:.2f}")
    print("=" * 80)
    
    print("\nGenerated Output Preview:")
    for idx, output in enumerate(outputs):
        print(f"\n[Prompt {idx + 1}]")
        print(f"Input: {prompts[idx]}")
        print(f"Output: {output.get('text', 'N/A')[:200]}...")
        print(f"Token Count: {len(output.get('token_ids', []))}")
        if 'n_diff_steps' in output:
            print(f"Diffusion Steps: {output['n_diff_steps']}")


if __name__ == "__main__":
    main()