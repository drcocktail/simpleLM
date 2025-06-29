#!/usr/bin/env python3
"""
Performance test comparing different optimization strategies
"""

import time
import torch
import psutil
import os
from simplestLLM import SimplestLLM

def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def performance_test():
    print("=" * 80)
    print("SimplestLLM Performance Comparison")
    print("=" * 80)
    
    prompt = "The future of artificial intelligence is"
    max_tokens = 200
    
    test_configs = [
        {"name": "Float32 + No KV Cache", "use_float16": False, "use_kv_cache": False},
        {"name": "Float32 + KV Cache", "use_float16": False, "use_kv_cache": True},
        {"name": "Float16 + KV Cache", "use_float16": True, "use_kv_cache": True},
    ]
    
    results = []
    
    for config in test_configs:
        print(f"\n{'='*60}")
        print(f"Testing: {config['name']}")
        print(f"{'='*60}")
        
        # Initialize model
        print("Initializing model...")
        init_start = time.time()
        initial_memory = get_memory_usage()
        
        model = SimplestLLM(
            "meta-llama/Llama-3.2-1B-Instruct", 
            context_length=4096,
            use_float16=config["use_float16"]
        )
        
        init_time = time.time() - init_start
        model_memory = get_memory_usage()
        memory_increase = model_memory - initial_memory
        
        print(f"Model loaded in {init_time:.2f}s, Memory: +{memory_increase:.1f}MB")
        
        # Warm up
        print("Warming up...")
        warmup_tokens = 0
        for token in model.generate(
            "Test", 
            max_new_tokens=10, 
            temperature=0.7, 
            use_kv_cache=config["use_kv_cache"]
        ):
            warmup_tokens += 1
        
        # Performance test
        print(f"Generating {max_tokens} tokens...")
        start_time = time.time()
        token_count = 0
        generated_text = prompt
        
        try:
            for token in model.generate(
                prompt,
                max_new_tokens=max_tokens,
                temperature=0.7,
                top_k=50,
                use_kv_cache=config["use_kv_cache"]
            ):
                generated_text += token
                token_count += 1
                
                # Print progress every 50 tokens
                if token_count % 50 == 0:
                    elapsed = time.time() - start_time
                    speed = token_count / elapsed
                    print(f"  Progress: {token_count}/{max_tokens} tokens ({speed:.1f} tok/s)")
        
        except Exception as e:
            print(f"Error during generation: {e}")
        
        total_time = time.time() - start_time
        final_memory = get_memory_usage()
        peak_memory = final_memory - initial_memory
        
        if total_time > 0:
            tokens_per_sec = token_count / total_time
        else:
            tokens_per_sec = 0
        
        # Store results
        result = {
            "config": config["name"],
            "tokens_generated": token_count,
            "time_taken": total_time,
            "tokens_per_sec": tokens_per_sec,
            "init_time": init_time,
            "peak_memory": peak_memory,
            "model_memory": memory_increase
        }
        results.append(result)
        
        print(f"\nResults for {config['name']}:")
        print(f"  Tokens generated: {token_count}")
        print(f"  Time taken: {total_time:.2f}s")
        print(f"  Speed: {tokens_per_sec:.2f} tokens/second")
        print(f"  Peak memory: {peak_memory:.1f}MB")
        
        # Clean up
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        time.sleep(2)  # Allow memory to be freed
    
    # Summary comparison
    print(f"\n{'='*80}")
    print("PERFORMANCE SUMMARY")
    print(f"{'='*80}")
    
    print(f"{'Configuration':<25} {'Speed (tok/s)':<15} {'Memory (MB)':<12} {'Init Time (s)':<12}")
    print("-" * 80)
    
    for result in results:
        print(f"{result['config']:<25} {result['tokens_per_sec']:<15.2f} {result['peak_memory']:<12.1f} {result['init_time']:<12.2f}")
    
    # Calculate improvements
    if len(results) >= 3:
        baseline = results[0]  # Float32 + No KV Cache
        kv_cache = results[1]   # Float32 + KV Cache  
        float16_kv = results[2] # Float16 + KV Cache
        
        print(f"\n{'='*80}")
        print("IMPROVEMENT ANALYSIS")
        print(f"{'='*80}")
        
        if baseline['tokens_per_sec'] > 0:
            kv_speedup = kv_cache['tokens_per_sec'] / baseline['tokens_per_sec']
            float16_speedup = float16_kv['tokens_per_sec'] / baseline['tokens_per_sec']
            
            print(f"KV Cache speedup: {kv_speedup:.2f}x")
            print(f"Float16 + KV Cache speedup: {float16_speedup:.2f}x")
        
        memory_reduction = (baseline['peak_memory'] - float16_kv['peak_memory']) / baseline['peak_memory'] * 100
        print(f"Memory reduction with Float16: {memory_reduction:.1f}%")

def stress_test():
    """Test with longer sequences to stress test the optimizations"""
    print(f"\n{'='*80}")
    print("STRESS TEST - Long Sequence Generation")
    print(f"{'='*80}")
    
    prompt = "In the distant future, humanity has spread across the galaxy. The year is 3024, and"
    
    print("Testing with optimized configuration (Float16 + KV Cache)...")
    
    model = SimplestLLM(
        "meta-llama/Llama-3.2-1B-Instruct", 
        context_length=8192,
        use_float16=True
    )
    
    start_time = time.time()
    token_count = 0
    max_tokens = 1000
    
    print(f"Generating {max_tokens} tokens...")
    print(prompt, end="", flush=True)
    
    try:
        for token in model.generate(
            prompt,
            max_new_tokens=max_tokens,
            temperature=0.8,
            top_k=100,
            use_kv_cache=True
        ):
            print(token, end="", flush=True)
            token_count += 1
            
            if token_count % 100 == 0:
                elapsed = time.time() - start_time
                speed = token_count / elapsed
                print(f"\n[{token_count}/{max_tokens} tokens, {speed:.1f} tok/s]", end="", flush=True)
    
    except Exception as e:
        print(f"\nError: {e}")
    
    total_time = time.time() - start_time
    final_speed = token_count / total_time if total_time > 0 else 0
    
    print(f"\n\nStress test completed!")
    print(f"Generated {token_count} tokens in {total_time:.2f}s")
    print(f"Average speed: {final_speed:.2f} tokens/second")

if __name__ == "__main__":
    performance_test()
    stress_test() 