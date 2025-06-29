import time
import torch
from simplestLLM import SimplestLLM

def test_performance():
    print("=== SimplestLLM Performance Test ===")
    
    # Test with float16 for speed
    print("\n1. Initializing model with float16...")
    model = SimplestLLM("meta-llama/Llama-3.2-1B-Instruct", context_length=2048, use_float16=True)
    
    prompts = [
        "The future of artificial intelligence is",
        "Machine learning algorithms work by",
        "Deep neural networks are"
    ]
    
    for i, prompt in enumerate(prompts, 1):
        print(f"\n{i}. Testing prompt: '{prompt}'")
        
        # Test with KV cache
        start_time = time.time()
        token_count = 0
        
        print("With KV cache: ", end="", flush=True)
        print(prompt, end=" ", flush=True)
        
        for token in model.generate(
            prompt, 
            max_new_tokens=30,
            temperature=0.8, 
            top_k=50,
            repetition_penalty=1.1,
            use_kv_cache=True
        ):
            print(token, end="", flush=True)
            token_count += 1
        
        total_time = time.time() - start_time
        speed_with_cache = token_count / total_time if total_time > 0 else 0
        
        print(f"\n   → {token_count} tokens in {total_time:.2f}s = {speed_with_cache:.2f} tok/s")
        
        # Test without KV cache for comparison
        start_time = time.time()
        token_count = 0
        
        print("Without KV cache: ", end="", flush=True)
        print(prompt, end=" ", flush=True)
        
        for token in model.generate(
            prompt, 
            max_new_tokens=30,
            temperature=0.8, 
            top_k=50,
            repetition_penalty=1.1,
            use_kv_cache=False
        ):
            print(token, end="", flush=True)
            token_count += 1
        
        total_time_no_cache = time.time() - start_time
        speed_no_cache = token_count / total_time_no_cache if total_time_no_cache > 0 else 0
        
        print(f"\n   → {token_count} tokens in {total_time_no_cache:.2f}s = {speed_no_cache:.2f} tok/s")
        
        speedup = speed_with_cache / speed_no_cache if speed_no_cache > 0 else 1
        print(f"   → Speedup: {speedup:.2f}x")
        
        # Check if we're hitting target performance
        target_speed = 17.7
        if speed_with_cache >= target_speed * 0.8:  # 80% of target
            print(f"   ✅ Good performance! (Target: {target_speed} tok/s)")
        else:
            print(f"   ⚠️  Below target performance (Target: {target_speed} tok/s)")
    
    print(f"\n{'='*50}")
    print("Performance test complete!")

if __name__ == "__main__":
    test_performance() 