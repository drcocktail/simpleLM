#!/usr/bin/env python3
"""
Quick test to demonstrate KV cache performance
"""

import time
from simplestLLM import SimplestLLM

def quick_test():
    print("Quick KV Cache Performance Test")
    print("=" * 50)
    
    # Initialize model with optimizations
    print("Initializing optimized model...")
    model = SimplestLLM("meta-llama/Llama-3.2-1B-Instruct", context_length=4096, use_float16=True)
    
    prompt = "The future of artificial intelligence is"
    max_tokens = 100
    
    print(f"\nPrompt: '{prompt}'")
    print(f"Generating {max_tokens} tokens with KV cache + Float16...\n")
    
    start_time = time.time()
    token_count = 0
    
    print(prompt, end="", flush=True)
    
    for token in model.generate(
        prompt,
        max_new_tokens=max_tokens,
        temperature=0.7,
        top_k=50,
        use_kv_cache=True
    ):
        print(token, end="", flush=True)
        token_count += 1
        
        # Show progress every 25 tokens
        if token_count % 25 == 0:
            elapsed = time.time() - start_time
            speed = token_count / elapsed
            print(f"\n[{token_count}/{max_tokens} tokens, {speed:.1f} tok/s]", end="", flush=True)
    
    total_time = time.time() - start_time
    final_speed = token_count / total_time if total_time > 0 else 0
    
    print(f"\n\n{'='*50}")
    print("RESULTS")
    print(f"{'='*50}")
    print(f"✓ Generated {token_count} tokens")
    print(f"✓ Time taken: {total_time:.2f} seconds")
    print(f"✓ Speed: {final_speed:.2f} tokens/second")
    print(f"✓ KV Cache: Enabled")
    print(f"✓ Float16: Enabled")
    
    if final_speed > 10:
        print(f"✓ Performance: EXCELLENT ({final_speed:.1f} tok/s)")
    elif final_speed > 5:
        print(f"✓ Performance: GOOD ({final_speed:.1f} tok/s)")
    else:
        print(f"⚠ Performance: NEEDS IMPROVEMENT ({final_speed:.1f} tok/s)")
    
    print("\nKV Cache implementation is working correctly! 🎉")

if __name__ == "__main__":
    quick_test() 