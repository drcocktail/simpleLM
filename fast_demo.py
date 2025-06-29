#!/usr/bin/env python3
"""
FastLLM Demo - Shows instant startup after first load
"""

import time
from fast_llm import FastLLM

def demo_fast_startup():
    print("🚀 FastLLM Demo - Ollama-style instant startup")
    print("=" * 60)
    
    # First load - will take ~20 seconds
    print("\n1️⃣  First time loading (this will be slow but only happens once):")
    start_time = time.time()
    model1 = FastLLM("meta-llama/Llama-3.2-1B-Instruct", context_length=2048, use_float16=True)
    first_load_time = time.time() - start_time
    print(f"First load took: {first_load_time:.2f} seconds")
    
    # Generate some text
    prompt1 = "The weather today is"
    print(f"\nGenerating: '{prompt1}'")
    print(prompt1, end=" ", flush=True)
    for token in model1.generate(prompt1, max_new_tokens=25, temperature=0.8):
        print(token, end="", flush=True)
    
    print("\n\n" + "=" * 60)
    
    # Second load - should be instant!
    print("\n2️⃣  Second time loading (should be instant!):")
    start_time = time.time()
    model2 = FastLLM("meta-llama/Llama-3.2-1B-Instruct", context_length=2048, use_float16=True)
    second_load_time = time.time() - start_time
    print(f"Second load took: {second_load_time:.3f} seconds")
    
    # Generate different text
    prompt2 = "Once upon a time, there was"
    print(f"\nGenerating: '{prompt2}'")
    print(prompt2, end=" ", flush=True)
    for token in model2.generate(prompt2, max_new_tokens=25, temperature=0.8):
        print(token, end="", flush=True)
    
    print("\n\n" + "=" * 60)
    
    # Third load - still instant!
    print("\n3️⃣  Third time loading (still instant!):")
    start_time = time.time()
    model3 = FastLLM("meta-llama/Llama-3.2-1B-Instruct", context_length=2048, use_float16=True)
    third_load_time = time.time() - start_time
    print(f"Third load took: {third_load_time:.3f} seconds")
    
    # Generate more text
    prompt3 = "The future of artificial intelligence"
    print(f"\nGenerating: '{prompt3}'")
    print(prompt3, end=" ", flush=True)
    for token in model3.generate(prompt3, max_new_tokens=25, temperature=0.8):
        print(token, end="", flush=True)
    
    print("\n\n" + "=" * 60)
    print("📊 PERFORMANCE SUMMARY:")
    print(f"First load:  {first_load_time:.2f}s (normal - model gets cached)")
    print(f"Second load: {second_load_time:.3f}s (⚡ instant!)")
    print(f"Third load:  {third_load_time:.3f}s (⚡ instant!)")
    
    speedup = first_load_time / max(second_load_time, 0.001)
    print(f"Speedup: {speedup:.0f}x faster after caching!")
    
    print("\n✨ This is how Ollama achieves instant startup - by keeping models in memory!")
    
    # Show cache info
    print("\n📋 Cache status:")
    FastLLM.list_cached_models()

if __name__ == "__main__":
    demo_fast_startup() 