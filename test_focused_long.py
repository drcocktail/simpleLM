#!/usr/bin/env python3
"""
Focused test for long generation with repetition handling
"""

import time
import torch
from simplestLLM import SimplestLLM

def test_focused_long_generation():
    print("=" * 80)
    print("Focused Long Generation Test with Repetition Handling")
    print("=" * 80)
    
    # Test with longer context length for better long-form generation
    print("Initializing model with 8192 context length...")
    start_time = time.time()
    model = SimplestLLM("meta-llama/Llama-3.2-1B-Instruct", context_length=8192)
    init_time = time.time() - start_time
    print(f"Model initialized in {init_time:.2f} seconds")
    
    # Test case optimized for long generation
    prompt = "The evolution of artificial intelligence has been marked by several key breakthroughs. Starting from the early days of symbolic AI in the 1950s"
    
    print(f"\nPrompt: {prompt}")
    print(f"\nGenerating text with better parameters...")
    print("-" * 80)
    
    # Parameters optimized to reduce repetition
    max_tokens = 4096
    temperature = 0.8  # Higher temperature for more diversity
    top_k = 100        # Larger top_k for more variety
    
    start_time = time.time()
    token_count = 0
    generated_text = prompt
    
    try:
        print(prompt, end="", flush=True)
        
        for token in model.generate(
            prompt,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k
        ):
            print(token, end="", flush=True)
            generated_text += token
            token_count += 1
            
            # Progress updates
            if token_count % 500 == 0:
                elapsed = time.time() - start_time
                tokens_per_sec = token_count / elapsed
                print(f"\n\n[Progress: {token_count}/{max_tokens} tokens ({token_count/max_tokens*100:.1f}%), Speed: {tokens_per_sec:.1f} tok/s]\n", end="", flush=True)
            
            # Early stopping if we detect excessive repetition
            if token_count > 100:
                # Check last 200 characters for repetition
                recent_text = generated_text[-200:]
                words = recent_text.split()
                if len(words) > 10:
                    # Count repeated consecutive patterns
                    repeated_sequences = 0
                    for i in range(len(words) - 3):
                        if words[i:i+2] == words[i+2:i+4]:
                            repeated_sequences += 1
                    
                    if repeated_sequences > 5:  # Too much repetition
                        print(f"\n\n[EARLY STOP: Excessive repetition detected after {token_count} tokens]")
                        break
        
        # Final stats
        total_time = time.time() - start_time
        tokens_per_sec = token_count / total_time if total_time > 0 else 0
        
        print(f"\n\n{'=' * 80}")
        print(f"Generation completed!")
        print(f"Tokens generated: {token_count}")
        print(f"Time taken: {total_time:.2f} seconds")
        print(f"Average speed: {tokens_per_sec:.2f} tokens/second")
        print(f"Context length used: {len(model.tokenizer.encode(generated_text))}")
        print(f"{'=' * 80}")
        
    except Exception as e:
        print(f"\nError during generation: {e}")
        print(f"Tokens generated before error: {token_count}")
        elapsed = time.time() - start_time
        if elapsed > 0:
            print(f"Speed before error: {token_count/elapsed:.2f} tokens/second")

def test_memory_efficiency():
    """Test memory usage during long generation"""
    print("\n" + "=" * 80)
    print("Memory Efficiency Test")
    print("=" * 80)
    
    import psutil
    import os
    
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    print(f"Initial memory usage: {initial_memory:.2f} MB")
    
    # Test with different context lengths
    context_lengths = [2048, 4096, 8192]
    
    for ctx_len in context_lengths:
        print(f"\nTesting context length: {ctx_len}")
        
        try:
            model = SimplestLLM("meta-llama/Llama-3.2-1B-Instruct", context_length=ctx_len)
            
            current_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = current_memory - initial_memory
            
            print(f"Memory usage with {ctx_len} context: {current_memory:.2f} MB (+{memory_increase:.2f} MB)")
            
            # Quick generation test
            start_time = time.time()
            token_count = 0
            for token in model.generate("The future of AI is", max_new_tokens=100, temperature=0.7):
                token_count += 1
            
            elapsed = time.time() - start_time
            speed = token_count / elapsed if elapsed > 0 else 0
            
            print(f"Generation speed: {speed:.2f} tokens/second")
            
            # Clean up
            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
        except Exception as e:
            print(f"Error with context length {ctx_len}: {e}")

if __name__ == "__main__":
    test_focused_long_generation()
    test_memory_efficiency() 