#!/usr/bin/env python3
"""
Test script for long generation with SimplestLLM
Tests generation capacity up to 8192 tokens
"""

import time
from simplestLLM import SimplestLLM

def test_long_generation():
    print("=" * 80)
    print("Testing SimplestLLM Long Generation Capacity")
    print("=" * 80)
    
    # Initialize model
    print("Initializing model...")
    start_time = time.time()
    model = SimplestLLM("meta-llama/Llama-3.2-1B-Instruct")
    init_time = time.time() - start_time
    print(f"Model initialized in {init_time:.2f} seconds")
    
    # Test prompts for different scenarios
    test_cases = [
        {
            "name": "Technical Article",
            "prompt": "Write a comprehensive technical article about artificial intelligence and machine learning:",
            "max_tokens": 2048,
            "temperature": 0.7,
            "top_k": 50
        },
        {
            "name": "Story Generation", 
            "prompt": "Once upon a time in a distant galaxy, there lived a young explorer who discovered",
            "max_tokens": 1536,
            "temperature": 0.8,
            "top_k": 40
        },
        {
            "name": "Code Documentation",
            "prompt": "# Python Deep Learning Framework\n\nThis is a comprehensive guide to building neural networks from scratch. Let's start with the basics:\n\n",
            "max_tokens": 2048,
            "temperature": 0.3,
            "top_k": 30
        },
        {
            "name": "Long Form Essay",
            "prompt": "The impact of technology on society has been profound and multifaceted. In this essay, I will explore",
            "max_tokens": 2560,
            "temperature": 0.6,
            "top_k": 45
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'=' * 60}")
        print(f"TEST {i}: {test_case['name']}")
        print(f"Max tokens: {test_case['max_tokens']}")
        print(f"Temperature: {test_case['temperature']}, Top-k: {test_case['top_k']}")
        print(f"{'=' * 60}")
        
        print(f"\nPrompt: {test_case['prompt']}")
        print("\nGenerated text:")
        print("-" * 40)
        
        # Track generation stats
        start_time = time.time()
        token_count = 0
        
        try:
            print(test_case['prompt'], end="", flush=True)
            
            for token in model.generate(
                test_case['prompt'], 
                max_new_tokens=test_case['max_tokens'],
                temperature=test_case['temperature'],
                top_k=test_case['top_k']
            ):
                print(token, end="", flush=True)
                token_count += 1
                
                # Print progress every 100 tokens
                if token_count % 100 == 0:
                    elapsed = time.time() - start_time
                    tokens_per_sec = token_count / elapsed
                    print(f"\n[{token_count} tokens, {tokens_per_sec:.1f} tok/s]", end="", flush=True)
            
            # Final stats
            total_time = time.time() - start_time
            tokens_per_sec = token_count / total_time if total_time > 0 else 0
            
            print(f"\n\n{'=' * 40}")
            print(f"Generation completed!")
            print(f"Tokens generated: {token_count}")
            print(f"Time taken: {total_time:.2f} seconds")
            print(f"Speed: {tokens_per_sec:.2f} tokens/second")
            print(f"{'=' * 40}")
            
        except Exception as e:
            print(f"\nError during generation: {e}")
            print(f"Tokens generated before error: {token_count}")
        
        # Small pause between tests
        time.sleep(1)
    
    # Ultimate test - very long generation
    print(f"\n{'=' * 80}")
    print("ULTIMATE TEST: 8192 Token Generation")
    print(f"{'=' * 80}")
    
    ultimate_prompt = """The future of artificial intelligence represents one of the most significant technological frontiers of our time. As we stand at the threshold of unprecedented computational capabilities, we must examine both the extraordinary opportunities and the profound challenges that lie ahead. This comprehensive analysis will explore"""
    
    print(f"\nPrompt: {ultimate_prompt}")
    print("\nGenerating 8192 tokens...")
    print("-" * 80)
    
    start_time = time.time()
    token_count = 0
    
    try:
        print(ultimate_prompt, end="", flush=True)
        
        for token in model.generate(
            ultimate_prompt,
            max_new_tokens=8192,
            temperature=0.7,
            top_k=50
        ):
            print(token, end="", flush=True)
            token_count += 1
            
            # Print progress every 200 tokens for long generation
            if token_count % 200 == 0:
                elapsed = time.time() - start_time
                tokens_per_sec = token_count / elapsed
                print(f"\n[Progress: {token_count}/8192 tokens ({token_count/8192*100:.1f}%), Speed: {tokens_per_sec:.1f} tok/s]", end="", flush=True)
        
        # Final stats
        total_time = time.time() - start_time
        tokens_per_sec = token_count / total_time if total_time > 0 else 0
        
        print(f"\n\n{'=' * 80}")
        print(f"ULTIMATE TEST COMPLETED!")
        print(f"Tokens generated: {token_count}")
        print(f"Time taken: {total_time:.2f} seconds")
        print(f"Average speed: {tokens_per_sec:.2f} tokens/second")
        print(f"Memory efficiency: Context length handled successfully")
        print(f"{'=' * 80}")
        
    except Exception as e:
        print(f"\nError during ultimate test: {e}")
        print(f"Tokens generated before error: {token_count}")
        elapsed = time.time() - start_time
        if elapsed > 0:
            print(f"Speed before error: {token_count/elapsed:.2f} tokens/second")

if __name__ == "__main__":
    test_long_generation() 