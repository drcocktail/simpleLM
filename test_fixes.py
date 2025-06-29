#!/usr/bin/env python3

"""
Test script to verify:
1. 3B model loading works correctly
2. Italian text generation issue is resolved
"""

from fast_llm import FastLLM
import traceback

def test_1b_model():
    """Test 1B model for Italian text issue"""
    print("="*60)
    print("🔬 Testing 1B model for Italian text issue...")
    print("="*60)
    
    try:
        model = FastLLM("meta-llama/Llama-3.2-1B-Instruct", context_length=4096, use_float16=True)
        
        # Test simple math question
        prompt = "What is 53 + 27?"
        print(f"\nPrompt: {prompt}")
        print("Response: ", end="")
        
        response_tokens = []
        for token in model.generate(
            prompt, 
            max_new_tokens=30,
            temperature=0.7, 
            top_k=40,
            repetition_penalty=1.1,
            use_kv_cache=True
        ):
            print(token, end="", flush=True)
            response_tokens.append(token)
        
        response = "".join(response_tokens)
        
        # Check if response contains Italian text or weird formatting
        italian_indicators = ["<<", ">>", "Ecco", "risultati", "operazione", "matematica"]
        has_italian = any(indicator in response for indicator in italian_indicators)
        
        if has_italian:
            print(f"\n❌ FAILED: Response contains Italian text or weird formatting")
            return False
        else:
            print(f"\n✅ PASSED: No Italian text detected")
            return True
            
    except Exception as e:
        print(f"\n❌ FAILED: Exception occurred: {e}")
        traceback.print_exc()
        return False

def test_3b_model():
    """Test 3B model loading"""
    print("\n" + "="*60)
    print("🔬 Testing 3B model loading...")
    print("="*60)
    
    try:
        model = FastLLM("meta-llama/Llama-3.2-3B-Instruct", context_length=4096, use_float16=True)
        
        # Test simple generation
        prompt = "The capital of France is"
        print(f"\nPrompt: {prompt}")
        print("Response: ", end="")
        
        token_count = 0
        for token in model.generate(
            prompt, 
            max_new_tokens=10,
            temperature=0.0,  # Use greedy decoding for consistent results
            top_k=1,
            repetition_penalty=1.0,
            use_kv_cache=True
        ):
            print(token, end="", flush=True)
            token_count += 1
            if token_count >= 5:  # Just generate a few tokens to test
                break
        
        print(f"\n✅ PASSED: 3B model loaded and generated text successfully")
        return True
        
    except Exception as e:
        print(f"\n❌ FAILED: Exception occurred: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("🚀 Starting FastLLM fix verification tests...")
    
    # Test 1B model for Italian text issue
    test1_passed = test_1b_model()
    
    # Test 3B model loading
    test2_passed = test_3b_model()
    
    # Summary
    print("\n" + "="*60)
    print("📊 TEST SUMMARY")
    print("="*60)
    print(f"1B Model Italian Text Fix: {'✅ PASSED' if test1_passed else '❌ FAILED'}")
    print(f"3B Model Loading Fix:      {'✅ PASSED' if test2_passed else '❌ FAILED'}")
    
    if test1_passed and test2_passed:
        print("\n🎉 All tests passed! Both fixes are working correctly.")
    else:
        print("\n⚠️  Some tests failed. Please check the issues above.")
    
    return test1_passed and test2_passed

if __name__ == "__main__":
    main() 