"""
Quick Test - Ollama + Llama 3.1 Setup

Tests that everything is working correctly.
"""

from self_consistency_ollama import SelfConsistencyOllama, create_arithmetic_prompt


def test_ollama_connection():
    """Test that Ollama is running and accessible."""
    print("="*70)
    print("TEST 1: Ollama Connection")
    print("="*70)
    
    import requests
    try:
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code == 200:
            models = response.json().get('models', [])
            print("✓ Ollama is running!")
            print(f"\nInstalled models:")
            for model in models:
                print(f"  - {model['name']}")
            
            # Check if llama3.1:8b is installed
            llama_installed = any('llama3.1' in m['name'] for m in models)
            if llama_installed:
                print("\n✓ Llama 3.1 is installed!")
                return True
            else:
                print("\n✗ Llama 3.1 not found!")
                print("Run: ollama pull llama3.1:8b")
                return False
        else:
            print(f"✗ Ollama returned status code: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Cannot connect to Ollama: {e}")
        print("\nMake sure Ollama is installed and running!")
        print("Download from: https://ollama.ai")
        return False


def test_simple_query():
    """Test a simple query to Llama 3.1."""
    print("\n" + "="*70)
    print("TEST 2: Simple Query")
    print("="*70)
    
    try:
        solver = SelfConsistencyOllama(model_name="llama3.1:8b")
        
        question = "What is 2 + 2?"
        prompt = f"Q: {question}\nA:"
        
        print(f"Question: {question}")
        print("Generating 1 response...\n")
        
        paths = solver.sample_reasoning_paths(prompt, num_samples=1)
        
        if paths:
            print(f"✓ Response received!")
            print(f"Answer: {paths[0][:200]}")
            return True
        else:
            print("✗ No response received")
            return False
            
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_self_consistency():
    """Test self-consistency with a simple math problem."""
    print("\n" + "="*70)
    print("TEST 3: Self-Consistency (5 samples)")
    print("="*70)
    
    try:
        solver = SelfConsistencyOllama(model_name="llama3.1:8b")
        
        question = "If you have 8 apples and buy 3 more, how many apples do you have?"
        prompt = create_arithmetic_prompt(question)
        
        print(f"Question: {question}")
        print("Expected: 11")
        print("\nRunning self-consistency with 5 samples...\n")
        
        results = solver.solve(prompt, num_samples=5, verbose=True)
        
        print(f"✓ Self-consistency completed!")
        print(f"Final Answer: {results['final_answer']}")
        print(f"Consistency: {results['consistency_score']:.0%}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("OLLAMA + LLAMA 3.1 - QUICK TEST")
    print("="*70 + "\n")
    
    tests_passed = 0
    tests_total = 3
    
    # Test 1: Connection
    if test_ollama_connection():
        tests_passed += 1
    else:
        print("\n⚠️ Ollama connection failed. Fix this first!")
        return
    
    # Test 2: Simple query
    if test_simple_query():
        tests_passed += 1
    
    # Test 3: Self-consistency
    if test_self_consistency():
        tests_passed += 1
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Passed: {tests_passed}/{tests_total}")
    
    if tests_passed == tests_total:
        print("\n✅ All tests passed! You're ready to go!")
        print("\nNext steps:")
        print("  1. Try: python demo_ollama.py")
        print("  2. Evaluate on 5 GSM8K examples: python quick_gsm8k_test_ollama.py")
        print("  3. Full GSM8K eval: python evaluate_gsm8k_ollama.py --max-questions 100")
    else:
        print("\n⚠️ Some tests failed. Check the errors above.")
    
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
