"""
Evaluate Self-Consistency on GSM8K using Ollama (Llama 3.1)

"""

import json
import re
from typing import List, Dict
from self_consistency_ollama import SelfConsistencyOllama, create_arithmetic_prompt


def download_gsm8k():
    """Download GSM8K dataset from HuggingFace."""
    try:
        from datasets import load_dataset
        print("Downloading GSM8K dataset from HuggingFace...")
        dataset = load_dataset("gsm8k", "main")
        return dataset
    except ImportError:
        print("Installing datasets package...")
        import os
        os.system("pip install datasets")
        from datasets import load_dataset
        dataset = load_dataset("gsm8k", "main")
        return dataset


def extract_numerical_answer(answer_text: str) -> str:
    """Extract numerical answer from GSM8K format (#### 42)."""
    match = re.search(r'####\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)', answer_text)
    if match:
        return match.group(1).replace(',', '')
    return None


def normalize_answer(answer: str) -> str:
    """Normalize answer for comparison."""
    if answer is None:
        return None
    
    answer = answer.lower().strip()
    
    # Extract number
    patterns = [r'(-?\d+(?:,\d{3})*(?:\.\d+)?)']
    
    for pattern in patterns:
        match = re.search(pattern, answer)
        if match:
            num_str = match.group(1).replace(',', '')
            try:
                num = float(num_str)
                if num == int(num):
                    return str(int(num))
                return str(num)
            except:
                continue
    
    return answer.strip()


def evaluate_gsm8k_ollama(
    model_name: str = "llama3.1:8b",
    dataset_split: str = "test",
    num_samples: int = 40,
    max_questions: int = None,
    temperature: float = 0.7,
    verbose: bool = True
):
    """
    Evaluate self-consistency on GSM8K using Ollama.
    
    Args:
        model_name: Ollama model name (e.g., "llama3.1:8b")
        dataset_split: "train" or "test"
        num_samples: Number of reasoning paths to sample
        max_questions: Maximum number of questions (None = all)
        temperature: Sampling temperature
        verbose: Print progress
    
    Returns:
        Dictionary with evaluation results
    """
    # Initialize solver
    solver = SelfConsistencyOllama(
        model_name=model_name,
        temperature=temperature
    )
    
    # Load dataset
    dataset = download_gsm8k()
    data = dataset[dataset_split]
    
    if max_questions:
        data = data.select(range(min(max_questions, len(data))))
    
    print(f"\n{'='*70}")
    print(f"Evaluating on GSM8K {dataset_split} set")
    print(f"Total questions: {len(data)}")
    print(f"Samples per question: {num_samples}")
    print(f"Model: {model_name}")
    print(f"{'='*70}\n")
    
    results = []
    correct = 0
    total = 0
    
    for i, example in enumerate(data):
        question = example['question']
        ground_truth_text = example['answer']
        ground_truth = extract_numerical_answer(ground_truth_text)
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"Question {i+1}/{len(data)}")
            print(f"{'='*70}")
            print(f"Q: {question}")
            print(f"Ground Truth: {ground_truth}")
            print(f"{'-'*70}")
        
        # Create prompt and solve
        prompt = create_arithmetic_prompt(question)
        
        try:
            result = solver.solve(
                prompt, 
                num_samples=num_samples, 
                verbose=verbose
            )
            
            predicted = normalize_answer(result['final_answer'])
            ground_truth_norm = normalize_answer(ground_truth)
            
            is_correct = (predicted == ground_truth_norm)
            
            if is_correct:
                correct += 1
            
            total += 1
            
            result_entry = {
                'question_id': i,
                'question': question,
                'ground_truth': ground_truth,
                'predicted': result['final_answer'],
                'predicted_normalized': predicted,
                'correct': is_correct,
                'consistency_score': result['consistency_score'],
                'answer_distribution': result['answer_counts']
            }
            results.append(result_entry)
            
            if verbose:
                status = "✓ CORRECT" if is_correct else "✗ WRONG"
                print(f"\n{status}")
                print(f"Predicted: {result['final_answer']} (normalized: {predicted})")
                print(f"Ground Truth: {ground_truth}")
                print(f"Consistency: {result['consistency_score']:.1%}")
                
        except Exception as e:
            print(f"Error on question {i+1}: {e}")
            results.append({
                'question_id': i,
                'question': question,
                'ground_truth': ground_truth,
                'error': str(e)
            })
    
    # Calculate final metrics
    accuracy = correct / total if total > 0 else 0
    
    print(f"\n{'='*70}")
    print(f"FINAL RESULTS")
    print(f"{'='*70}")
    print(f"Correct: {correct}/{total}")
    print(f"Accuracy: {accuracy:.1%}")
    print(f"{'='*70}\n")
    
    # Save results
    results_summary = {
        'dataset': 'GSM8K',
        'split': dataset_split,
        'model': model_name,
        'num_samples': num_samples,
        'temperature': temperature,
        'total_questions': total,
        'correct': correct,
        'accuracy': accuracy,
        'detailed_results': results
    }
    
    return results_summary


def save_results(results: Dict, filename: str = "gsm8k_ollama_results.json"):
    """Save evaluation results to JSON file."""
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {filename}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate Self-Consistency on GSM8K using Ollama')
    parser.add_argument('--model', type=str, default='llama3.1:8b',
                        help='Ollama model name (default: llama3.1:8b)')
    parser.add_argument('--samples', type=int, default=40,
                        help='Number of reasoning paths (default: 40)')
    parser.add_argument('--max-questions', type=int, default=None,
                        help='Max questions to evaluate (default: all)')
    parser.add_argument('--temperature', type=float, default=0.7,
                        help='Sampling temperature (default: 0.7)')
    parser.add_argument('--split', type=str, default='test',
                        choices=['train', 'test'],
                        help='Dataset split (default: test)')
    parser.add_argument('--output', type=str, default='gsm8k_ollama_results.json',
                        help='Output file (default: gsm8k_ollama_results.json)')
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("GSM8K EVALUATION WITH OLLAMA")
    print("="*70 + "\n")
    
    # Evaluate
    results = evaluate_gsm8k_ollama(
        model_name=args.model,
        dataset_split=args.split,
        num_samples=args.samples,
        max_questions=args.max_questions,
        temperature=args.temperature,
        verbose=True
    )
    
    # Save results
    save_results(results, args.output)
    
    print(f"\n Evaluation complete!")
    print(f"Results saved to: {args.output}")

