import re
import requests
from collections import Counter
from typing import List, Dict, Tuple, Optional, Callable
import json


class SelfConsistencyOllama:
    """
    Self-Consistency decoding strategy using Ollama (local models).

    """
    
    def __init__(
        self,
        model_name: str = "llama3.1:8b",
        temperature: float = 0.7,
        max_tokens: int = 1024,
        ollama_host: str = "http://localhost:11434"
    ):
        """
        Initialize the Self-Consistency solver with Ollama.
        
        Args:
            model_name: Ollama model name (e.g., "llama3.1:8b", "deepseek-r1:7b")
            temperature: Sampling temperature (higher = more diverse)
            max_tokens: Maximum tokens to generate per sample
            ollama_host: Ollama server URL (default: localhost)
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.ollama_host = ollama_host
        
        # Test connection
        try:
            response = requests.get(f"{self.ollama_host}/api/tags")
            if response.status_code != 200:
                print(f"Warning: Cannot connect to Ollama at {self.ollama_host}")
                print("Make sure Ollama is running!")
        except Exception as e:
            print(f"Warning: Cannot connect to Ollama: {e}")
            print("Make sure Ollama is installed and running!")
    
    def sample_reasoning_paths(
        self,
        prompt: str,
        num_samples: int = 40
    ) -> List[str]:
        """
        Sample multiple diverse reasoning paths from Ollama.
        
        Args:
            prompt: The input prompt with CoT examples and question
            num_samples: Number of reasoning paths to sample
            
        Returns:
            List of generated reasoning paths (as strings)
        """
        reasoning_paths = []
        
        print(f"Sampling {num_samples} reasoning paths from {self.model_name}...")
        
        for i in range(num_samples):
            try:
                # Call Ollama API
                response = requests.post(
                    f"{self.ollama_host}/api/generate",
                    json={
                        "model": self.model_name,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": self.temperature,
                            "num_predict": self.max_tokens,
                        }
                    },
                    timeout=60  # 60 second timeout
                )
                
                if response.status_code == 200:
                    result = response.json()
                    response_text = result.get('response', '')
                    reasoning_paths.append(response_text)
                else:
                    print(f"  Error on path {i + 1}: HTTP {response.status_code}")
                    continue
                
                if (i + 1) % 10 == 0:
                    print(f"  Generated {i + 1}/{num_samples} paths...")
                    
            except requests.exceptions.Timeout:
                print(f"  Timeout on path {i + 1}, skipping...")
                continue
            except Exception as e:
                print(f"  Error generating path {i + 1}: {e}")
                continue
        
        print(f"Successfully generated {len(reasoning_paths)} reasoning paths.")
        return reasoning_paths
    
    def extract_answer(
        self,
        reasoning_path: str,
        answer_parser: Optional[Callable[[str], str]] = None
    ) -> Optional[str]:
        """
        Extract the final answer from a reasoning path.
        
        Args:
            reasoning_path: The generated reasoning text
            answer_parser: Optional custom parser function
            
        Returns:
            Extracted answer as string, or None if parsing failed
        """
        if answer_parser:
            return answer_parser(reasoning_path)
        
        # Default parser: look for "The answer is X" pattern
        patterns = [
            r"[Tt]he answer is[:\s]+([^\.\n]+)",
            r"[Tt]herefore[,\s]+([^\.\n]+)",
            r"[Ss]o[,\s]+the answer is[:\s]+([^\.\n]+)",
            r"[Tt]he final answer is[:\s]+([^\.\n]+)",
            r"= ([0-9]+)[\s]*$",  # Ends with "= 42"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, reasoning_path, re.IGNORECASE)
            if match:
                answer = match.group(1).strip()
                answer = answer.rstrip('.')
                answer = answer.strip()
                return answer
        
        # If no pattern found, try to get the last number
        numbers = re.findall(r'\b\d+\b', reasoning_path)
        if numbers:
            return numbers[-1]
        
        return None
    
    def aggregate_answers(
        self,
        answers: List[Optional[str]],
        normalize: bool = True
    ) -> Tuple[str, Dict[str, int], float]:
        """
        Aggregate answers using majority voting.
        
        Args:
            answers: List of extracted answers
            normalize: Whether to normalize answers before aggregation
            
        Returns:
            Tuple of (most_common_answer, answer_counts, consistency_score)
        """
        # Filter out None values
        valid_answers = [a for a in answers if a is not None]
        
        if not valid_answers:
            return None, {}, 0.0
        
        # Optional normalization
        if normalize:
            valid_answers = [a.lower().strip() for a in valid_answers]
        
        # Count occurrences
        answer_counts = Counter(valid_answers)
        
        # Get most common answer
        most_common_answer, max_count = answer_counts.most_common(1)[0]
        
        # Calculate consistency score
        consistency_score = max_count / len(valid_answers)
        
        return most_common_answer, dict(answer_counts), consistency_score
    
    def solve(
        self,
        prompt: str,
        num_samples: int = 40,
        answer_parser: Optional[Callable[[str], str]] = None,
        verbose: bool = True
    ) -> Dict:
        """
        Solve a reasoning problem using self-consistency.
        
        Args:
            prompt: The CoT prompt with examples and question
            num_samples: Number of reasoning paths to sample
            answer_parser: Optional custom answer extraction function
            verbose: Whether to print detailed output
            
        Returns:
            Dictionary with results
        """
        # Step 1: Sample diverse reasoning paths
        reasoning_paths = self.sample_reasoning_paths(prompt, num_samples)
        
        # Step 2: Extract answers from each path
        if verbose:
            print("\nExtracting answers from reasoning paths...")
        
        answers = []
        for i, path in enumerate(reasoning_paths):
            answer = self.extract_answer(path, answer_parser)
            answers.append(answer)
            
            if verbose and (i + 1) % 10 == 0:
                print(f"  Extracted {i + 1}/{len(reasoning_paths)} answers...")
        
        # Step 3: Aggregate by majority vote
        if verbose:
            print("\nAggregating answers by majority vote...")
        
        final_answer, answer_counts, consistency_score = self.aggregate_answers(answers)
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"RESULTS:")
            print(f"{'='*60}")
            print(f"Final Answer: {final_answer}")
            print(f"Consistency Score: {consistency_score:.1%}")
            print(f"\nAnswer Distribution:")
            for answer, count in sorted(answer_counts.items(), key=lambda x: x[1], reverse=True):
                print(f"  {answer}: {count} ({count/len([a for a in answers if a])*100:.1f}%)")
            print(f"{'='*60}\n")
        
        return {
            "final_answer": final_answer,
            "answer_counts": answer_counts,
            "consistency_score": consistency_score,
            "reasoning_paths": reasoning_paths,
            "all_answers": answers
        }


def create_arithmetic_prompt(question: str) -> str:
    """
    Create a Chain-of-Thought prompt for arithmetic reasoning.
    Uses the 8-shot examples from the paper.
    """
    examples = """Q: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
A: We start with 15 trees. Later we have 21 trees. The difference must be the number of trees they planted. So, they must have planted 21 - 15 = 6 trees. The answer is 6.

Q: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
A: There are 3 cars in the parking lot already. 2 more arrive. Now there are 3 + 2 = 5 cars. The answer is 5.

Q: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
A: Leah had 32 chocolates and Leah's sister had 42. That means there were originally 32 + 42 = 74 chocolates. 35 have been eaten. So in total they still have 74 - 35 = 39 chocolates. The answer is 39.

Q: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?
A: Jason had 20 lollipops. Since he only has 12 now, he must have given the rest to Denny. The number of lollipops he has given to Denny must have been 20 - 12 = 8 lollipops. The answer is 8.

Q: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?
A: He has 5 toys. He got 2 from mom, so after that he has 5 + 2 = 7 toys. Then he got 2 more from dad, so in total he has 7 + 2 = 9 toys. The answer is 9.

Q: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?
A: There are 4 days from monday to thursday. 5 computers were added each day. That means in total 4 * 5 = 20 computers were added. There were 9 computers in the beginning, so now there are 9 + 20 = 29 computers. The answer is 29.

Q: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?
A: Michael initially had 58 balls. He lost 23 on Tuesday, so after that he has 58 - 23 = 35 balls. On Wednesday he lost 2 more so now he has 35 - 2 = 33 balls. The answer is 33.

Q: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?
A: She bought 5 bagels for $3 each. This means she spent 5 * $3 = $15 on the bagels. She had $23 in beginning, so now she has $23 - $15 = $8. The answer is 8.

Q: {question}
A:"""
    
    return examples.format(question=question)


# Example usage
if __name__ == "__main__":
    # Initialize solver with Llama 3.1 8B
    solver = SelfConsistencyOllama(
        model_name="llama3.1:8b",
        temperature=0.7
    )
    
    # Example problem
    question = "If you have 12 apples and give away 5, how many apples do you have?"
    
    prompt = create_arithmetic_prompt(question)
    
    # Solve using self-consistency with 10 samples (use 40 for full accuracy)
    results = solver.solve(
        prompt=prompt,
        num_samples=10,
        verbose=True
    )
    
    print(f"\nFinal Answer: {results['final_answer']}")
    print(f"Consistency: {results['consistency_score']:.1%}")
