# Evaluating Self-Consistency on GSM8K Dataset

This guide explains how to evaluate the self-consistency method on the real **GSM8K** (Grade School Math 8K) dataset used in the paper.

---

## 📊 What is GSM8K?

**GSM8K** is a dataset of 8,500 high-quality grade school math word problems:
- **7,473** training examples
- **1,319** test examples
- Each problem requires 2-8 steps to solve
- Created by OpenAI (Cobbe et al., 2021)

**Example:**
```
Question: "Janet's ducks lay 16 eggs per day. She eats three for breakfast 
every morning and bakes muffins for her friends every day with four. She 
sells the remainder for $2 per egg. How much does she make every day?"

Answer: "#### 18"
```

---

## 🚀 Quick Start (5 Examples)

Test on just 5 examples first:

```bash
python quick_gsm8k_test.py
```

This will:
- Download GSM8K dataset automatically
- Test on 5 random examples
- Use 10 samples per question (fast)
- Save results to `quick_test_results.json`

**Estimated time:** 2-3 minutes  
**Estimated cost:** ~$0.01 (GPT-3.5-Turbo)

---

## 📈 Full Evaluation

### Option 1: GPT-3.5-Turbo (Budget-Friendly)

```bash
python evaluate_gsm8k.py --model gpt-3.5-turbo --samples 40
```

**What this does:**
- Evaluates on all 1,319 test examples
- Uses 40 samples per question (as in paper)
- Uses GPT-3.5-Turbo

**Estimated time:** 2-4 hours  
**Estimated cost:** ~$25-30

### Option 2: GPT-4 (Best Accuracy)

```bash
python evaluate_gsm8k.py --model gpt-4 --samples 40
```

**Estimated time:** 8-12 hours  
**Estimated cost:** ~$1,600-2,000 (expensive!)

### Option 3: GPT-4-Turbo (Balanced)

```bash
python evaluate_gsm8k.py --model gpt-4-turbo --samples 40
```

**Estimated time:** 4-6 hours  
**Estimated cost:** ~$650-750

---

## 💰 Cost Optimization

### Test on Subset First

```bash
# Test on 100 questions
python evaluate_gsm8k.py --max-questions 100 --samples 40

# Test on 50 questions
python evaluate_gsm8k.py --max-questions 50 --samples 40
```

### Use Fewer Samples

```bash
# 20 samples instead of 40 (2x faster, 2x cheaper)
python evaluate_gsm8k.py --samples 20

# 10 samples (4x faster, 4x cheaper)
python evaluate_gsm8k.py --samples 10
```

### Recommended Development Strategy

1. **Quick test** (5 questions, 10 samples): `python quick_gsm8k_test.py`
2. **Small eval** (50 questions, 40 samples): `--max-questions 50`
3. **Medium eval** (200 questions, 40 samples): `--max-questions 200`
4. **Full eval** (all 1,319 questions, 40 samples): Run overnight

---

## 🎯 Expected Results

### Paper Results (for comparison)

| Model | Greedy | Self-Consistency (40) | Improvement |
|-------|--------|----------------------|-------------|
| PaLM-540B | 56.5% | 74.4% | **+17.9%** |
| GPT-3 (175B) | 60.1% | 78.0% | **+17.9%** |

### Expected Results with Our Implementation

**GPT-3.5-Turbo:**
- Greedy: ~60-65%
- Self-Consistency (40): ~75-80%
- Improvement: ~+15-20%

**GPT-4:**
- Greedy: ~85-90%
- Self-Consistency (40): ~92-95%
- Improvement: ~+5-10%

**GPT-4-Turbo:**
- Greedy: ~80-85%
- Self-Consistency (40): ~88-92%
- Improvement: ~+8-12%

---

## ⚙️ All Command-Line Options

```bash
python evaluate_gsm8k.py \
  --model gpt-3.5-turbo \      # Model to use
  --samples 40 \               # Number of reasoning paths
  --temperature 0.7 \          # Sampling temperature
  --split test \               # 'train' or 'test'
  --max-questions 100 \        # Limit number of questions
  --output results.json        # Output file name
```

---

## 📝 Understanding the Output

### During Evaluation

```
===========================================================
Question 1/1319
===========================================================
Q: Janet's ducks lay 16 eggs per day...
Ground Truth: 18
-----------------------------------------------------------
Sampling 40 reasoning paths...
  Generated 10/40 paths...
  Generated 20/40 paths...
  Generated 30/40 paths...
  Generated 40/40 paths...

Extracting answers...
Aggregating answers...

============================================================
RESULTS:
============================================================
Final Answer: 18
Consistency Score: 85.0%

Answer Distribution:
  18: 34 (85.0%)
  26: 4 (10.0%)
  14: 2 (5.0%)
============================================================

✓ CORRECT
Predicted: 18 (normalized: 18)
Ground Truth: 18
Consistency: 85.0%
```

### Final Results

```
===========================================================
FINAL RESULTS
===========================================================
Correct: 998/1319
Accuracy: 75.7%
===========================================================
```

### Results File (JSON)

The evaluation saves detailed results to a JSON file:

```json
{
  "dataset": "GSM8K",
  "split": "test",
  "model": "gpt-3.5-turbo",
  "num_samples": 40,
  "temperature": 0.7,
  "total_questions": 1319,
  "correct": 998,
  "accuracy": 0.757,
  "detailed_results": [
    {
      "question_id": 0,
      "question": "Janet's ducks lay 16 eggs...",
      "ground_truth": "18",
      "predicted": "$18",
      "predicted_normalized": "18",
      "correct": true,
      "consistency_score": 0.85,
      "answer_distribution": {"18": 34, "26": 4, "14": 2}
    },
    ...
  ]
}
```

---

## 🔍 Analyzing Results

### Check Overall Accuracy

```python
import json

with open('gsm8k_results.json', 'r') as f:
    results = json.load(f)

print(f"Accuracy: {results['accuracy']:.1%}")
print(f"Correct: {results['correct']}/{results['total_questions']}")
```

### Find Low Confidence Predictions

```python
low_confidence = [
    r for r in results['detailed_results'] 
    if r.get('consistency_score', 0) < 0.6
]

print(f"Low confidence questions: {len(low_confidence)}")
for r in low_confidence[:5]:
    print(f"\nQ: {r['question'][:80]}...")
    print(f"Predicted: {r['predicted']}")
    print(f"Consistency: {r['consistency_score']:.1%}")
```

### Analyze Errors

```python
errors = [
    r for r in results['detailed_results']
    if not r.get('correct', False)
]

print(f"Total errors: {len(errors)}")
print(f"\nSample errors:")
for r in errors[:3]:
    print(f"\nQ: {r['question'][:80]}...")
    print(f"Ground Truth: {r['ground_truth']}")
    print(f"Predicted: {r['predicted']}")
    print(f"Consistency: {r.get('consistency_score', 0):.1%}")
```

---

## 🐛 Troubleshooting

### Issue: "Out of quota"

**Solution:** You need to add credits to your OpenAI account
- Go to https://platform.openai.com/account/billing
- Add payment method and credits
- For full evaluation, budget ~$30-50 for GPT-3.5

### Issue: Rate limit errors

**Solution:** Add delays between API calls

Edit `self_consistency.py` and add sleep:
```python
import time

for i in range(num_samples):
    time.sleep(0.5)  # 500ms delay
    # ... make API call
```

### Issue: Taking too long

**Solutions:**
1. Use fewer samples: `--samples 20`
2. Test on subset: `--max-questions 100`
3. Use GPT-3.5 instead of GPT-4
4. Run overnight

### Issue: ModuleNotFoundError: datasets

**Solution:** Install the datasets library
```bash
pip install datasets
```

---

## 📊 Comparing Different Configurations

### Test Different Sample Counts

```bash
# Test with different numbers of samples
python evaluate_gsm8k.py --max-questions 100 --samples 10 --output results_s10.json
python evaluate_gsm8k.py --max-questions 100 --samples 20 --output results_s20.json
python evaluate_gsm8k.py --max-questions 100 --samples 40 --output results_s40.json
python evaluate_gsm8k.py --max-questions 100 --samples 80 --output results_s80.json
```

### Test Different Models

```bash
# Compare models on same subset
python evaluate_gsm8k.py --max-questions 100 --model gpt-3.5-turbo --output gpt35.json
python evaluate_gsm8k.py --max-questions 100 --model gpt-4-turbo --output gpt4t.json
python evaluate_gsm8k.py --max-questions 100 --model gpt-4 --output gpt4.json
```

### Test Different Temperatures

```bash
# Compare temperatures
python evaluate_gsm8k.py --max-questions 100 --temperature 0.5 --output temp_05.json
python evaluate_gsm8k.py --max-questions 100 --temperature 0.7 --output temp_07.json
python evaluate_gsm8k.py --max-questions 100 --temperature 0.9 --output temp_09.json
```

---

## 🎓 Research Tips

### For Academic Papers

1. **Run full evaluation** with `--samples 40` (as in paper)
2. **Report confidence intervals** by running multiple times
3. **Compare with baselines** (greedy decoding)
4. **Analyze failure cases** in detail
5. **Check consistency scores** as uncertainty metric

### For Production Use

1. **Start with subset** (100-200 examples)
2. **Use GPT-3.5** for cost efficiency
3. **Try fewer samples** (20-30) if cost is concern
4. **Monitor consistency scores** for confidence
5. **Set threshold** for high-confidence answers

---

## 📚 Additional Datasets

Want to test on other datasets? The same script can be adapted for:

- **SVAMP** - Challenge set for math word problems
- **AQuA** - Algebraic word problems
- **MultiArith** - Multi-step arithmetic
- **AddSub** - Addition/subtraction problems

Let me know if you want scripts for these!

---

## ✅ Quick Reference

**Quick test (5 examples):**
```bash
python quick_gsm8k_test.py
```

**Small eval (100 examples):**
```bash
python evaluate_gsm8k.py --max-questions 100
```

**Full eval (cheap):**
```bash
python evaluate_gsm8k.py --model gpt-3.5-turbo --samples 40
```

**Full eval (best quality):**
```bash
python evaluate_gsm8k.py --model gpt-4 --samples 40
```

---

**Ready to evaluate?** Start with `python quick_gsm8k_test.py`!
