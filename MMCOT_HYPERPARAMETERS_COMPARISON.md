# MM-CoT Implementation Comparison: Original vs Your Implementation

## Summary of Your Questions

1. **Is your prompt strategy the same as original MM-CoT?**
2. **Are your hyperparameters the same as original MM-CoT?**

---

## Original MM-CoT Paper (Zhang et al., 2023)

### Framework Architecture: TWO-STAGE

#### Stage 1: Rationale Generation
- **Input:** Question text + Image features (from ViT-large encoder)
- **Output:** Reasoning rationale (chain-of-thought)
- **Length:** 512 tokens max

#### Stage 2: Answer Inference
- **Input:** Question text + Generated rationale + Image features
- **Output:** Final answer
- **Length:** 64 tokens max

### Hyperparameters (Official GitHub: amazon-science/mm-cot)

#### Rationale Generation Stage
| Parameter | UnifiedQA-Base | UnifiedQA-Large |
|-----------|----------------|-----------------|
| Learning Rate | 8e-5 | 5e-5 |
| Batch Size | 8 | 2 |
| Eval Batch Size | 8 | 4 |
| Epochs | 20 | 50 |
| Output Length | 512 | 512 |
| Input Length | 512 | 512 |

#### Answer Inference Stage
| Parameter | UnifiedQA-Base | UnifiedQA-Large |
|-----------|----------------|-----------------|
| Learning Rate | 8e-5 | 5e-5 |
| Batch Size | 8 | 4 |
| Eval Batch Size | 8 | 8 |
| Epochs | 20 | 50 |
| Output Length | 64 | 64 |
| Input Length | 512 | 512 |

### Model Configuration
- **Base Model:** UnifiedQA (T5-based)
- **Vision Encoder:** ViT-large (frozen)
- **Model Size:** Base (< 1B params) or Large
- **Training Method:** Fine-tuning on ScienceQA
- **Hardware:** 8× NVIDIA Tesla V100 32GB GPUs

### Prompt Strategy (from Paper)
The original MM-CoT uses:
- **Input Format:** `<context> [image features] <question> <choices>`
- **Two-stage prompting:**
  - Stage 1: Generate rationale without seeing answer choices
  - Stage 2: Use rationale + question to predict answer
- **No explicit few-shot examples** (model is fine-tuned)

### Generation Parameters
- **Temperature:** Not explicitly mentioned (likely default ~1.0 for training)
- **Inference:** Greedy decoding (no sampling mentioned)

### Results on ScienceQA
- **Overall Accuracy:** 91.68%
- **With Image:** ~91.5%
- **Without Image:** ~91.8%

---

## Your Implementation (Qwen 2.5 VL + Self-Consistency)

### Framework Architecture: NEEDS CLARIFICATION

**Question for you:** Did you implement:
- ✅ **Two-stage MM-CoT** (rationale generation → answer inference)?
- ✅ **Single-stage with Self-Consistency** (generate multiple answers, vote)?
- ✅ **Hybrid** (two-stage MM-CoT + self-consistency voting)?

### Your Reported Results
- **Overall Accuracy:** 88.47% (3752/4241)
- **Average Consensus:** 92.22%
- **Samples:** 5 (mentioned in your results)
- **Processing Time:** 37.3s per question

### Your Hyperparameters: NEEDS DETAILS

**Please provide from your ScienceQA implementation:**

#### Model Configuration
- **Base Model:** Qwen 2.5 VL (which size? 2B, 7B, 72B?)
- **Vision Encoder:** (Built into Qwen 2.5 VL)
- **Training Method:** Fine-tuned? Zero-shot? Few-shot prompting?

#### Generation Parameters
- **Temperature:** ?
- **Top-p / Top-k:** ?
- **Max Tokens:** ?
- **Num Samples (Self-Consistency):** 5 (confirmed from results)

#### Training Parameters (if fine-tuned)
- **Learning Rate:** ?
- **Batch Size:** ?
- **Epochs:** ?
- **Dataset:** ScienceQA train split?

### Your Prompt Strategy: NEEDS DETAILS

**Please share:**
1. What prompts do you use for rationale generation?
2. What prompts do you use for answer inference?
3. Do you use few-shot examples? How many?
4. What is your exact input format?

Example questions:
- Do you use "Let's think step by step"?
- Do you provide example CoT reasoning?
- How do you format the image + text input for Qwen 2.5 VL?

---

## Comparison Matrix (Partial - Awaiting Your Details)

| Aspect | Original MM-CoT | Your Implementation |
|--------|-----------------|---------------------|
| **Framework** | Two-stage (separate rationale & answer) | ❓ NEEDS CLARIFICATION |
| **Base Model** | UnifiedQA-Base (<1B) | Qwen 2.5 VL (❓ size?) |
| **Vision Encoder** | ViT-large (frozen) | Built-in (Qwen VL) |
| **Training** | Fine-tuned on ScienceQA | ❓ Fine-tuned? Prompting? |
| **Inference** | Greedy decoding (single pass) | Self-Consistency (5 samples) |
| **Temperature** | ~1.0 (default, not specified) | ❓ NEEDS INFO |
| **Learning Rate** | 5e-5 (large) / 8e-5 (base) | ❓ (if fine-tuned) |
| **Batch Size** | 2-8 | ❓ (if fine-tuned) |
| **Epochs** | 20-50 | ❓ (if fine-tuned) |
| **Few-shot Examples** | No (fine-tuned model) | ❓ NEEDS INFO |
| **Output Length** | 512 (rationale) + 64 (answer) | ❓ NEEDS INFO |
| **Accuracy** | 91.68% | 88.47% |
| **Consensus Score** | N/A (single inference) | 92.22% |

---

## Key Differences We Know So Far

### 1. Self-Consistency Addition
- **Original MM-CoT:** Single inference per question
- **Your Method:** 5 samples per question + majority voting
- **Impact:** Higher computational cost (5×) but better reliability (92.22% consensus)

### 2. Model Architecture
- **Original:** T5-based UnifiedQA (encoder-decoder)
- **Yours:** Qwen 2.5 VL (decoder-only with vision)
- **Impact:** Different capacity for multimodal reasoning

### 3. Training vs Prompting
- **Original:** Fine-tuned on ScienceQA training set
- **Yours:** ❓ Need to know if you fine-tuned or used prompting

---

## Questions for You

To complete this comparison, please answer:

### Critical Questions:
1. **Did you fine-tune Qwen 2.5 VL on ScienceQA, or use zero/few-shot prompting?**
2. **What is your exact prompt template?** (Please share code snippet)
3. **Do you use a two-stage approach** (separate rationale generation + answer inference)?
4. **What temperature did you use for generation?**
5. **Which Qwen 2.5 VL model size** (2B, 7B, or 72B)?

### Implementation Details:
6. What is your max output length?
7. Do you use few-shot examples in your prompts? How many?
8. How do you format multimodal inputs (text + image)?
9. What is your answer extraction method?
10. Do you use any special prompting techniques (e.g., "Let's think step by step")?

### Code Location:
Please point me to the specific file in your ScienceQA repository:
- Prompt construction code
- Model inference code
- Hyperparameter configuration

---

## What We CAN Compare (Based on Available Info)

### Similar Aspects:
✅ Both use multimodal models (vision + language)
✅ Both evaluate on ScienceQA test set (4,241 questions)
✅ Both report breakdown by image presence and subject

### Different Aspects:
❌ **Accuracy:** 91.68% (MM-CoT) vs 88.47% (Yours) = -3.21%
❌ **Model:** UnifiedQA vs Qwen 2.5 VL
❌ **Inference:** Single vs Self-Consistency (5 samples)

### Unknown Aspects (Need Your Input):
❓ Prompt strategy comparison
❓ Hyperparameter comparison
❓ Two-stage vs single-stage approach
❓ Training vs prompting methodology

---

## Next Steps

1. **Share your ScienceQA implementation code** so I can analyze:
   - Exact prompts used
   - Model configuration
   - Hyperparameters
   - Framework architecture (one-stage vs two-stage)

2. **Clarify your methodology:**
   - Fine-tuning or prompting?
   - Two-stage MM-CoT or direct prediction?
   - How self-consistency is integrated

3. **Then I can provide:**
   - ✅ Complete prompt strategy comparison
   - ✅ Complete hyperparameter comparison
   - ✅ Detailed analysis of differences
   - ✅ Recommendations for improvement

---

## Preliminary Answer to Your Questions

### Q1: "Is my prompt strategy the same as MM-CoT?"
**Answer:** ❓ **Cannot determine without seeing your code.**

Original MM-CoT uses fine-tuned models (no explicit prompts), while your method likely uses prompting with Qwen 2.5 VL. Need to see your prompts to compare.

### Q2: "Are my hyperparameters the same as MM-CoT?"
**Answer:** ❓ **Cannot determine without your configuration.**

However, likely **different** because:
- Original MM-CoT uses training hyperparameters (learning rate, epochs, batch size)
- Your method likely uses inference hyperparameters (temperature, top_p, num_samples=5)
- Different model architectures (UnifiedQA vs Qwen 2.5 VL)

**Action Required:** Please share your ScienceQA implementation details!

---

**Generated:** 2025-11-03
**Status:** INCOMPLETE - Awaiting implementation details from user
